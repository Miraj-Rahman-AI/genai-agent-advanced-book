import datetime
import urllib.parse
from typing import Optional

import cohere
import feedparser  # type: ignore
from arxiv_researcher.logger import logger
from arxiv_researcher.models import ArxivPaper
from arxiv_researcher.searcher.searcher import Searcher
from arxiv_researcher.settings import settings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

FIELD_SELECTOR_PROMPT = """\
Determine the arXiv categories that need to be searched based on the user's query.
Select one or more category names, separated by commas.
Reply only with the exact category names (e.g., cs.AI, math.CO).

User Query: {query}
""".strip()

DATE_SELECTOR_PROMPT = """\
Determine the time range to be retrieved based on the user's query and the current system time.
Use the format YYMM-YYMM (e.g., 2203-2402 for March 2022 to February 2024).
If no time range is specified, reply with "NONE".

Current Date: {current_date}
User Query: {query}
""".strip()

EXPAND_QUERY_PROMPT = """\
<system>
You are an expert in generating effective arXiv search queries from a given single subquery. Your role is to understand the academic context and create the optimal search query that can be used directly in the arXiv search system.

{feedback}
</system>

## Main Tasks

1. Analyze the provided subquery
2. Extract important keywords from the subquery
3. Construct an effective search query that can be used directly on arXiv using the extracted keywords

## Detailed Instructions

<instructions>
1. Carefully read the subquery and identify the main concepts and technical terms.
2. Select specific keywords appropriate for the academic context.
3. Consider synonyms and related terms as well.
4. Use arXiv search syntax appropriately to create an effective search query.
5. Use field specifiers when necessary so that the search results are properly narrowed down.
6. Ensure that the generated query can be directly copied and pasted into the arXiv search box.
</instructions>

## Important Rules

<rules>
1. The query must include one or two main keywords or phrases.
2. Avoid terms that are too general or non-academic.
3. Keep the search query within 20 characters.
4. Do not include extra spaces or quotation marks before or after the query.
5. Do not include explanations or reasoning; output only the pure search query.
6. Use at most two keywords.
7. Do not use OR search.
</rules>

## arXiv Search Syntax Hints

<arxiv_syntax>
- AND: Search for documents containing multiple terms (e.g., quantum AND computing)
- OR: Search for documents containing either term (e.g., neural OR quantum)
- Quotation marks: Phrase search (e.g., "quantum computing")
- Field specifiers: ti: (title), au: (author), abs: (abstract)
- Minus sign: Exclude specific terms (e.g., quantum -classical)
- Wildcard: Partial match search (e.g., neuro*)
</arxiv_syntax>

<keywords>
- Examples of research-oriented keywords: RL, Optimization, LLM, etc.
- When searching for survey papers, use the following keywords: Survey, Review
- When searching for datasets, use the following keyword: Benchmark
- If the paper title is known, search by the paper title
</keywords>

## Examples

<example>
Query: Retrieve information about recent advances in quantum computing.

arXiv Search Query:
ti:"quantum computing"
</example>

<example>
Query: Find the latest research on applications of deep reinforcement learning to financial markets.

arXiv Search Query:
"deep reinforcement learning" AND "financial markets"
</example>

## Input Format

<input_format>
Goal: {goal_setting}
Query: {query}
</input_format>

REMEMBER: You must follow the content of the rules tag.
""".strip()


class ArxivFields(BaseModel):
    values: list[str] = Field(
        description="The arXiv categories that need to be searched based on the user's query."
    )


class ArxivTimeRange(BaseModel):
    start: Optional[datetime.datetime] = Field(
        default=None,
        description="The start date of the time range to be retrieved.",
    )
    end: Optional[datetime.datetime] = Field(
        default=None, description="The end date of the time range to be retrieved."
    )

    @property
    def text(self) -> Optional[str]:
        if self.start and self.end:
            return f"{self.start.strftime('%Y%m%d')}+TO+{self.end.strftime('%Y%m%d')}"
        elif self.start:
            return f"{self.start.strftime('%Y%m%d')}+TO+LATEST"
        elif self.end:
            return f"EARLIEST+TO+{self.end.strftime('%Y%m%d')}"
        return None


class ArxivSearcher(Searcher):
    RELEVANCE_SCORE_THRESHOLD = 0.7

    def __init__(
        self,
        llm: ChatOpenAI,
        cohere_client: cohere.Client = settings.cohere_client,
        max_search_results: int = settings.arxiv_search_agent.max_search_results,
        max_papers: int = settings.arxiv_search_agent.max_papers,
        max_retries: int = settings.arxiv_search_agent.max_retries,
        debug: bool = settings.debug,
    ):
        self.llm = llm
        self.cohere_client = cohere_client
        self.current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.max_search_results = max_search_results
        self.max_papers = max_papers
        self.max_retries = max_retries
        self.debug = debug

    def _field_selector(self, query: str) -> ArxivFields:
        prompt = ChatPromptTemplate.from_template(FIELD_SELECTOR_PROMPT)
        chain = prompt | self.llm.with_structured_output(
            ArxivFields,
            method="function_calling",
        )
        return chain.invoke({"query": query})  # type: ignore

    def _date_selector(self, query: str) -> ArxivTimeRange:
        prompt = ChatPromptTemplate.from_template(DATE_SELECTOR_PROMPT)
        chain = prompt | self.llm.with_structured_output(
            ArxivTimeRange,
            method="function_calling",
        )
        return chain.invoke(
            {
                "current_date": self.current_date,
                "query": query,
            }
        )  # type: ignore

    def _expand_query(self, goal_setting: str, query: str, feedback: str = "") -> str:
        prompt = ChatPromptTemplate.from_template(EXPAND_QUERY_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {"goal_setting": goal_setting, "query": query, "feedback": feedback}
        )

    def run(self, goal_setting: str, query: str) -> list[ArxivPaper]:
        base_url = "https://export.arxiv.org/api/query?search_query="
        retry_count = 0
        feedback = ""
        papers = []

        while retry_count < self.max_retries:
            filterquery_str = ""

            arxiv_time_range: ArxivTimeRange = self._date_selector(query)
            query_filterdate = arxiv_time_range.text

            expanded_query = self._expand_query(goal_setting, query, feedback)

            search_query = (
                f"{filterquery_str} AND all:{expanded_query}"
                if filterquery_str
                else f"all:{expanded_query}"
            )
            encoded_search_query = urllib.parse.quote(search_query)

            full_url = f"{base_url}{encoded_search_query}&sortBy=relevance&max_results={self.max_search_results}"
            if query_filterdate:
                full_url += f"&submittedDate={query_filterdate}"
            logger.info(f"Searching for papers: {full_url}")

            feed = feedparser.parse(full_url)
            entries = feed.entries

            papers = [
                ArxivPaper(
                    id=entry.id.split("/")[-1].split("v")[0],
                    title=entry.title,
                    link=entry.link,
                    pdf_link=next(
                        (
                            link.href
                            for link in entry.links
                            if link.type == "application/pdf"
                        ),
                        "",
                    ),
                    abstract=entry.summary.replace("\n", " "),
                    published=datetime.datetime(*entry.published_parsed[:6]),
                    updated=datetime.datetime(*entry.updated_parsed[:6]),
                    version=int(entry.id.split("/")[-1].split("v")[-1]),
                    authors=[
                        author.get("name", "") for author in entry.get("authors", [])
                    ],
                    categories=[tag.get("term", "") for tag in entry.get("tags", [])],
                )
                for entry in entries
            ]

            if self.debug:
                logger.info(f"Found {len(papers)} papers.")

            if papers:
                logger.info("Papers found. Exiting retry loop.")
                break  # Exit the loop because results were found

            else:
                retry_count += 1
                if retry_count < self.max_retries:
                    feedback = "The search returned zero results. Please adjust the query to be more general or use related keywords."
                    logger.info(
                        f"No papers found. Retrying with adjusted query. Attempt {retry_count}/{self.max_retries}"
                    )
                else:
                    logger.info("Max retries reached. No results found.")
                    break  # Exit the loop because the maximum number of retries has been reached

        if papers:
            reranked = self.cohere_client.rerank(
                model=settings.model.cohere_rerank_model,
                query=f"{goal_setting}\n{query}",
                documents=[f"{paper.title}\n{paper.abstract}" for paper in papers],
                top_n=min(self.max_papers, len(papers)),
            )

            reranked_papers = []
            for result in reranked.results:
                paper = papers[result.index]
                paper.relevance_score = result.relevance_score
                reranked_papers.append(paper)

            # Return only results whose relevance score is above the threshold
            papers = [
                paper
                for paper in reranked_papers
                if paper.relevance_score is not None
                and paper.relevance_score >= self.RELEVANCE_SCORE_THRESHOLD
            ]

        return papers


def main():
    from arxiv_researcher.settings import settings

    searcher = ArxivSearcher(settings.llm, settings.cohere_client, debug=True)

    query = input("Enter your arXiv search query: ")
    results = searcher.run(goal_setting="", query=query)

    print(f"\nFound {len(results)} results:")
    for i, paper in enumerate(results, 1):
        print(f"\n{i}. Title: {paper.title}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   Summary: {paper.summary[:500]}...")
        print(f"   arXiv ID: {paper.id}")
        print(f"   PDF Link: {paper.pdf_link}")
        print(f"   Relevance Score: {paper.relevance_score:.4f}")


if __name__ == "__main__":
    main()
