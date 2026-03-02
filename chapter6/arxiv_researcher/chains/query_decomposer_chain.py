from datetime import datetime
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from arxiv_researcher.chains.task_evaluator_chain import TaskEvaluation
from arxiv_researcher.chains.utils import load_prompt
from arxiv_researcher.settings import settings


class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        min_length=settings.query_decomposer.min_decomposed_tasks,
        max_length=settings.query_decomposer.max_decomposed_tasks,
        description="List of decomposed tasks",
    )


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.min_decomposed_tasks = settings.query_decomposer.min_decomposed_tasks
        self.max_decomposed_tasks = settings.query_decomposer.max_decomposed_tasks

    def __call__(self, state: dict) -> Command[Literal["paper_search_agent"]]:
        evaluation: TaskEvaluation | None = state.get("evaluation", None)

        content = evaluation.content if evaluation else state.get("goal", "")
        decomposed_tasks: DecomposedTasks = self.run(content)

        return Command(
            goto="paper_search_agent",
            update={"tasks": decomposed_tasks.tasks},
        )

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(load_prompt("query_decomposer"))
        chain = prompt | self.llm.with_structured_output(
            DecomposedTasks,
            method="function_calling",
        )
        return chain.with_retry().invoke(
            {
                "min_decomposed_tasks": self.min_decomposed_tasks,
                "max_decomposed_tasks": self.max_decomposed_tasks,
                "current_date": self.current_date,
                "query": query,
            }
        )


if __name__ == "__main__":
    from arxiv_researcher.settings import settings

    decomposer = QueryDecomposer(settings.fast_llm)
    print(
        decomposer.run(
            "Please collect information on fact verification datasets in NLP from the following three perspectives: "
            "1. A general overview of the datasets and their contributions to fact verification "
            "2. Specific features and structure of representative datasets (e.g., FEVER, SQuAD) "
            "3. Practical use cases of these datasets and their impact on academia and industry"
        )
    )
