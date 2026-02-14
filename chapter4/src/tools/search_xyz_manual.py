from elasticsearch import Elasticsearch
from langchain.tools import tool
from pydantic import BaseModel, Field

from src.custom_logger import setup_logger
from src.models import SearchOutput

# Maximum number of search results to retrieve from Elasticsearch
MAX_SEARCH_RESULTS = 3

# Initialize a logger for this module to record runtime events and debugging information
logger = setup_logger(__file__)


# Define the input schema class used to validate tool arguments
class SearchKeywordInput(BaseModel):
    # Keyword string used for full-text search against stored documents
    keywords: str = Field(description="Keyword used for full-text search")


# Convert this function into a LangChain tool using the @tool decorator
# This allows LLM agents to call this function automatically when needed
@tool(args_schema=SearchKeywordInput)
def search_xyz_manual(keywords: str) -> list[SearchOutput]:
    """
    Function for searching documentation related to the XYZ system.

    This tool performs keyword-based full-text search on internal documentation.
    It should be used when user queries contain:
    - Error codes
    - Technical terms
    - System-specific terminology
    - Any identifiable keywords requiring manual lookup

    Args:
        keywords (str): Keyword string used to search the documentation.

    Returns:
        list[SearchOutput]: A list of structured search results extracted from matching documents.
    """

    # Log the start of the search operation with the provided keyword
    logger.info(f"Searching XYZ manual by keyword: {keywords}")

    # Create an Elasticsearch client instance and connect to the local Elasticsearch server
    es = Elasticsearch("http://localhost:9200")

    # Specify the index name to search within Elasticsearch
    index_name = "documents"

    # Construct a full-text search query targeting the 'content' field
    # This performs keyword matching against stored document content
    keyword_query = {"query": {"match": {"content": keywords}}}

    # Send the search query request to Elasticsearch and store the response
    response = es.search(index=index_name, body=keyword_query)

    # Log the number of search hits returned from Elasticsearch
    logger.info(f"Search results: {len(response['hits']['hits'])} hits")

    # Initialize a list to store structured search outputs
    outputs = []

    # Iterate through the search hits, limiting results to MAX_SEARCH_RESULTS
    for hit in response["hits"]["hits"][:MAX_SEARCH_RESULTS]:
        # Convert each raw Elasticsearch hit into a structured SearchOutput object
        # using the custom from_hit() class method
        outputs.append(SearchOutput.from_hit(hit))

    # Log completion of the search process
    logger.info("Finished searching XYZ manual by keyword")

    # Return the structured list of search results
    return outputs
