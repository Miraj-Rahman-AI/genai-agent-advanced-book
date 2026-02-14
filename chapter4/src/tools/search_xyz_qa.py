from langchain.tools import tool
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.configs import Settings
from src.custom_logger import setup_logger
from src.models import SearchOutput

# Maximum number of search results to retrieve from the vector database
MAX_SEARCH_RESULTS = 3

# Initialize a logger for this module to capture runtime logs and debugging information
logger = setup_logger(__name__)


# Define the input schema used to validate tool arguments
class SearchQueryInput(BaseModel):
    # The search query provided by the user for semantic similarity search
    query: str = Field(description="Search query")


# Convert this function into a LangChain-compatible tool
# This allows an LLM agent to call this function automatically when semantic search is needed
@tool(args_schema=SearchQueryInput)
def search_xyz_qa(query: str) -> list[SearchOutput]:
    """
    Function for searching past question-and-answer pairs related to the XYZ system.

    This tool performs semantic vector search over stored QA data.
    It converts the user's query into an embedding vector and retrieves
    the most relevant QA entries from the vector database (Qdrant).

    Args:
        query (str): The user’s natural language search query.

    Returns:
        list[SearchOutput]: A list of structured search results representing relevant QA entries.
    """

    # Log the start of the QA search process
    logger.info(f"Searching XYZ QA by query: {query}")

    # Initialize Qdrant client and connect to local Qdrant vector database
    qdrant_client = QdrantClient("http://localhost:6333")

    # Load application settings (contains OpenAI API configuration)
    settings = Settings()

    # Initialize OpenAI client using API key from settings
    openai_client = OpenAI(api_key=settings.openai_api_key)

    # Generate an embedding vector from the input query
    # This embedding will be used for semantic similarity search in Qdrant
    logger.info("Generating embedding vector from input query")
    query_vector = (
        openai_client.embeddings.create(input=query, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

    # Perform vector similarity search in Qdrant collection
    # Retrieve the top matching points based on cosine similarity
    search_results = qdrant_client.query_points(
        collection_name="documents",
        query=query_vector,
        limit=MAX_SEARCH_RESULTS
    ).points

    # Log number of search results retrieved
    logger.info(f"Search results: {len(search_results)} hits")

    # Initialize list to store structured search outputs
    outputs = []

    # Convert each returned Qdrant point into a structured SearchOutput object
    for point in search_results:
        outputs.append(SearchOutput.from_point(point))

    # Log completion of the QA search process
    logger.info("Finished searching XYZ QA by query")

    # Return structured search results
    return outputs
