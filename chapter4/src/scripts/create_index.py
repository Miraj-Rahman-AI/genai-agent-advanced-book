import os
from glob import glob

from elasticsearch import Elasticsearch, helpers
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class Settings(BaseSettings):
    # Environment configuration class for securely loading API credentials and model settings
    openai_api_key: str
    openai_api_base: str
    openai_model: str

    # Load variables from .env file and ignore extra undefined fields
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def load_pdf_docs(data_dir_path: str) -> list[Document]:
    """
    Load and split all PDF documents from the specified directory recursively.

    This function:
    1. Searches for all PDF files under the given directory.
    2. Loads each PDF file using LangChain's PyPDFLoader.
    3. Splits text into small chunks using RecursiveCharacterTextSplitter.
    4. Returns a list of processed Document objects for downstream indexing/search.
    """
    pdf_path = glob(os.path.join(data_dir_path, "**", "*.pdf"), recursive=True)
    docs = []

    # Initialize text splitter to break large documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # Small chunk size for demonstration purposes
        chunk_overlap=20,    # Overlap between chunks to preserve context
        length_function=len,
        is_separator_regex=False,
    )

    for path in pdf_path:
        loader = PyPDFLoader(path)
        # Load and split the document into manageable chunks
        pages = loader.load_and_split(text_splitter)
        docs.extend(pages)

    return docs


def load_csv_docs(data_dir_path: str) -> list[Document]:
    """
    Load CSV files from the specified directory and convert them into Document objects.

    Each CSV file is loaded and converted into LangChain Document format,
    which can then be indexed into search systems such as Elasticsearch or Qdrant.
    """
    csv_path = glob(os.path.join(data_dir_path, "**", "*.csv"), recursive=True)
    docs = []

    for path in csv_path:
        loader = CSVLoader(file_path=path)
        docs.extend(loader.load())

    return docs


def create_keyword_search_index(es: Elasticsearch, index_name: str) -> None:
    """
    Create an Elasticsearch index configured for keyword-based full-text search,
    optimized for Japanese language processing using Kuromoji tokenizer.
    """

    # Define index mapping configuration
    mapping = {
        "mappings": {
            # Define field properties inside each indexed document
            "properties": {
                "content": {
                    # 'content' field is used for full-text search
                    "type": "text",
                    # Use custom Japanese analyzer for better tokenization and search accuracy
                    "analyzer": "kuromoji_analyzer",
                }
            },
        },
        "settings": {
            "analysis": {
                "analyzer": {
                    # Define custom analyzer named 'kuromoji_analyzer'
                    "kuromoji_analyzer": {
                        "type": "custom",
                        # Normalize characters using ICU normalization
                        "char_filter": ["icu_normalizer"],
                        # Use Kuromoji tokenizer for Japanese morphological analysis
                        "tokenizer": "kuromoji_tokenizer",
                        "filter": [
                            # Convert verbs/adjectives to base form
                            "kuromoji_baseform",
                            # Filter tokens based on part-of-speech
                            "kuromoji_part_of_speech",
                            # Remove Japanese stop words
                            "ja_stop",
                            # Normalize numbers
                            "kuromoji_number",
                            # Extract word stems for Japanese
                            "kuromoji_stemmer",
                        ],
                    }
                }
            }
        },
    }

    # Create index only if it does not already exist
    if not es.indices.exists(index=index_name):
        result = es.indices.create(index=index_name, body=mapping)
        if result:
            print(f"Index {index_name} created successfully")
        else:
            print(f"Failed to create index {index_name}")


def create_vector_search_index(qdrant_client: QdrantClient, index_name: str) -> None:
    """
    Create a Qdrant vector collection for semantic search using embeddings.

    The vector size 1536 corresponds to OpenAI embedding model output size.
    Cosine distance is used for similarity search.
    """
    result = qdrant_client.create_collection(
        collection_name=index_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    if result:
        print(f"Collection {index_name} created successfully")
    else:
        print(f"Failed to create collection {index_name}")


def add_documents_to_es(
    es: Elasticsearch, index_name: str, docs: list[Document]
) -> None:
    """
    Insert documents into Elasticsearch keyword search index.

    Each document is transformed into Elasticsearch bulk API format,
    including file name and textual content fields.
    """
    insert_docs = []

    for doc in docs:
        content = doc.page_content

        # Construct document structure for Elasticsearch indexing
        insert_doc = {
            "_index": index_name,
            "_source": {
                "file_name": os.path.basename(doc.metadata["source"]),
                "content": content,
            },
        }
        insert_docs.append(insert_doc)

    # Bulk insert documents into Elasticsearch
    helpers.bulk(es, insert_docs)


def add_documents_to_qdrant(
    qdrant_client: QdrantClient,
    index_name: str,
    docs: list[Document],
    settings: Settings,
) -> None:
    """
    Convert documents into vector embeddings and store them in Qdrant.

    Steps:
    1. Generate embeddings using OpenAI embedding model.
    2. Create vector points with payload metadata.
    3. Upload (upsert) vectors into Qdrant collection.
    """
    points = []
    client = OpenAI(api_key=settings.openai_api_key)

    for i, doc in enumerate(docs):
        content = doc.page_content

        # Remove spaces for cleaner embedding input (optional preprocessing)
        content = content.replace(" ", "")

        # Generate embedding vector using OpenAI embedding model
        embedding = client.embeddings.create(
            model="text-embedding-3-small", input=content
        )

        # Create Qdrant vector point structure
        points.append(
            PointStruct(
                id=i,
                vector=embedding.data[0].embedding,
                payload={
                    "file_name": os.path.basename(doc.metadata["source"]),
                    "content": content,
                },
            )
        )

    # Upload vectors to Qdrant collection
    operation_info = qdrant_client.upsert(
        collection_name=index_name,
        points=points,
        wait=True,
    )
    print(operation_info)


if __name__ == "__main__":
    # Initialize Elasticsearch and Qdrant clients
    es = Elasticsearch("http://localhost:9200")
    qdrant_client = QdrantClient("http://localhost:6333")

    # Load environment settings
    settings = Settings()

    index_name = "documents"

    # Create keyword-based search index in Elasticsearch
    print(f"Creating index for keyword search {index_name}")
    create_keyword_search_index(es, index_name)
    print("--------------------------------")

    # Create vector-based semantic search index in Qdrant
    print(f"Creating index for vector search {index_name}")
    create_vector_search_index(qdrant_client, index_name)
    print("--------------------------------")

    # Load manual PDF documents
    print("Loading documents from manual data")
    manual_docs = load_pdf_docs(data_dir_path="data")
    print(f"Loaded {len(manual_docs)} documents")

    print("--------------------------------")

    # Load Q&A CSV documents
    print("Loading documents from qa data")
    qa_docs = load_csv_docs(data_dir_path="data")
    print(f"Loaded {len(qa_docs)} documents")

    # Add documents to Elasticsearch keyword search index
    print("Adding documents to keyword search index")
    add_documents_to_es(es, index_name, manual_docs)
    print("--------------------------------")

    # Add documents to Qdrant vector search index
    print("Adding documents to vector search index")
    add_documents_to_qdrant(qdrant_client, index_name, qa_docs, settings)
    print("--------------------------------")

    print("Done")
