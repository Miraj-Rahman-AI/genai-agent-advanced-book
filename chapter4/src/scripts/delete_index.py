from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient


def delete_es_index(es: Elasticsearch, index_name: str) -> None:
    """
    Delete an index from Elasticsearch if it exists.

    This function first checks whether the specified index exists
    in the Elasticsearch instance. If the index exists, it deletes it.
    Otherwise, it prints a message indicating that the index does not exist.

    Args:
        es (Elasticsearch): Elasticsearch client instance.
        index_name (str): Name of the index to be deleted.
    """

    # Check whether the specified Elasticsearch index exists
    if es.indices.exists(index=index_name):
        # Delete the index if it exists
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' has been deleted.")
    else:
        # Notify if the index does not exist
        print(f"Index '{index_name}' does not exist.")


def delete_qdrant_index(qdrant_client: QdrantClient, collection_name: str) -> None:
    """
    Delete a collection (vector index) from Qdrant if it exists.

    This function checks whether a specified collection exists
    in the Qdrant vector database. If it exists, the collection
    will be deleted. Otherwise, a message will be printed indicating
    that the collection does not exist.

    Args:
        qdrant_client (QdrantClient): Qdrant client instance.
        collection_name (str): Name of the collection to be deleted.
    """

    # Check whether the specified Qdrant collection exists
    if qdrant_client.collection_exists(collection_name=collection_name):
        # Delete the collection from Qdrant if it exists
        qdrant_client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' has been deleted.")
    else:
        # Notify if the collection does not exist
        print(f"Collection '{collection_name}' does not exist.")


if __name__ == "__main__":
    """
    Main execution block.

    This section initializes connections to both Elasticsearch and Qdrant,
    then deletes the specified keyword search index and vector search collection.
    """

    # Connect to local Elasticsearch instance
    es = Elasticsearch("http://localhost:9200")

    # Connect to local Qdrant instance
    qdrant_client = QdrantClient("http://localhost:6333")

    # Define index/collection name to delete
    index_name = "documents"

    # Delete Elasticsearch keyword search index
    delete_es_index(es=es, index_name=index_name)

    # Delete Qdrant vector search collection
    delete_qdrant_index(qdrant_client=qdrant_client, collection_name=index_name)
