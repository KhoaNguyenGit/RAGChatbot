from qdrant_client import QdrantClient

def checking_connection(host: str = "localhost", port: int = 6333) -> bool:
    """
    Check if Qdrant is reachable and responding.
    Returns True if connected successfully, otherwise False.
    """
    try:
        client = QdrantClient(host=host, port=port)
        response = client.get_collections()
        print(f"Connected to Qdrant. Found {len(response.collections)} collections.")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False