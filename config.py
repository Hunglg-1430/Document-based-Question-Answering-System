import pinecone

def initialize_pinecone(api_key: str, environment: str):
    """
    Initialize Pinecone with the given API key and environment.

    Args:
    api_key (str): The API key for Pinecone.
    environment (str): The environment for Pinecone.

    Returns:
    None
    """
    pinecone.init(api_key=api_key, environment=environment)

def main():
    # Example usage
    api_key = "your_pinecone_api_key"
    environment = "your_pinecone_environment"
    initialize_pinecone(api_key, environment)

if __name__ == "__main__":
    main()
