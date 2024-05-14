from langchain.embeddings.openai import OpenAIEmbeddings

def create_embeddings(model_name: str = "ada"):
    """
    Create embeddings using OpenAI's model.

    Args:
    model_name (str): The name of the OpenAI model to use for embeddings.

    Returns:
    OpenAIEmbeddings: The embeddings object.
    """
    return OpenAIEmbeddings(model_name=model_name)

def main():
    # Example usage
    model_name = "ada"
    embeddings = create_embeddings(model_name)
    print(f"Created embeddings with model {model_name}")

if __name__ == "__main__":
    main()
