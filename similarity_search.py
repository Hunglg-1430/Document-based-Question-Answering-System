from langchain.vectorstores import Pinecone

def create_pinecone_index(docs, embeddings, index_name: str):
    """
    Create a Pinecone index from documents and embeddings.

    Args:
    docs (List[Document]): A list of document chunks.
    embeddings (OpenAIEmbeddings): The embeddings to use.
    index_name (str): The name of the Pinecone index.

    Returns:
    Pinecone: The created Pinecone index.
    """
    return Pinecone.from_documents(docs, embeddings, index_name=index_name)

def get_similar_docs(index, query: str, k: int = 2, score: bool = False):
    """
    Get similar documents to a query from the Pinecone index.

    Args:
    index (Pinecone): The Pinecone index to search.
    query (str): The query to search for.
    k (int): The number of similar documents to retrieve.
    score (bool): Whether to return similarity scores.

    Returns:
    List[Document]: A list of similar documents (and optionally scores).
    """
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def main():
    # Example usage
    from embeddings import create_embeddings
    from document_processing import load_docs, split_docs

    directory = '/content/data'
    documents = load_docs(directory)
    docs = split_docs(documents)

    embeddings = create_embeddings(model_name="ada")
    index_name = "langchain-demo"
    index = create_pinecone_index(docs, embeddings, index_name=index_name)

    query = "Hello world"
    similar_docs = get_similar_docs(index, query)
    print(f"Similar documents to query '{query}': {similar_docs}")

if __name__ == "__main__":
    main()
