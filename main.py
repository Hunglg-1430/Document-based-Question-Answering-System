from config import initialize_pinecone
from document_processing import load_docs, split_docs
from embeddings import create_embeddings
from similarity_search import create_pinecone_index, get_similar_docs
from question_answering import initialize_llm, load_qa, get_answer

def main():
    # Configuration
    API_KEY = "your_openai_api_key"
    PINECONE_API_KEY = "your_pinecone_api_key"
    PINECONE_ENV = "your_pinecone_environment"
    DIRECTORY = '/content/data'
    INDEX_NAME = "langchain-demo"
    MODEL_NAME = "gpt-4"

    # Initialize Pinecone
    initialize_pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    # Load and process documents
    documents = load_docs(DIRECTORY)
    print(f"Number of documents loaded: {len(documents)}")

    docs = split_docs(documents)
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    embeddings = create_embeddings(model_name="ada")

    # Create Pinecone index
    index = create_pinecone_index(docs, embeddings, INDEX_NAME)

    # Initialize the language model
    llm = initialize_llm(model_name=MODEL_NAME)

    # Load the question answering chain
    chain = load_qa(llm)

    # Example queries
    query = "How is India's economy?"
    answer = get_answer(chain, index, query)
    print(f"Answer to query '{query}': {answer}")

    query = "How have relations between India and the US improved?"
    answer = get_answer(chain, index, query)
    print(f"Answer to query '{query}': {answer}")

if __name__ == "__main__":
    main()
