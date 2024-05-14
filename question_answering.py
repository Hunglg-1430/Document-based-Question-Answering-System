from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

def initialize_llm(model_name: str = "gpt-4"):
    """
    Initialize the language model.

    Args:
    model_name (str): The name of the OpenAI model to use.

    Returns:
    OpenAI: The initialized language model.
    """
    return OpenAI(model_name=model_name)

def load_qa(llm, chain_type: str = "stuff"):
    """
    Load the question answering chain.

    Args:
    llm (OpenAI): The language model to use.
    chain_type (str): The type of QA chain to use.

    Returns:
    Any: The loaded QA chain.
    """
    return load_qa_chain(llm, chain_type=chain_type)

def get_answer(chain, index, query: str):
    """
    Get an answer to the query using the QA chain and document index.

    Args:
    chain: The QA chain.
    index: The document index.
    query (str): The query to answer.

    Returns:
    str: The answer to the query.
    """
    similar_docs = index.similarity_search(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

def main():
    # Example usage
    model_name = "gpt-4"
    llm = initialize_llm(model_name=model_name)
    chain = load_qa(llm)
    # Assuming `index` and `query` are defined
    # answer = get_answer(chain, index, query)
    # print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
