from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_docs(directory: str):
    """
    Load documents from the specified directory.

    Args:
    directory (str): The directory to load documents from.

    Returns:
    List[Document]: A list of loaded documents.
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size: int = 1000, chunk_overlap: int = 20):
    """
    Split documents into chunks.

    Args:
    documents (List[Document]): A list of documents to split.
    chunk_size (int): The maximum size of each chunk.
    chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
    List[Document]: A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def main():
    # Example usage
    directory = '/content/data'
    documents = load_docs(directory)
    print(f"Number of documents loaded: {len(documents)}")

    chunk_size = 1000
    chunk_overlap = 20
    docs = split_docs(documents, chunk_size, chunk_overlap)
    print(f"Number of document chunks: {len(docs)}")

if __name__ == "__main__":
    main()
