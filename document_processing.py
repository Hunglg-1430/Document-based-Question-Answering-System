import os
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from constants import PINECONE_INDEX_NAME
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# Initialize API keys and environment for OpenAI and Pinecone
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)

# Ensure the Pinecone index exists, creating it if necessary
def initialize_index():
    """
    Initialize the Pinecone index. Create the index if it does not exist.
    """
    print('Verifying if the index exists...')
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        print('Index does not exist, creating a new index...')
        pinecone.create_index(
            name=PINECONE_INDEX_NAME,
            metric='cosine',
            dimension=1536
        )

# Load the PDF document
def load_pdf_document(file_path):
    """
    Load a PDF document using UnstructuredPDFLoader.

    :param file_path: Path to the PDF file to be loaded
    :return: Loaded PDF document data
    """
    print('Loading document...')
    loader = UnstructuredPDFLoader(file_path)
    document_data = loader.load()
    return document_data

# Split the document into smaller chunks
def split_document(data, chunk_size=1000, chunk_overlap=100):
    """
    Split the document data into smaller chunks.

    :param data: Document data to be split
    :param chunk_size: Size of each chunk
    :param chunk_overlap: Overlap between chunks
    :return: List of smaller document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_texts = text_splitter.split_documents(data)
    return split_texts

# Create embeddings and index from the document texts
def create_embeddings_and_index(texts, index_name):
    """
    Create embeddings from document texts and index them using Pinecone.

    :param texts: List of document texts
    :param index_name: Name of the Pinecone index
    """
    print('Creating embeddings and indexing...')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    print('Done!')

# Main execution
if __name__ == "__main__":
    initialize_index()

    document_path = 'data\sample.pdf'
    data = load_pdf_document(document_path)

    print(f'Loaded a PDF with {len(data)} pages')
    print(f'The document contains {len(data[0].page_content)} characters')

    texts = split_document(data)
    print(f'Split the document into {len(texts)} smaller chunks')

    create_embeddings_and_index(texts, PINECONE_INDEX_NAME)
