# Document Processing and Q&A System using OpenAI and Pinecone

This project processes documents and performs question-answering using OpenAI and Pinecone API.

## Repository Structure:

- `document_processing.py`: The main script for loading a PDF, dividing its content, generating embeddings with OpenAI, and storing them in Pinecone.
- `constants.py`: Holds the constants used across the repository.
- `main.py`: A Streamlit application that enables querying of the embedded documents using a question-answering chain.

## Requirements:

1. Python 3+
2. Pinecone API key
3. OpenAI API key
4. Streamlit

## Setup:

1. **Install the Required Libraries**:

   - Install the required libraries using the following command:

     ```bash
     $ pip install -r requirements.txt
     ```

2. **Set Up Configuration**:

   - Replace the key in `config.py` with your key::

     ```python
     OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
     PINECONE_API_KEY = 'YOUR_PINECONE_API_KEY'
     PINECONE_API_ENVIRONMENT = 'YOUR_PINECONE_ENVIRONMENT'
     ```

3. **Run `document_processing.py`**:

   - This will load the provided PDF, split its content, generate embeddings, and save them to Pinecone.

     ```bash
     $ python document_processing.py
     ```

4. **Start the Streamlit**:

   - Use Streamlit to run the `main.py` script.

     ```bash
     $ streamlit run main.py
     ```

   - Once the application is running, you can enter questions related to the PDF content, and it will provide relevant answers using the created embeddings and the question-answering chain.
