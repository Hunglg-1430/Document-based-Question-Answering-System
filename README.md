# Document Processing and Q&A System

This project processes documents and performs question-answering using OpenAI and Pinecone.

## Setup

1. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the script:**

    ```bash
    python main.py
    ```

## Description

This project:
- Loads documents from a specified directory.
- Splits the documents into chunks.
- Embeds the chunks using OpenAI embeddings.
- Stores the embeddings in a Pinecone index.
- Performs similarity search on the embeddings.
- Uses a language model to answer questions based on the retrieved documents.
