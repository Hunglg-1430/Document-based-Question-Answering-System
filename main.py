import os
import streamlit as st
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from constants import PINECONE_INDEX_NAME
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone

# Initialize API keys and environment for OpenAI and Pinecone
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)

# Streamlit application setup
st.title('Document Answering with Langchain and Pinecone')
user_query = st.text_input('What is your question?')

# Configure OpenAI LLM and embeddings
chat_model = ChatOpenAI(temperature=0.9, max_tokens=150, model='gpt-3.5-turbo-0613', client='')
embedding_model = OpenAIEmbeddings(client='')

# Initialize Pinecone index
document_search = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embedding_model)

# Load question-answering chain
qa_chain = load_qa_chain(chat_model)

# Process Streamlit input
def process_user_input(user_input):
    """
    Process the user input, perform similarity search and generate response using the LLM.

    :param user_input: The question input by the user
    """
    try:
        # Perform similarity search
        search_results = document_search.similarity_search(user_input)

        # Generate response using the QA chain
        generated_response = qa_chain.run(input_documents=search_results, question=user_input)

        print('Response:', generated_response)
        st.write(generated_response)

        # Display similarity search results
        with st.expander('Document Similarity Search'):
            print('Search results:', search_results)
            st.write(search_results)
    except Exception as error:
        st.write('It looks like you entered an invalid prompt. Please try again.')
        print(error)

# Check for user input and process it
if user_query:
    process_user_input(user_query)
