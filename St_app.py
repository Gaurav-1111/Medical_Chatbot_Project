import os
import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import *

# Load environment variables
load_dotenv()

# Set API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone setup
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwags={"k": 3})

# LLM setup
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name='llama-3.3-70b-versatile'
)

# Prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit UI
st.title("💊 Medical Chatbot")
st.write("Ask me anything about medical topics!")

# User input
msg = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if msg:
        response = rag_chain.invoke({"input": msg})  # Fixed input format
        st.write("**Answer:**", response["answer"])
    else:
        st.warning("Please enter a question.")
