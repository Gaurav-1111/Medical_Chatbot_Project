from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
LLM_API_KEY  = os.environ.get('LLM_API_KEY')

extracted_data = load_pdf_file(data= '/workspaces/Medical_Chatbot_Project/Data/')
text_chunks = text_split(extracted_data)
embeddings= download_hugging_face_embeddings()

pc= Pinecone(api_keys=PINECONE_API_KEY)

index_name= "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws',
        region="us-east-1"
    )
)


docsearch = PineconeVectorStore.from_documents(
    documents= text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
