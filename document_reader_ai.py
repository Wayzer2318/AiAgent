from langchain_core import embeddings
from langchain_core.messages.utils import _chunk_to_msg
import streamlit as st
import faiss
import numpy as np
import PyPDF2
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter, TextSplitter
from langchain.schema import Document


# load model
llm = OllamaLLM(model="mistral")
# load embedding model from HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# initialize faiss vector database
index = faiss.IndexFlatL2(384)
vector_store = {}


# extract from pdf func
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


# store text in faiss func
def store_in_faiss(text, filename):
    global index, vector_store
    st.write(f"storing Document '{filename}' in faiss ...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_ovelap=100)
    texts = splitter.split_text(text)
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)
    index.add(vectors)
    vector_store[len(vector_store)] = (filename, texts)
    return " document stored"


# retrive and answer func


def retrive_answer(query):
    global index, vector_store
