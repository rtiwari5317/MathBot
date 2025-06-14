
# utils/vectorstore_loader.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os



#Loading VectorStore:
def load_vectorstore(path, allow_dangerous_deserialization=False):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", trust_remote_code=True, use_auth_token=True, verify_ssl=False)
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=allow_dangerous_deserialization)