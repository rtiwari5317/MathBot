import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


pdf_dir = "./Maths_Datasets"
all_documents = []

for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, file))
        all_documents.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(all_documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True}, encode_kwargs={"normalize_embeddings": True}, cache_folder=None)
# Add this argument if your embedding class supports it:
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", trust_remote_code=True, use_auth_token=True, verify_ssl=False)
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local("./Maths_Datasets/pdf_vectorstore")

print(" Vectorstore created.")

