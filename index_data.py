import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Define paths
DATA_DIR = "data"
VECTOR_STORE_PATH = "vector_store.pkl"

# Load all documents
docs = []
for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(path)
        docs.extend(loader.load())

print(f"✅ Loaded {len(docs)} documents.")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"📚 Split into {len(chunks)} chunks.")

# Create embeddings using SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index + metadata
with open(VECTOR_STORE_PATH, "wb") as f:
    pickle.dump({"index": index, "texts": texts}, f)

print("🚀 Vector store saved to vector_store.pkl")
