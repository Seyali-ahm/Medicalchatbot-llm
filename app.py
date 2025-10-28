import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)

# Load FAISS index and documents
with open("vector_store.pkl", "rb") as f:
    store = pickle.load(f)

index = store["index"]
texts = store["texts"]

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load local LLaMA model (Ollama)
llm = OllamaLLM(model="llama3")


def retrieve_similar_chunks(query, k=4):
    """Retrieve top-k similar chunks from FAISS based on semantic similarity."""
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)
    return [texts[i] for i in indices[0]]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question") or (request.json.get("question") if request.is_json else None)

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    print(f"🧠 User asked: {user_question}")

    # Retrieve top matching text chunks
    context_chunks = retrieve_similar_chunks(user_question)
    context_text = "\n".join(context_chunks[:4])
    print(f"📚 Retrieved {len(context_chunks)} context chunks.")

    # Construct the prompt for LLaMA
    prompt = f"""
You are a helpful medical assistant.
Answer the question using the context below. If the answer isn’t clear, say so.

Context:
{context_text}

Question: {user_question}
Answer:
"""
    print("🚀 Sending prompt to LLaMA...")

    try:
        response = llm.invoke(prompt)
        print(f"✅ LLaMA response: {response}")
        return jsonify({"answer": response})
    except Exception as e:
        print(f"❌ Error from LLaMA: {e}")
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(debug=True)