# 🩺 Medical Chatbot LLM (Offline, Llama-Based)

This project is a **medical question-answering chatbot** built with **Flask**, **LangChain**, and **Llama 3** (via Ollama).  
It can read and understand your **medical PDFs or documents**, store their embeddings locally, and then answer questions based on that knowledge — all **100% offline** and **free** to use (no API keys or cloud services needed).

---

## 🚀 Features

- 📚 **Upload & Index Medical Books:** Reads and splits your medical PDFs into chunks for embedding.  
- 🧠 **Offline LLM Integration:** Uses **Llama 3** through Ollama locally — no need for OpenAI or AWS.  
- 🗂 **Vector Store (Local):** All embeddings are stored locally in `vector_store.pkl`.  
- 💬 **Chat Interface:** A Flask-powered web UI for chatting with your offline AI medical assistant.  
- 🔒 **Completely Free:** No tokens, no billing, no internet dependency once setup.

---

## 🧩 Tech Stack

- **Backend:** Python, Flask  
- **LLM Engine:** Llama 3 via Ollama  
- **Framework:** LangChain  
- **Embeddings:** SentenceTransformer (local embeddings)  
- **Storage:** Local pickle-based vector database  
- **Frontend:** HTML, CSS, JS (Flask templates)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Seyali-ahm/Medicalchatbot-llm.git
cd Medicalchatbot-llm
```

---

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate     # On Windows
# or
source .venv/bin/activate    # On Mac/Linux
```

---

### 3️⃣ Install the Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Install and Run Ollama

Download and install **Ollama** from:  
👉 [https://ollama.ai](https://ollama.ai)

Then pull the **Llama 3** model locally:

```bash
ollama pull llama3
```

This will install the model used for answering your medical questions locally — no internet or API key required.

---

### 5️⃣ 📘 Index Your Medical Book

To teach the chatbot your medical data:

```bash
# Place your medical PDFs in the 'data/' folder
python index_data.py
```

This script:

- 📂 Loads your PDF files  
- ✂️ Splits them into smaller text chunks  
- 🧠 Generates embeddings locally using SentenceTransformer  
- 💾 Saves them to `vector_store.pkl` for future queries  

---

### 6️⃣ 💬 Run the Chatbot

Start the Flask web app:

```bash
python app.py
```

Then open your browser and visit:  
👉 **http://127.0.0.1:5000**

You can now chat with your **local, offline medical assistant** 🧠💬  

---

## 🧠 Notes

- Ensure **Ollama** is running in the background before launching the chatbot.  
- You can replace or add new medical PDFs anytime — just re-run `index_data.py` to refresh the knowledge base.  
- All data stays **local** — perfect for privacy-sensitive environments.  

---

## 🧾 License

This project is released under the **MIT License** — free to use, modify, and share.

---

👩‍💻 Created by [Seyali-ahm](https://github.com/Seyali-ahm)  
✨ Offline, Private, and Open Source.
