---
title: VNIT ECE Department Chatbot
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# 🎓 VNIT Nagpur — ECE Department RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for the Electronics & Communication Engineering department at **VNIT Nagpur**, powered by:

| Component | Technology |
|---|---|
| LLM | Groq · `llama3-8b-8192` (free tier) |
| RAG Framework | LangChain `ConversationalRetrievalChain` |
| Vector DB | ChromaDB (persistent, collection `ece_department`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| UI | Gradio 4 |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) |

---

## 🚀 Deploy to HuggingFace Spaces

### Step 1 — Create the Space

```bash
# Install HF Hub CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a new Gradio space
huggingface-cli repo create vnit-ece-chatbot --type space --space_sdk gradio
```

### Step 2 — Push your files

```bash
cd vnit-ece-chatbot/
git init
git remote add origin https://huggingface.co/spaces/<YOUR_HF_USERNAME>/vnit-ece-chatbot

# Copy ChromaDB data into repo (REQUIRED)
cp -r /path/to/your/chroma_db ./chroma_db

git add .
git commit -m "Initial deploy"
git push origin main
```

### Step 3 — Add Groq API secret

In your Space → **Settings → Repository secrets** → add:

```
Name : GROQ_API_KEY
Value: gsk_xxxxxxxxxxxxxxxx
```

The app auto-reads this secret so users don't need to enter the key manually.

---

## 💻 Run Locally

```bash
# 1. Clone / enter project folder
cd vnit-ece-chatbot/

# 2. Create virtual env
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq key
export GROQ_API_KEY=gsk_xxxx    # or add to .env

# 5. Launch
python app.py
# → Open http://localhost:7860
```

---

## 🧪 RAGAS Evaluation

```bash
# Run with default 5-question test set
python evaluate.py --key gsk_xxxx

# Run with custom questions (JSON array of {question, ground_truth})
python evaluate.py --key gsk_xxxx --questions my_qa.json --out results.json
```

Sample output:

```
=============================================
  Metric                  Score
---------------------------------------------
  faithfulness            0.8732  █████████████████
  answer_relevancy        0.9104  ██████████████████
  context_precision       0.8210  ████████████████
  context_recall          0.7891  ███████████████
  average                 0.8484  ████████████████
=============================================
```

---

## 📁 Project Structure

```
vnit-ece-chatbot/
├── app.py            # Gradio UI + event wiring
├── rag_chain.py      # LangChain chain, memory, retriever
├── evaluate.py       # RAGAS evaluation script
├── requirements.txt  # Pinned dependencies
├── README.md         # This file (HF Spaces config lives in YAML header)
└── chroma_db/        # ← copy your persisted ChromaDB folder here
    ├── chroma.sqlite3
    └── ...
```

---

## ⚙️ Configuration

All tunables live at the top of `rag_chain.py`:

| Variable | Default | Description |
|---|---|---|
| `CHROMA_PATH` | `./chroma_db` | Path to persisted ChromaDB |
| `GROQ_MODEL` | `llama3-8b-8192` | Any Groq-supported model |
| `RETRIEVER_K` | `5` | Docs returned per query |
| `RETRIEVER_FETCH_K` | `12` | MMR candidate pool |
| `MEMORY_WINDOW` | `6` | Conversation turns kept |

---

*Built for VNIT Nagpur ECE · Deadline May 31, 2025*