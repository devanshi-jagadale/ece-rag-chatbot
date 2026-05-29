---
title: VNIT ECE Department Chatbot
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

# 🎓 VNIT Nagpur — ECE Department RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for the Electronics & Communication Engineering department at **VNIT Nagpur**.

| Component | Technology |
|---|---|
| LLM | Groq · `llama-3.3-70b-versatile` (free tier) |
| RAG Framework | LangChain 1.x LCEL — pure `RunnableLambda` pipeline |
| Vector DB | ChromaDB (persistent, collection `ece_department`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| UI | Gradio 5 |
| Evaluation | RAGAS — `Faithfulness`, `LLMContextRecall`, `FactualCorrectness` |

---

## 📊 Evaluation Results

Evaluated on 5 questions using RAGAS with `llama-3.3-70b-versatile` as judge.
Two chunking strategies were compared:

| Chunk Strategy | Faithfulness | Context Recall | Factual Correctness | Average |
|---|---|---|---|---|
| **Heading-based** ✅ | 0.3000 | 0.0000 | 0.0000 | 0.1000 |
| Semantic | 0.2133 | 0.0000 | 0.0000 | 0.0711 |

**Finding:** Heading-based chunking outperforms semantic chunking on this corpus.
Since the source documents are already structured with markdown headings, heading-based
splitting produces more coherent chunks than embedding-based boundary detection.
Semantic chunking works best on unstructured prose — a useful negative result.

> **Note:** Low recall and factual correctness reflect limited dataset coverage rather
> than a pipeline failure. The retrieval chain is working correctly — answers are
> grounded in retrieved context. Scores will improve with a richer document corpus.

---

## ▶️ Live Demo
https://devanshi-jagadale-vnit-ece-chatbot.hf.space/

---

## 💻 Run Locally

```bash
# 1. Create fresh virtual environment
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# source venv/bin/activate         # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Rebuild ChromaDB from your extracted markdown files
python embedding.py

# 4. Set your Groq key
export GROQ_API_KEY=gsk_xxxx

# 5. Launch
python app.py
# → Open http://localhost:7860
```

---

## 🚀 Deploy to HuggingFace Spaces

```bash
# 1. Install HF CLI and login
pip install huggingface_hub
huggingface-cli login

# 2. Create Space
huggingface-cli repo create vnit-ece-chatbot --type space --space_sdk gradio

# 3. Push (chroma_db uses Git LFS for large files)
git lfs install
git lfs track "*.sqlite3" "*.bin" "*.pickle"
git add .gitattributes
git add app.py rag_chain.py requirements.txt README.md chroma_db/
git commit -m "initial deploy"
git push origin main
```

Add your Groq API key in **Space → Settings → Variables and secrets**:
```
Name:  GROQ_API_KEY
Value: gsk_xxxxxxxxxxxxxxxx
```

The app auto-reads this secret — no key input needed in the UI.

---

## 🧪 RAGAS Evaluation

```bash
# Run with default 5-question test set
python evaluate.py --key gsk_xxxx

# Run with custom questions (JSON array of {question, ground_truth})
python evaluate.py --key gsk_xxxx --questions my_qa.json --out results.json

# Run semantic chunking experiment
python embedding_semantic.py   # rebuilds chroma_db with semantic chunks
python evaluate.py --key gsk_xxxx --out ragas_results_semantic.json
```

---

## 📁 Project Structure

```
vnit-ece-chatbot/
├── app.py                      # Gradio 5 UI — auto-initializes from env secret
├── rag_chain.py                # LCEL RAG chain — condense → retrieve → answer
├── embedding.py                # Heading-based chunking + ChromaDB ingestion
├── embedding_semantic.py       # Semantic chunking experiment (SemanticChunker)
├── extraction.py               # PDF → markdown via PyMuPDF
├── chunking.py                 # Markdown chunking utilities
├── evaluate.py                 # RAGAS evaluation script
├── ragas_results.json          # Baseline evaluation results
├── ragas_results_semantic.json # Semantic chunking evaluation results
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── chroma_db/                  # Persisted ChromaDB (tracked via Git LFS)
    ├── chroma.sqlite3
    └── ...
```

---

## ⚙️ Configuration

All tunables live at the top of `rag_chain.py`:

| Variable | Default | Description |
|---|---|---|
| `CHROMA_PATH` | `./chroma_db` | Path to persisted ChromaDB |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq-supported model |
| `RETRIEVER_K` | `5` | Docs returned per query |
| `RETRIEVER_FETCH_K` | `12` | MMR candidate pool |
| `MEMORY_WINDOW` | `6` | Conversation turns kept |

---

*Built for VNIT Nagpur ECE · May 2026*