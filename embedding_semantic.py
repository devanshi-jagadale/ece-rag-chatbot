"""
embedding_semantic.py — Semantic chunking + ChromaDB ingestion
Replaces embedding.py's heading-based chunking with SemanticChunker.
Usage: python embedding_semantic.py
"""

from pathlib import Path
import time

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR        = Path("./data/extracted")
DB_DIR          = "./chroma_db"
COLLECTION_NAME = "ece_department"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE      = 64


def infer_doc_type(filename: str) -> str:
    fname = filename.lower()
    if any(k in fname for k in ["pyq", "previous", "question", "exam", "qp"]):
        return "pyq"
    elif any(k in fname for k in ["lab", "manual", "experiment"]):
        return "lab_manual"
    elif any(k in fname for k in ["syllabus", "curriculum", "scheme"]):
        return "syllabus"
    elif any(k in fname for k in ["timetable", "schedule"]):
        return "timetable"
    return "other"


print("\nSTEP 1: Loading embedding model for semantic chunking...")
start = time.time()

lc_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

splitter = SemanticChunker(
    lc_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,
)

print(f"✓ Semantic chunker ready ({time.time() - start:.2f}s)")


print("\nSTEP 2: Reading and chunking markdown files...")

md_files = list(DATA_DIR.glob("*.md"))
print(f"✓ Found {len(md_files)} markdown files")

if not md_files:
    raise FileNotFoundError(f"No markdown files found in {DATA_DIR}")

all_chunks = []

for md_path in md_files:
    print(f"\nProcessing: {md_path.name}")
    start = time.time()

    text = md_path.read_text(encoding="utf-8").strip()
    if not text:
        print("  ⚠ Empty file, skipping")
        continue

    docs = splitter.create_documents([text])
    print(f"  ✓ {len(docs)} semantic chunks ({time.time() - start:.2f}s)")

    for i, doc in enumerate(docs):
        chunk_text = doc.page_content.strip()
        if len(chunk_text) < 30:
            continue

        first_line = chunk_text.split("\n")[0]
        heading = first_line if len(first_line) < 100 else ""

        all_chunks.append({
            "text":      chunk_text,
            "source":    md_path.name,
            "page":      0,
            "doc_type":  infer_doc_type(md_path.name),
            "headings":  heading,
            "chunk_idx": i,
            "chunk_id":  f"{md_path.stem}_sc{i}",   # sc = semantic chunk
        })

print(f"\n✓ Total semantic chunks: {len(all_chunks)}")


print("\nSTEP 3: Loading sentence-transformers for embedding...")
start = time.time()
embed_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✓ Model loaded ({time.time() - start:.2f}s)")


print("\nSTEP 4: Connecting to ChromaDB (fresh collection)...")
client = chromadb.PersistentClient(path=DB_DIR)

# Delete old collection and recreate fresh
try:
    client.delete_collection(COLLECTION_NAME)
    print("  ✓ Deleted old collection")
except Exception:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
print(f"✓ Collection '{COLLECTION_NAME}' ready")


print("\nSTEP 5: Embedding + storing chunks...")
total_batches = (len(all_chunks) - 1) // BATCH_SIZE + 1

for i in range(0, len(all_chunks), BATCH_SIZE):
    batch     = all_chunks[i : i + BATCH_SIZE]
    texts     = [c["text"]     for c in batch]
    ids       = [c["chunk_id"] for c in batch]
    metadatas = [
        {
            "source":    c["source"],
            "page":      c["page"],
            "doc_type":  c["doc_type"],
            "headings":  c["headings"],
            "chunk_idx": c["chunk_idx"],
        }
        for c in batch
    ]

    batch_num = i // BATCH_SIZE + 1
    print(f"\nEmbedding batch {batch_num}/{total_batches}...")
    start = time.time()

    embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()
    print(f"  ✓ Embeddings done ({time.time() - start:.2f}s)")

    start = time.time()
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"  ✓ Stored ({time.time() - start:.2f}s)")


print(f"\n✅ Done. Collection has {collection.count()} semantic chunks.")


print("\nSTEP 6: Test query...")
test_query = "What is the syllabus for signal processing?"
query_embedding = embed_model.encode([test_query]).tolist()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    include=["documents", "metadatas", "distances"],
)

print(f"\nQuery: {test_query}\n")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
)):
    print(f"Result {i+1} | Score: {1-dist:.3f} | Source: {meta['source']}")
    print(f"  {doc[:200]}...")
    print()