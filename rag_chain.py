"""
rag_chain.py — RAG chain for VNIT ECE Chatbot
Stack: LangChain 1.x + Groq + ChromaDB + all-MiniLM-L6-v2
Uses pure LCEL (no langchain.chains dependency)
"""

import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH       = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME   = "ece_department"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
GROQ_MODEL        = "llama-3.1-8b-instant"   # fast, free-tier, current
RETRIEVER_K       = 5
RETRIEVER_FETCH_K = 12
MEMORY_WINDOW     = 6

# ── Session storage ───────────────────────────────────────────────────────────
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    history = session_store[session_id]
    if len(history.messages) > MEMORY_WINDOW * 2:
        history.messages = history.messages[-(MEMORY_WINDOW * 2):]
    return history

# ── Prompts ───────────────────────────────────────────────────────────────────
_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, "
     "rewrite it as a standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful assistant for the ECE department at VNIT Nagpur.
Use ONLY the context below to answer. Be concise and student-friendly.
If the answer is not in the context say:
"I don't have that information. Please contact the ECE office or visit https://ece.vnit.ac.in"

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ── Vectorstore ───────────────────────────────────────────────────────────────
def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

# ── Chain ─────────────────────────────────────────────────────────────────────
def build_rag_chain(groq_api_key: str):
    if not groq_api_key or not groq_api_key.strip():
        raise ValueError("Groq API key must not be empty.")

    llm = ChatGroq(
        api_key=groq_api_key.strip(),
        model_name=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
    )

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K},
    )

    condense_chain = _CONDENSE_PROMPT | llm | StrOutputParser()

    def rag_pipeline(inputs: dict) -> dict:
        user_input   = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        # Step 1: condense if there's history
        if chat_history:
            question = condense_chain.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
        else:
            question = user_input

        # Step 2: retrieve
        docs = retriever.invoke(question)

        # Step 3: generate answer
        context_str = "\n\n".join(d.page_content for d in docs)
        answer = (
            _QA_PROMPT | llm | StrOutputParser()
        ).invoke({
            "input": user_input,
            "chat_history": chat_history,
            "context": context_str,
        })

        return {"answer": answer, "context": docs, "input": user_input}

    base_chain = RunnableLambda(rag_pipeline)

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )