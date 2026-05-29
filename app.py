"""
app.py — Gradio front-end for VNIT ECE RAG Chatbot
"""

import os
import gradio as gr
from rag_chain import build_rag_chain, session_store

_DEFAULT_KEY = os.getenv("GROQ_API_KEY", "")
_chain = None
SESSION_ID = "vnit_ece_chat_session"


def auto_initialize():
    global _chain
    if _DEFAULT_KEY:
        try:
            _chain = build_rag_chain(_DEFAULT_KEY)
            print("Auto-initialized with env key")
        except Exception as e:
            print(f"Auto-init failed: {e}")

auto_initialize()


def _format_sources(source_docs):
    if not source_docs:
        return ""
    seen = set()
    lines = []
    for doc in source_docs[:4]:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        label = src + (f" (p.{page})" if page else "")
        if label not in seen:
            seen.add(label)
            lines.append(f"- {label}")
    return "\n\n### Sources\n" + "\n".join(lines)


def respond(user_msg, history):
    global _chain
    if history is None:
        history = []
    if not user_msg.strip():
        return history, ""
    if _chain is None:
        history.append([user_msg, "Chatbot is not initialized. Please try again later."])
        return history, ""
    try:
        result = _chain.invoke(
            {"input": user_msg},
            config={"configurable": {"session_id": SESSION_ID}}
        )
        answer = result["answer"]
        sources = result.get("context", [])
        answer += _format_sources(sources)
    except Exception as exc:
        answer = f"LLM Error:\n{str(exc)}"
    history.append([user_msg, answer])
    return history, ""


def clear_chat():
    if SESSION_ID in session_store:
        session_store[SESSION_ID].clear()
    return []


with gr.Blocks() as demo:

    gr.HTML("""
    <div style="text-align:center;">
        <h1>VNIT Nagpur · ECE Department Assistant</h1>
        <p style="color:gray;">Powered by RAG · LangChain · Groq · ChromaDB</p>
    </div>
    """)

    chatbot = gr.Chatbot(label="ECE Assistant", min_height=500)

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask anything about VNIT ECE...",
            show_label=False,
            scale=5
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear Conversation")

    gr.Markdown("""
**Example questions:** What labs does ECE have? · List core courses in 3rd year · Who are the faculty in VLSI? · What are the placement stats?
""")

    send_btn.click(respond, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    msg_box.submit(respond, inputs=[msg_box, chatbot], outputs=[chatbot, msg_box])
    clear_btn.click(clear_chat, outputs=chatbot)


if __name__ == "__main__":
    print("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)