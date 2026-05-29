"""
app.py — Gradio front-end for VNIT ECE RAG Chatbot
Run locally : python app.py
HF Spaces   : set GROQ_API_KEY secret → key input auto-populates
"""

import os
import gradio as gr

import gradio_client.utils as _cu
import gradio.blocks as _gb

from rag_chain import build_rag_chain, session_store

# Replace the _DEFAULT_KEY and _chain lines at the top with this:
_DEFAULT_KEY = os.getenv("GROQ_API_KEY", "")
_chain = None

def auto_initialize():
    global _chain
    if _DEFAULT_KEY:
        try:
            _chain = build_rag_chain(_DEFAULT_KEY)
            print("Auto-initialized with env key")
        except Exception as e:
            print(f"Auto-init failed: {e}")

auto_initialize()

# Patch 1: fix get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "any"
    if "const" in schema:
        return type(schema["const"]).__name__
    return schema.get("type", "any")

_cu.get_type = _safe_get_type

# Patch 2: silence APIInfoParseError by making get_api_info return empty
def _safe_get_api_info(self):
    return {"named_endpoints": {}, "unnamed_endpoints": {}}

_gb.Blocks.get_api_info = _safe_get_api_info
# ────────────────────────────────────────────────────────────────────────────

# Load API key from environment
_DEFAULT_KEY = os.getenv("GROQ_API_KEY", "")

# Global RAG chain
_chain = None
# Static session ID for single-user Gradio context
SESSION_ID = "vnit_ece_chat_session"


def _format_sources(source_docs):
    """Format retrieved source documents from modern LangChain 'context' output."""
    if not source_docs:
        return ""

    seen = set()
    lines = []

    for doc in source_docs[:4]:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")

        label = src
        if page:
            label += f" (p.{page})"

        if label not in seen:
            seen.add(label)
            lines.append(f"- {label}")

    return "\n\n### Sources\n" + "\n".join(lines)


def initialise(api_key):
    """Initialize the RAG chain."""
    global _chain

    key = api_key.strip() or _DEFAULT_KEY

    if not key:
        return "Enter your Groq API key."

    try:
        _chain = build_rag_chain(key)
        return "Chatbot initialized successfully."
    except Exception as exc:
        _chain = None
        return f"Initialization failed:\n{str(exc)}"


def respond(user_msg, history):
    """Handle chatbot interaction using modern LCEL syntax configuration."""
    global _chain

    if history is None:
        history = []

    if not user_msg.strip():
        return history, ""

    if _chain is None:
        history.append([
            user_msg,
            "Please initialize the chatbot first."
        ])
        return history, ""

    try:
        # Call modern chain using invoke() with input keys and session tracking configuration
        result = _chain.invoke(
            {"input": user_msg},
            config={"configurable": {"session_id": SESSION_ID}}
        )
        answer = result["answer"]

        # In LangChain v0.3+, retrieved docs reside in 'context' instead of 'source_documents'
        sources = result.get("context", [])
        answer += _format_sources(sources)

    except Exception as exc:
        answer = f"LLM Error:\n{str(exc)}"

    history.append([user_msg, answer])

    return history, ""


def clear_chat():
    """Clear conversational history out of the modern backend session dictionary."""
    global _chain

    if SESSION_ID in session_store:
        session_store[SESSION_ID].clear()

    return []


CSS = """
#title {
    text-align: center;
}

#chatbot {
    min-height: 500px;
}

footer {
    display: none !important;
}
"""


with gr.Blocks() as demo:

    gr.HTML("""
    <div id="title">
        <h1>VNIT Nagpur · ECE Department Assistant</h1>
        <p style="color:gray;">
            Powered by RAG · LangChain · Groq · ChromaDB
        </p>
    </div>
    """)

    with gr.Row():

        # LEFT PANEL
        # Replace the left column with this:
        with gr.Column(scale=1, min_width=260, visible=False):
            api_key_box = gr.Textbox(value=_DEFAULT_KEY, type="password")
            init_btn = gr.Button("Initialize")
            status_box = gr.Textbox()

            gr.Markdown("""
### Example Questions

- What labs does ECE have?
- List core courses in 3rd year
- Who are the faculty in VLSI?
- What are the placement stats?
- Describe the M.Tech specialisations
""")

        # RIGHT PANEL
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="ECE Assistant",
                elem_id="chatbot"
            )

            with gr.Row():

                msg_box = gr.Textbox(
                    placeholder="Ask anything about VNIT ECE...",
                    show_label=False,
                    scale=5
                )

                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1
                )

            clear_btn = gr.Button("Clear Conversation")

    # EVENTS

    init_btn.click(
        initialise,
        inputs=api_key_box,
        outputs=status_box
    )

    send_btn.click(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box]
    )

    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box]
    )

    clear_btn.click(
        clear_chat,
        outputs=chatbot
    )

if __name__ == "__main__":

    print("Starting Gradio app...")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )