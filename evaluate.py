"""
evaluate.py — RAGAS evaluation for VNIT ECE RAG Chatbot
Usage:
    python evaluate.py --key gsk_xxxx
    python evaluate.py --key gsk_xxxx --questions custom_qa.json --out results.json
"""

import sys
from types import ModuleType

# Create a mock module structure in memory
mock_vertex = ModuleType("langchain_community.chat_models.vertexai")
mock_vertex.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex

import argparse
import json
import time
import warnings 

# 1. SILENCE WARNINGS FIRST BEFORE ANY OTHER IMPORTS RUN
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
warnings.filterwarnings("ignore", category=UserWarning, module="ragas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from pathlib import Path
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper  # Using reliable wrapper fallback
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig 

from rag_chain import build_rag_chain, load_vectorstore

DEFAULT_QA = [
    {
        "question": "What are the core subjects in the ECE curriculum at VNIT Nagpur?",
        "ground_truth": "Core subjects include Digital Electronics, Signals and Systems, Analog Circuits, Communication Systems, Electromagnetic Theory, VLSI Design, Microprocessors, and Control Systems.",
    },
    {
        "question": "What research areas does the ECE department focus on?",
        "ground_truth": "Research areas include signal and image processing, wireless communications, VLSI design, embedded systems, IoT, antenna design, and optical communication.",
    },
    {
        "question": "What laboratories are available in the ECE department?",
        "ground_truth": "The ECE department has Electronics Lab, Communication Lab, VLSI Lab, Microprocessor Lab, Signal Processing Lab, and DSP Lab.",
    },
    {
        "question": "What M.Tech specialisations are offered in ECE at VNIT Nagpur?",
        "ground_truth": "VNIT Nagpur ECE offers M.Tech in Communication Engineering, VLSI Design, Embedded Systems, and Signal Processing.",
    },
    {
        "question": "Who can I contact for admission queries in the ECE department?",
        "ground_truth": "Admission queries can be directed to the Head of the ECE Department or the departmental office at VNIT Nagpur.",
    },
]


def run_ragas(groq_api_key, qa_pairs, output_file="ragas_results.json"):
    print(f"\n{'='*60}")
    print("  RAGAS Evaluation — VNIT ECE RAG Chatbot")
    print(f"{'='*60}\n")

    chain = build_rag_chain(groq_api_key)
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    samples = []

    for i, pair in enumerate(qa_pairs, 1):
        q  = pair["question"]
        gt = pair["ground_truth"]
        print(f"[{i}/{len(qa_pairs)}] {q[:70]}…")

        try:
            result = chain.invoke(
                {"input": q},
                config={"configurable": {"session_id": f"eval_{i}"}}
            )
            answer = result["answer"]
        except Exception as e:
            print(f"  ⚠ Chain error: {e}")
            answer = ""

        docs = retriever.invoke(q)
        contexts = [d.page_content for d in docs]

        samples.append(SingleTurnSample(
            user_input=q,
            response=answer,
            retrieved_contexts=contexts,
            reference=gt,
        ))

        time.sleep(0.5)

    # Wrap Groq cleanly using the traditional LangchainLLMWrapper
    chat_model = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )
    evaluator_llm = LangchainLLMWrapper(chat_model)

    dataset = EvaluationDataset(samples=samples)

    # Throttling config to stay well below the 6,000 Tokens-Per-Minute free-tier cap
    run_config = RunConfig(
        max_workers=1,  # Keeps it sequential to avoid TPM limits
        timeout=300,    # Boosted from 90 to 300 seconds to prevent TimeoutErrors
        max_retries=3,  # Automatically retry if the API drops or hangs
    )

    print("\n⏳ Computing RAGAS metrics (Sequential Mode to prevent Rate Limits)…\n")
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm,
        run_config=run_config
    )

    # Defensive function to safely round results even if a specific API row failed
    # Convert result directly to a dict so we can safely read it, or fall back to 0.0
    result_dict = result.scores

    # Defensive function to safely round results even if a specific metric run failed
    def safe_round(val):
        try:
            if isinstance(val, list):
                val = val[0] if val else 0.0
            return round(float(val), 4)
        except (TypeError, ValueError):
            return 0.0

    df = result.to_pandas()
    # Defensive function to safely extract the mean of a column
    def get_column_average(df_obj, col_name):
        try:
            if col_name in df_obj.columns:
                # Calculate the average of the column, ignoring any NaN rows
                return round(float(df_obj[col_name].mean()), 4)
            return 0.0
        except Exception:
            return 0.0

    # 2. Extract column metrics dynamically from the dataframe
    scores = {
        "faithfulness":       get_column_average(df, "faithfulness"),
        "context_recall":     get_column_average(df, "llm_context_recall"),
        "factual_correctness": get_column_average(df, "factual_correctness"),
    }
    scores["average"] = round(sum(scores.values()) / len(scores), 4)
    output = {
        "aggregate": scores,
        "per_question": json.loads(df.to_json(orient="records")),
    }
    Path(output_file).write_text(json.dumps(output, indent=2, default=str))

    print("=" * 45)
    print(f"  {'Metric':<25} {'Score':>6}")
    print("-" * 45)
    for k, v in scores.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<25} {v:>6.4f}  {bar}")
    print("=" * 45)
    print(f"\n✅ Results saved → {output_file}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key",       required=True)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--out",       default="ragas_results.json")
    args = parser.parse_args()

    qa_pairs = json.loads(Path(args.questions).read_text()) if args.questions else DEFAULT_QA

    scores = run_ragas(
        groq_api_key=args.key,
        qa_pairs=qa_pairs,
        output_file=args.out,
    )
    sys.exit(0 if scores["average"] >= 0.5 else 1)


if __name__ == "__main__":
    main()