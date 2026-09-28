#!/usr/bin/env python
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import RAGPipeline  # noqa: E402

HERE = os.path.dirname(__file__)
INDEX_DIR = os.path.join(HERE, "data", "index")
EVAL_PATH = os.path.join(HERE, "data", "eval_questions.json")
RESULTS_DIR = os.path.join(HERE, "results")


def retrieval_recall_precision(retrieved_chunks, relevant_topics):
    if not relevant_topics:
        return None, None
    retrieved_topics = [c.chunk.topic for c in retrieved_chunks]
    covered = set(retrieved_topics) & set(relevant_topics)
    recall = len(covered) / len(set(relevant_topics))
    precision = sum(1 for t in retrieved_topics if t in relevant_topics) / len(retrieved_topics)
    return recall, precision


def avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print("Loading pipeline (embedder, cross-encoder, NLI model, LLM)...")
    pipeline = RAGPipeline(INDEX_DIR)

    results = []
    for q in questions:
        question = q["question"]
        relevant_topics = q["relevant_topics"]
        print(f"  evaluating {q['id']}: {question}")

        retrieved_rerank = pipeline.retriever.retrieve(question, k_embed=8, k_final=3, rerank=True)
        retrieved_norerank = pipeline.retriever.retrieve(question, k_embed=8, k_final=3, rerank=False)

        recall_rr, precision_rr = retrieval_recall_precision(retrieved_rerank, relevant_topics)
        recall_nr, precision_nr = retrieval_recall_precision(retrieved_norerank, relevant_topics)

        full = pipeline.answer(question, rerank=True)

        results.append(
            {
                "id": q["id"],
                "question": question,
                "in_scope": q["in_scope"],
                "relevant_topics": relevant_topics,
                "retrieved_topics_rerank": [c.chunk.topic for c in retrieved_rerank],
                "retrieved_topics_norerank": [c.chunk.topic for c in retrieved_norerank],
                "rerank_scores": [round(c.rerank_score, 2) for c in retrieved_rerank],
                "recall_at_3_rerank": recall_rr,
                "precision_at_3_rerank": precision_rr,
                "recall_at_3_norerank": recall_nr,
                "precision_at_3_norerank": precision_nr,
                "abstained": full.abstained,
                "low_confidence": full.low_confidence,
                "rag_answer": full.rag_answer,
                "baseline_answer": full.baseline_answer,
                "rag_faithfulness": full.faithfulness["faithfulness_score"],
                "rag_contradiction_rate": full.faithfulness["contradiction_rate"],
                "rag_faithfulness_per_chunk": full.faithfulness_per_chunk["faithfulness_score"],
                "rag_contradiction_rate_per_chunk": full.faithfulness_per_chunk["contradiction_rate"],
                "baseline_faithfulness_vs_lecture": full.baseline_faithfulness["faithfulness_score"],
                "baseline_faithfulness_vs_lecture_per_chunk": full.baseline_faithfulness_per_chunk["faithfulness_score"],
                "expected_answer": q["expected_answer"],
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    in_scope = [r for r in results if r["in_scope"]]
    out_of_scope = [r for r in results if not r["in_scope"]]

    avg_recall_rr = avg([r["recall_at_3_rerank"] for r in in_scope])
    avg_recall_nr = avg([r["recall_at_3_norerank"] for r in in_scope])
    avg_precision_rr = avg([r["precision_at_3_rerank"] for r in in_scope])
    avg_precision_nr = avg([r["precision_at_3_norerank"] for r in in_scope])
    avg_faithfulness = avg([r["rag_faithfulness"] for r in in_scope])
    avg_baseline_faithfulness = avg([r["baseline_faithfulness_vs_lecture"] for r in in_scope])
    avg_faithfulness_pc = avg([r["rag_faithfulness_per_chunk"] for r in in_scope])
    avg_baseline_faithfulness_pc = avg([r["baseline_faithfulness_vs_lecture_per_chunk"] for r in in_scope])
    avg_contradiction = avg([r["rag_contradiction_rate"] for r in in_scope])
    avg_contradiction_pc = avg([r["rag_contradiction_rate_per_chunk"] for r in in_scope])

    lines = []
    lines.append("# RAG Prototype Evaluation Summary")
    lines.append("")
    lines.append(f"Questions evaluated: {len(results)} ({len(in_scope)} in-scope, {len(out_of_scope)} out-of-scope)")
    lines.append("")
    lines.append("## Retrieval quality (mean recall@3 / precision@3 by topic, in-scope questions)")
    lines.append("")
    lines.append(f"- With cross-encoder re-ranking:    recall={avg_recall_rr:.2f}, precision={avg_precision_rr:.2f}")
    lines.append(f"- Without re-ranking (embeds only): recall={avg_recall_nr:.2f}, precision={avg_precision_nr:.2f}")
    lines.append("")
    lines.append("## Answer faithfulness (NLI entailment rate vs retrieved context)")
    lines.append("")
    lines.append("Two scoring methods are compared:")
    lines.append("- **concatenated**: all retrieved chunks joined into one premise (original method).")
    lines.append("- **per-chunk**: each answer sentence is checked against each retrieved chunk")
    lines.append("  separately, keeping the chunk with the highest entailment probability.")
    lines.append("")
    lines.append(f"- RAG answers (grounded in retrieved context):")
    lines.append(f"  - concatenated: mean faithfulness = {avg_faithfulness:.2f}, contradiction rate = {avg_contradiction:.2f}")
    lines.append(f"  - per-chunk:    mean faithfulness = {avg_faithfulness_pc:.2f}, contradiction rate = {avg_contradiction_pc:.2f}")
    lines.append(f"- No-retrieval baseline answers, checked against the SAME retrieved context:")
    lines.append(f"  - concatenated: mean faithfulness = {avg_baseline_faithfulness:.2f}")
    lines.append(f"  - per-chunk:    mean faithfulness = {avg_baseline_faithfulness_pc:.2f}")
    lines.append("")
    lines.append("## Answer-aware confidence gate")
    lines.append("")
    lines.append(
        "Post-generation gate (`RAGPipeline.low_confidence`): flags an answer when "
        "per-chunk faithfulness < 0.15 or per-chunk contradiction rate >= 0.5. This "
        "is separate from retrieval-based abstention and catches the case where "
        "retrieval looked confident enough to proceed but the generated answer "
        "still ended up poorly grounded."
    )
    lines.append("")
    low_conf = [r for r in in_scope if r["low_confidence"]]
    lines.append(f"- In-scope answers flagged low-confidence: {len(low_conf)} / {len(in_scope)}")
    for r in low_conf:
        lines.append(
            f"  - {r['id']}: faithfulness(per-chunk)={r['rag_faithfulness_per_chunk']:.2f}, "
            f"contradiction(per-chunk)={r['rag_contradiction_rate_per_chunk']:.2f}"
        )
    lines.append("")
    lines.append("## Out-of-scope question handling")
    lines.append("")
    for r in out_of_scope:
        lines.append(f"- Q: {r['question']}")
        lines.append(f"  - top-3 rerank scores: {r['rerank_scores']}")
        lines.append(f"  - abstained: {r['abstained']}")
        lines.append(f"  - RAG answer: {r['rag_answer']}")
        lines.append(f"  - Baseline answer: {r['baseline_answer']}")
    lines.append("")
    lines.append("## Per-question detail (in-scope)")
    lines.append("")
    for r in in_scope:
        lines.append(f"### {r['id']}: {r['question']}")
        lines.append(f"- relevant topic(s): {r['relevant_topics']}")
        lines.append(
            f"- retrieved (reranked): {r['retrieved_topics_rerank']}  "
            f"(recall={r['recall_at_3_rerank']:.2f}, precision={r['precision_at_3_rerank']:.2f})"
        )
        lines.append(
            f"- retrieved (no rerank): {r['retrieved_topics_norerank']}  "
            f"(recall={r['recall_at_3_norerank']:.2f}, precision={r['precision_at_3_norerank']:.2f})"
        )
        lines.append(f"- RAG answer: {r['rag_answer']}")
        if r["rag_faithfulness"] is not None:
            lines.append(
                f"  - faithfulness (concatenated)={r['rag_faithfulness']:.2f}, contradiction_rate={r['rag_contradiction_rate']:.2f}"
            )
            lines.append(
                f"  - faithfulness (per-chunk)=   {r['rag_faithfulness_per_chunk']:.2f}, contradiction_rate={r['rag_contradiction_rate_per_chunk']:.2f}"
            )
        if r["low_confidence"]:
            lines.append("  - **[FLAGGED LOW CONFIDENCE]**")
        lines.append(f"- Baseline answer: {r['baseline_answer']}")
        lines.append(f"- Expected answer: {r['expected_answer']}")
        lines.append("")

    summary_text = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "eval_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + "\n".join(lines[: lines.index("## Per-question detail (in-scope)")]))
    print(f"\nFull results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
