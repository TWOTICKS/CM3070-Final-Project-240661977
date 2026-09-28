#!/usr/bin/env python
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from src.chunking import build_chunks  # noqa: E402
from src.index_store import build_index, save_index  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

HERE = os.path.dirname(__file__)
ML_INDEX_DIR = os.path.join(HERE, "data", "index")
RESULTS_DIR = os.path.join(HERE, "results")
TEST_DIR = os.path.join(HERE, "data", "test_transcripts")

DOMAINS = [
    {
        "name": "ML Lecture (original, evaluated corpus)",
        "transcript": os.path.join(HERE, "data", "lecture_transcript.txt"),
        "eval_path": os.path.join(HERE, "data", "eval_questions.json"),
        "index_dir": ML_INDEX_DIR,  # reuse the existing evaluated index, do not rebuild
    },
    {
        "name": "Roman History",
        "transcript": os.path.join(TEST_DIR, "roman_history_lecture.txt"),
        "eval_path": os.path.join(TEST_DIR, "roman_history_eval_questions.json"),
        "index_dir": None,  # built fresh below
    },
    {
        "name": "Nutrition Science",
        "transcript": os.path.join(TEST_DIR, "nutrition_science_lecture.txt"),
        "eval_path": os.path.join(TEST_DIR, "nutrition_science_eval_questions.json"),
        "index_dir": None,
    },
]


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


def run_domain(pipeline: RAGPipeline, domain: dict) -> dict:
    with open(domain["eval_path"], encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        question = q["question"]
        relevant_topics = q["relevant_topics"]
        print(f"    {q['id']}: {question}")

        retrieved_rerank = pipeline.retriever.retrieve(question, k_embed=8, k_final=3, rerank=True)
        retrieved_norerank = pipeline.retriever.retrieve(question, k_embed=8, k_final=3, rerank=False)

        recall_rr, precision_rr = retrieval_recall_precision(retrieved_rerank, relevant_topics)
        recall_nr, precision_nr = retrieval_recall_precision(retrieved_norerank, relevant_topics)

        full = pipeline.answer(question, rerank=True)

        results.append({
            "id": q["id"],
            "question": question,
            "in_scope": q["in_scope"],
            "relevant_topics": relevant_topics,
            "retrieved_topics_rerank": [c.chunk.topic for c in retrieved_rerank],
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
            "expected_answer": q["expected_answer"],
        })

    in_scope = [r for r in results if r["in_scope"]]
    out_of_scope = [r for r in results if not r["in_scope"]]

    summary = {
        "domain": domain["name"],
        "n_questions": len(results),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_of_scope),
        "avg_recall_rerank": avg([r["recall_at_3_rerank"] for r in in_scope]),
        "avg_precision_rerank": avg([r["precision_at_3_rerank"] for r in in_scope]),
        "avg_recall_norerank": avg([r["recall_at_3_norerank"] for r in in_scope]),
        "avg_precision_norerank": avg([r["precision_at_3_norerank"] for r in in_scope]),
        "avg_faithfulness_concat": avg([r["rag_faithfulness"] for r in in_scope]),
        "avg_contradiction_concat": avg([r["rag_contradiction_rate"] for r in in_scope]),
        "avg_faithfulness_perchunk": avg([r["rag_faithfulness_per_chunk"] for r in in_scope]),
        "avg_contradiction_perchunk": avg([r["rag_contradiction_rate_per_chunk"] for r in in_scope]),
        "n_low_confidence": sum(1 for r in in_scope if r["low_confidence"]),
        "out_of_scope_abstained": [r["abstained"] for r in out_of_scope],
        "out_of_scope_details": out_of_scope,
        "per_question": results,
    }
    return summary


def main():
    print("Loading pipeline (embedder, cross-encoder, NLI model, LLM)...")
    pipeline = RAGPipeline(ML_INDEX_DIR)

    domain_summaries = []
    for domain in DOMAINS:
        print(f"\n=== {domain['name']} ===")
        if domain["index_dir"] is None:
            chunks = build_chunks(domain["transcript"])
            work_dir = tempfile.mkdtemp(prefix="cross_domain_eval_")
            index, _ = build_index(chunks)
            save_index(index, chunks, work_dir)
            pipeline.load_index(work_dir)
            print(f"  Built fresh index: {len(chunks)} chunks, "
                  f"{len({c.topic for c in chunks})} topics")
        else:
            pipeline.load_index(domain["index_dir"])
            print(f"  Using existing evaluated index at {domain['index_dir']}")

        domain_summaries.append(run_domain(pipeline, domain))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "cross_domain_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(domain_summaries, f, indent=2)

    lines = []
    lines.append("# Cross-Domain Generalisation Evaluation")
    lines.append("")
    lines.append(
        "The same evaluation methodology as the main evaluation (Appendix A) "
        "run across three lecture domains with no shared vocabulary: the "
        "original evaluated ML lecture, a Roman History lecture, and a "
        "Nutrition Science lecture. This tests whether retrieval and "
        "faithfulness scoring generalise beyond the ML domain, or were "
        "implicitly tuned to its phrasing."
    )
    lines.append("")
    lines.append("## Comparison across domains")
    lines.append("")
    lines.append("| Domain | Recall@3 | Precision@3 | Faithfulness (per-chunk) | Contradiction (per-chunk) | Low-confidence flags | Out-of-scope abstained |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in domain_summaries:
        oos = ", ".join(str(x) for x in s["out_of_scope_abstained"]) or "n/a"
        n_lc = s.get("n_low_confidence", "n/a")
        lines.append(
            f"| {s['domain']} | {s['avg_recall_rerank']:.2f} | "
            f"{s['avg_precision_rerank']:.2f} | {s['avg_faithfulness_perchunk']:.2f} | "
            f"{s['avg_contradiction_perchunk']:.2f} | {n_lc}/{s['n_in_scope']} | {oos} |"
        )
    lines.append("")

    for s in domain_summaries:
        lines.append(f"## {s['domain']}")
        lines.append("")
        lines.append(f"Questions: {s['n_questions']} ({s['n_in_scope']} in-scope, {s['n_out_of_scope']} out-of-scope)")
        lines.append("")
        lines.append("**Retrieval (mean recall@3 / precision@3, in-scope questions)**")
        lines.append("")
        lines.append(f"- With re-ranking:    recall={s['avg_recall_rerank']:.2f}, precision={s['avg_precision_rerank']:.2f}")
        lines.append(f"- Without re-ranking: recall={s['avg_recall_norerank']:.2f}, precision={s['avg_precision_norerank']:.2f}")
        lines.append("")
        lines.append("**Faithfulness (mean over in-scope RAG answers)**")
        lines.append("")
        lines.append(f"- Concatenated premise: faithfulness={s['avg_faithfulness_concat']:.2f}, contradiction={s['avg_contradiction_concat']:.2f}")
        lines.append(f"- Per-chunk premise:    faithfulness={s['avg_faithfulness_perchunk']:.2f}, contradiction={s['avg_contradiction_perchunk']:.2f}")
        lines.append("")
        if s["out_of_scope_details"]:
            lines.append("**Out-of-scope question(s)**")
            lines.append("")
            for r in s["out_of_scope_details"]:
                lines.append(f"- Q: {r['question']}")
                lines.append(f"  - top-3 rerank scores: {r['rerank_scores']}")
                lines.append(f"  - abstained: {r['abstained']}")
                lines.append(f"  - RAG answer: {r['rag_answer']}")
                lines.append(f"  - Baseline answer: {r['baseline_answer']}")
            lines.append("")
        lines.append("**Per-question detail (in-scope)**")
        lines.append("")
        for r in s["per_question"]:
            if not r["in_scope"]:
                continue
            lines.append(f"- **{r['id']}**: {r['question']}")
            lines.append(
                f"  - retrieved: {r['retrieved_topics_rerank']} "
                f"(recall={r['recall_at_3_rerank']:.2f}, precision={r['precision_at_3_rerank']:.2f})"
            )
            lines.append(f"  - RAG answer: {r['rag_answer']}")
            lines.append(
                f"  - faithfulness (concat)={r['rag_faithfulness']:.2f}, "
                f"(per-chunk)={r['rag_faithfulness_per_chunk']:.2f}, "
                f"contradiction (per-chunk)={r['rag_contradiction_rate_per_chunk']:.2f}"
            )
        lines.append("")

    summary_text = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "cross_domain_eval_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + "\n".join(lines[:lines.index("## " + domain_summaries[0]["domain"])]))
    print(f"\nFull results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
