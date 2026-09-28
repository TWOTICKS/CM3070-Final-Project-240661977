#!/usr/bin/env python
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline import RAGPipeline  # noqa: E402

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask about the lecture")
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder re-ranking")
    args = parser.parse_args()

    print("Loading models (this can take ~30-60s on first run)...")
    pipeline = RAGPipeline(INDEX_DIR)

    result = pipeline.answer(args.question, rerank=not args.no_rerank)

    print(f"\nQuestion: {result.question}")
    print("\nRetrieved chunks:")
    for r in result.retrieved:
        score = r.rerank_score if r.rerank_score is not None else r.embed_score
        print(f"  [{r.chunk.chunk_id:02d}] score={score:6.2f}  ({r.chunk.topic})")
        print(f"        {r.chunk.text[:140]}...")

    if result.abstained:
        print("\n[Retrieval confidence below threshold -> pipeline abstains from answering]")

    print(f"\nRAG answer:\n  {result.rag_answer}")
    if result.faithfulness["faithfulness_score"] is not None:
        print(
            f"  (faithfulness [concatenated]: {result.faithfulness['faithfulness_score']:.2f}, "
            f"contradiction rate: {result.faithfulness['contradiction_rate']:.2f})"
        )
        print(
            f"  (faithfulness [per-chunk]:    {result.faithfulness_per_chunk['faithfulness_score']:.2f}, "
            f"contradiction rate: {result.faithfulness_per_chunk['contradiction_rate']:.2f})"
        )
    if result.low_confidence:
        print(
            "  [LOW CONFIDENCE: per-chunk faithfulness is very low or contradiction rate is "
            "high -> verify this answer against the retrieved passages above before trusting it]"
        )

    print(f"\nBaseline answer (no retrieval):\n  {result.baseline_answer}")


if __name__ == "__main__":
    main()
