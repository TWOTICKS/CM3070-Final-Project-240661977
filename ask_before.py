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
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    print("Loading models...")
    pipeline = RAGPipeline(INDEX_DIR)
    result = pipeline.answer(args.question, rerank=not args.no_rerank)

    print("\n" + "=" * 60)
    print("  V1 PROTOTYPE  (concatenated NLI faithfulness — pre-fix)")
    print("=" * 60)
    print(f"\nQuestion: {result.question}")

    print("\nRetrieved chunks:")
    for r in result.retrieved:
        score = r.rerank_score if r.rerank_score is not None else r.embed_score
        print(f"  [{r.chunk.chunk_id:02d}] score={score:6.2f}  ({r.chunk.topic})")
        print(f"        {r.chunk.text[:140]}...")

    if result.abstained:
        print("\n[Pipeline abstained — retrieval confidence below threshold]")

    print(f"\nRAG answer:\n  {result.rag_answer}")

    if result.faithfulness["faithfulness_score"] is not None:
        score = result.faithfulness["faithfulness_score"]
        print(f"\n  Faithfulness (concatenated premise): {score:.2f}")
        if score == 0.0:
            print(
                "\n  *** FLAW: faithfulness=0.00 even though the answer looks correct. ***\n"
                "  The NLI model receives all 3 retrieved chunks joined into a single\n"
                "  ~300-word premise. Small NLI models trained on short SNLI/MNLI pairs\n"
                "  default to 'neutral' (not 'entailment') for any hypothesis checked\n"
                "  against a long multi-sentence premise — even faithful paraphrases.\n"
                "  Result: the faithfulness metric cannot distinguish a grounded answer\n"
                "  from a hallucination. This flaw is fixed in ask.py (v2) using\n"
                "  per-chunk scoring."
            )

    print(f"\nBaseline answer (no retrieval):\n  {result.baseline_answer}")


if __name__ == "__main__":
    main()
