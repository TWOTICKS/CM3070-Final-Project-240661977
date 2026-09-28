import re

import numpy as np
from sentence_transformers import CrossEncoder

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

EMPTY_RESULT = {"sentence_results": [], "faithfulness_score": None, "contradiction_rate": None}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class FaithfulnessChecker:
    def __init__(self, model_name: str = NLI_MODEL_NAME):
        self.model = CrossEncoder(model_name)
        id2label = self.model.model.config.id2label
        self.labels = [id2label[i].lower() for i in range(len(id2label))]

    def _summarise(self, results: list[dict]) -> dict:
        n = len(results)
        entailed = sum(1 for r in results if r["label"] == "entailment")
        contradicted = sum(1 for r in results if r["label"] == "contradiction")
        return {
            "sentence_results": results,
            "faithfulness_score": entailed / n,
            "contradiction_rate": contradicted / n,
        }

    def _result(self, sentence: str, probs: np.ndarray) -> dict:
        return {
            "sentence": sentence,
            "label": self.labels[int(np.argmax(probs))],
            "scores": dict(zip(self.labels, probs.tolist())),
        }

    def check(self, answer: str, context_chunks: list[str]) -> dict:
        sentences = [s.strip() for s in SENTENCE_RE.split(answer) if s.strip()]
        if not sentences:
            return dict(EMPTY_RESULT)

        context = " ".join(context_chunks)
        raw_scores = self.model.predict([(context, s) for s in sentences])
        results = [
            self._result(sentence, _softmax(np.asarray(scores)))
            for sentence, scores in zip(sentences, raw_scores)
        ]
        return self._summarise(results)

    def check_per_chunk(self, answer: str, context_chunks: list[str]) -> dict:
        sentences = [s.strip() for s in SENTENCE_RE.split(answer) if s.strip()]
        if not sentences or not context_chunks:
            return dict(EMPTY_RESULT)

        pairs = [(chunk, s) for s in sentences for chunk in context_chunks]
        raw_scores = self.model.predict(pairs)

        entail_idx = self.labels.index("entailment")
        n_chunks = len(context_chunks)
        results = []
        for i, sentence in enumerate(sentences):
            per_chunk = [_softmax(np.asarray(s)) for s in raw_scores[i * n_chunks:(i + 1) * n_chunks]]
            best = max(per_chunk, key=lambda p: p[entail_idx])
            results.append(self._result(sentence, best))
        return self._summarise(results)


if __name__ == "__main__":
    checker = FaithfulnessChecker()
    context = [
        "One popular method is dropout, where during training we randomly switch off a fraction of "
        "neurons in each forward pass, which prevents the network from relying too heavily on any "
        "single neuron and encourages more robust, distributed representations."
    ]
    grounded = "Dropout randomly switches off a fraction of neurons during training to prevent overfitting."
    hallucinated = "Dropout was invented in 1990 by a team at Stanford and is mainly used for image compression."

    print("Grounded answer:", checker.check(grounded, context))
    print("\nHallucinated answer:", checker.check(hallucinated, context))
