from dataclasses import dataclass

from .faithfulness import EMPTY_RESULT, FaithfulnessChecker
from .generator import Generator
from .retriever import Retriever, RetrievedChunk

ABSTAIN_RERANK_THRESHOLD = -5.0
LOW_CONFIDENCE_MAX_FAITHFULNESS = 0.15
LOW_CONFIDENCE_MIN_CONTRADICTION = 0.5

ABSTAIN_MESSAGE = "I couldn't find anything in this lecture covering that topic."


@dataclass
class PipelineResult:
    question: str
    retrieved: list[RetrievedChunk]
    abstained: bool
    low_confidence: bool
    rag_answer: str
    baseline_answer: str
    faithfulness: dict
    faithfulness_per_chunk: dict
    baseline_faithfulness: dict
    baseline_faithfulness_per_chunk: dict


class RAGPipeline:
    def __init__(self, index_dir: str):
        self.retriever = Retriever(index_dir)
        self.generator = Generator()
        self.faithfulness_checker = FaithfulnessChecker()

    def load_index(self, index_dir: str) -> None:
        self.retriever = Retriever(index_dir)

    def answer(self, question: str, k_embed: int = 8, k_final: int = 3, rerank: bool = True) -> PipelineResult:
        retrieved = self.retriever.retrieve(question, k_embed=k_embed, k_final=k_final, rerank=rerank)
        scores = [(r.rerank_score if rerank else r.embed_score) for r in retrieved]
        abstained = rerank and (not scores or max(scores) < ABSTAIN_RERANK_THRESHOLD)

        context_texts = [r.chunk.text for r in retrieved]

        if abstained:
            rag_answer = ABSTAIN_MESSAGE
            faithfulness = dict(EMPTY_RESULT)
            faithfulness_per_chunk = dict(EMPTY_RESULT)
            low_confidence = False
        else:
            rag_answer = self.generator.answer_with_context(question, context_texts)
            faithfulness = self.faithfulness_checker.check(rag_answer, context_texts)
            faithfulness_per_chunk = self.faithfulness_checker.check_per_chunk(rag_answer, context_texts)

            fscore = faithfulness_per_chunk["faithfulness_score"]
            crate = faithfulness_per_chunk["contradiction_rate"]
            low_confidence = (
                (fscore is not None and fscore < LOW_CONFIDENCE_MAX_FAITHFULNESS)
                or (crate is not None and crate >= LOW_CONFIDENCE_MIN_CONTRADICTION)
            )

        baseline_answer = self.generator.answer_without_context(question)
        if context_texts:
            baseline_faithfulness = self.faithfulness_checker.check(baseline_answer, context_texts)
            baseline_faithfulness_per_chunk = self.faithfulness_checker.check_per_chunk(baseline_answer, context_texts)
        else:
            baseline_faithfulness = dict(EMPTY_RESULT)
            baseline_faithfulness_per_chunk = dict(EMPTY_RESULT)

        return PipelineResult(
            question=question,
            retrieved=retrieved,
            abstained=abstained,
            low_confidence=low_confidence,
            rag_answer=rag_answer,
            baseline_answer=baseline_answer,
            faithfulness=faithfulness,
            faithfulness_per_chunk=faithfulness_per_chunk,
            baseline_faithfulness=baseline_faithfulness,
            baseline_faithfulness_per_chunk=baseline_faithfulness_per_chunk,
        )
