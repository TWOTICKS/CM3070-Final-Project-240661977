import os
from dataclasses import dataclass

from sentence_transformers import CrossEncoder, SentenceTransformer

from .chunking import Chunk
from .index_store import EMBED_MODEL_NAME, embed_query, load_index

CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    embed_score: float
    rerank_score: float | None = None


class Retriever:
    def __init__(self, index_dir: str, load_cross_encoder: bool = True):
        self.index, self.chunks = load_index(index_dir)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME) if load_cross_encoder else None

    def retrieve(self, query: str, k_embed: int = 8, k_final: int = 3, rerank: bool = True):
        q_vec = embed_query(query, self.embedder)
        scores, idxs = self.index.search(q_vec, k_embed)
        candidates = [
            RetrievedChunk(chunk=self.chunks[i], embed_score=float(s))
            for s, i in zip(scores[0], idxs[0])
            if i != -1
        ]

        if rerank and self.cross_encoder is not None and candidates:
            pairs = [(query, c.chunk.text) for c in candidates]
            rerank_scores = self.cross_encoder.predict(pairs)
            for c, rs in zip(candidates, rerank_scores):
                c.rerank_score = float(rs)
            candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        else:
            candidates.sort(key=lambda c: c.embed_score, reverse=True)

        return candidates[:k_final]


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    index_dir = os.path.join(here, "..", "data", "index")
    retriever = Retriever(index_dir)

    for query in [
        "What is dropout and why is it used?",
        "Compare CNNs and RNNs.",
    ]:
        print(f"\nQuery: {query}")
        for r in retriever.retrieve(query):
            print(f"  [{r.chunk.chunk_id:02d}] embed={r.embed_score:.3f} rerank={r.rerank_score:.3f} ({r.chunk.topic})")
            print(f"       {r.chunk.text[:120]}...")
