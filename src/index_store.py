import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk, build_chunks

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_index(chunks: list[Chunk], embedder: SentenceTransformer | None = None):
    embedder = embedder or SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = _normalize(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def save_index(index, chunks: list[Chunk], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    meta = [
        {"chunk_id": c.chunk_id, "topic": c.topic, "section_index": c.section_index, "text": c.text}
        for c in chunks
    ]
    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_index(out_dir: str):
    index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "chunks.json"), encoding="utf-8") as f:
        meta = json.load(f)
    chunks = [Chunk(**m) for m in meta]
    return index, chunks


def embed_query(query: str, embedder: SentenceTransformer | None = None) -> np.ndarray:
    embedder = embedder or SentenceTransformer(EMBED_MODEL_NAME)
    vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    return _normalize(vec).astype("float32")


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    transcript = os.path.join(here, "..", "data", "lecture_transcript.txt")
    out_dir = os.path.join(here, "..", "data", "index")

    chunks = build_chunks(transcript)
    print(f"Building index over {len(chunks)} chunks using {EMBED_MODEL_NAME} ...")
    index, embeddings = build_index(chunks)
    save_index(index, chunks, out_dir)
    print(f"Saved index ({embeddings.shape[0]} vectors, dim={embeddings.shape[1]}) to {out_dir}")
