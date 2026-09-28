import re
from dataclasses import dataclass

TIMESTAMP_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]\s*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: int
    topic: str
    section_index: int
    text: str


def load_sections(transcript_path: str):
    raw = open(transcript_path, encoding="utf-8").read()
    parts = re.split(r"##\s*Topic:\s*(.+)", raw)

    if len(parts) == 1:
        body = TIMESTAMP_RE.sub("", raw).strip()
        return [("Lecture", body)] if body else []

    sections = []
    for i in range(1, len(parts), 2):
        topic = parts[i].strip()
        body = TIMESTAMP_RE.sub("", parts[i + 1]).strip()
        sections.append((topic, body))
    return sections


def chunk_section(text: str, target_words: int = 110, overlap_sentences: int = 1):
    sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    chunks = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        current.append(sentence)
        current_words += len(sentence.split())
        if current_words >= target_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)

    if current and (not chunks or " ".join(current) != chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


def build_chunks(transcript_path: str, target_words: int = 110, overlap_sentences: int = 1):
    chunks: list[Chunk] = []
    for section_index, (topic, body) in enumerate(load_sections(transcript_path)):
        for piece in chunk_section(body, target_words, overlap_sentences):
            chunks.append(Chunk(len(chunks), topic, section_index, piece))
    return chunks


if __name__ == "__main__":
    import os

    transcript = os.path.join(os.path.dirname(__file__), "..", "data", "lecture_transcript.txt")
    chunks = build_chunks(transcript)
    print(f"Total chunks: {len(chunks)}")
    for c in chunks:
        print(f"[{c.chunk_id:02d}] ({c.topic}) {len(c.text.split())} words")
        print("    " + c.text[:160] + ("..." if len(c.text) > 160 else ""))
