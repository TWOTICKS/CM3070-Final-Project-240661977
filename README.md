# AI Education Assistant (Source-Grounded Lecture Q&A)

CM3070 Final Project (University of London). Project template: Orchestrating
AI Models to Achieve a Goal.

A retrieval- augmented question answering system for lecture transcripts. It
answers only from the transcript you give it, scores how well each answer is
actually supported by the retrieved text, and refuses when the lecture does
not cover the question.

The wider project design covers four modules (transcription, summarisation,
Q&A, engagement analysis). This repository implements and evaluates one of
them (the RAG Q&A module). The report is explicit about that scope and why
it was chosen; the other three is designed but not built.

## What it does

- Two-stage retrieval: FAISS bi-encoder search over chunked transcript,
  then a cross-encoder re-ranks the shortlist.
- Grounded generation: a local LLM answers only from retrieved passages,
  alongside a no-retrieval baseline for direct comparison.
- Faithfulness checking: an NLI model scores whether each answer sentence
  is entailed by the retrieved text.
- Two abstention gates: one on retrieval confidence before generation, one
  on answer faithfulness after it.

## Models

| Role | Model |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` |
| Faithfulness | `cross-encoder/nli-deberta-v3-small` |

All run locally. No API keys required.

## Setup

Requires Python 3.12 and an NVIDIA GPU (developed on an RTX 4080 SUPER, 16 GB;
CPU will work but generation will be slow).

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install torch separately as shown. Putting its index URL in
requirements.txt breaks resolution of the other packages.

Build the index for the bundled sample transcript:

```bash
.venv\Scripts\python.exe -m src.index_store
```

## Usage

Web interface (easiest, upload your own transcript, see the trust signals):

```bash
.venv\Scripts\python.exe app.py
```

Then open http://localhost:7860. The first question load all four models and
takes 30–60 seconds; subsequent questions are fast.

Command line:

```bash
.venv\Scripts\python.exe ask.py "What is dropout and how does it help prevent overfitting?"
```

`ask_before.py` is a preserved copy of the original faithfulness
implementation, kept so the bug described in the report can be reproduced
side by side with the fix.

Evaluation:

```bash
.venv\Scripts\python.exe evaluate.py               # 11 questions, ML lecture
.venv\Scripts\python.exe evaluate_cross_domain.py  # 33 questions, 3 domains
```

Both write raw JSON and a readable summary into `results/`. Every figure quoted
in the report comes from those files.

## Results summary

Evaluated across three lecture transcripts with no shared vocabulary
(machine learning, Roman history, nutrition science), 33 questions total.

| Domain | Recall@3 | Precision@3 | Faithfulness | Out-of-scope abstained |
|---|---|---|---|---|
| ML lecture | 1.00 | 0.80 | 0.33 | No |
| Roman history | 1.00 | 0.70 | 0.54 | Yes |
| Nutrition science | 1.00 | 0.67 | 0.49 | Yes |

Two findings the report treats in detail, both negative:

1. The original faithfulness metric scored 0.00 on nine of ten correct
   answers, because a ~300-word concatenated premise falls outside the NLI
   model's training distribution. Scoring each sentence against each chunk
   separately fixed it, but introduced its own regression on answers that
   combine facts from two chunks.
2. The post-generation confidence gate flags correct refusals. A refusal is
   a statement about the transcript rather than a claim drawn from it, so no
   passage can entail it. It cannot be cleanly fixed without replacing the
   underlying metric.

## Repository layout

```
app.py                     Web interface (Gradio)
ask.py / ask_before.py     CLI demo — current and pre-fix versions
evaluate.py                Single-domain evaluation
evaluate_cross_domain.py   Three-domain evaluation
src/
  chunking.py              Topic-aware sentence-aligned chunking
  index_store.py           Embeddings -> FAISS index
  retriever.py             Two-stage retrieval
  generator.py             Grounded + baseline generation
  faithfulness.py          NLI checking (concatenated and per-chunk)
  pipeline.py              Orchestration and both abstention gates
data/
  lecture_transcript.txt   Sample ML lecture
  eval_questions.json      Evaluation set
  test_transcripts/        Roman history + nutrition science, with question sets
results/                   Evaluation output
```

## Note on the transcripts

All three lecture transcripts are synthetic, written for this project. They are
not recordings of real lectures, which avoids any copyright or consent issue
and give a known ground truth for evaluation.
