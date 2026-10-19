# RAG for Industrial Machine Safety

Question answering over a specialized French document collection (INRS brochures, EU Machinery Regulation 2023/1230, Machinery Directive 2006/42/EC), built as a controlled experiment rather than a demo: every component is hand-coded, compared against baselines, and evaluated.

*RAG-based question answering over a specialized document collection.*

> **Status: preliminary results.** 20 test questions were reviewed and validated by hand; the others have their evidence verified automatically but are not hand-reviewed. Generation is compared across all four systems on 34 questions (S0 and S1 also on 61). See *Limitations*.

## What it does

| System | Description |
|---|---|
| S0 | LLM alone, no retrieval (closed-book baseline) |
| S1 | BM25 + LLM (lexical baseline) |
| S2 | bge-m3 dense retrieval (FAISS) + LLM |
| S3 | BM25 + dense, Reciprocal Rank Fusion, cross-encoder reranker (bge-reranker-v2-m3) + LLM |

Corpus: 40 documents, 1,516 pages (38 INRS brochures, 2 EU legal texts). Source list in [`data/sources.csv`](data/sources.csv); PDFs are not versioned.

## Pipeline (all in `src/`, no LangChain / LlamaIndex)

1. `extract.py` — hybrid PDF extraction (PyMuPDF for text, pdfplumber for tables), heading detection, header/footer and hyphenation cleaning.
2. `chunking.py` — fixed-size vs structure-aware chunking (256 / 512 / 1024 tokens, bge-m3 tokenizer).
3. `bm25.py` — BM25 written from scratch, checked against `rank_bm25`.
4. `dense.py`, `hybrid.py` — embeddings + FAISS, RRF fusion, cross-encoder reranking.
5. `testset.py` — LLM-generated questions with **verbatim evidence checked against the source**, review export.
6. `retrieval_eval.py` — chunking-independent Recall@k and MRR.
7. `rag.py`, `run_generation.py`, `analyze.py` — generation with mandatory citations and abstention, LLM-as-judge, bootstrap confidence intervals, error taxonomy.

Everything expensive is cached (extracted text, embeddings, LLM calls), seeds are fixed, and results are written to `results/`.

## Preliminary results

**Retrieval** (structure-aware chunks, 512 tokens, 132 answerable questions, 95% bootstrap CI on paired differences):

| System | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| S1 BM25 | 0.72 | 0.92 | 0.86 |
| S2 dense | 0.53 | 0.85 | 0.72 |
| RRF only | 0.70 | 0.89 | 0.84 |
| **S3 hybrid + reranker** | **0.82** | **0.96** | **0.93** |

S3 vs BM25: Recall@1 +0.10 [0.03, 0.17] and MRR +0.07 [0.02, 0.12] (significant); Recall@5 +0.03 [0.00, 0.07] (not significant). BM25 beats dense retrieval (significant). RRF fusion alone does not beat BM25.

**Chunking ablation** (6 configurations x 4 systems): structure-aware chunking is significantly better than fixed-size at 512 tokens (BM25, dense, S3) and for dense retrieval at 1,024 tokens, but not at 256 tokens; the effect of chunk size itself is modest.

**Generation** (34 questions: 27 answerable, 7 unanswerable; same questions for all systems):

| | S0 (no retrieval) | S1 BM25 | S2 dense | S3 hybrid + reranker |
|---|---|---|---|---|
| Correct answers (LLM judge) | 0.28 [0.15, 0.41] | 0.81 | 0.89 | **0.89 [0.81, 0.96]** |
| Correct abstentions (7 questions) | 0/7 | 4/7 | 4/7 | 5/7 |
| Faithfulness to passages | n/a | 0.97 | 1.00 | 1.00 |
| Tokens per request | ~260 | ~2,090 | ~2,190 | ~2,250 |

Retrieval raises correct answers by +0.54 to +0.61 (significant, also on a 61-question sample: 0.24 to 0.78 for S1). Differences **between** the RAG systems are not significant (S3 vs S1: +0.07 [-0.02, 0.20]): the retrieval gain of S3 is not demonstrated in generation at this sample size. RAG does not eliminate hallucination: 2 to 3 of 7 unanswerable questions still receive an invented answer.

**Cost and latency** (34 questions): RAG multiplies tokens per request by about 8 (about 2,100-2,250 versus 260) and costs roughly the same per correct answer for all three RAG systems (about 2,700 tokens). The reranker adds about 750 ms per query on a local RTX 3050, which more than doubles end-to-end latency for S3 (1.37 s versus 0.62 s for S1) without a demonstrated generation gain; S1 or S2 give the best quality/cost/latency trade-off on this test set. Dollar figures use unverified pricing and are only indicative.

## Limitations

- Only **20 factual questions** were hand-reviewed; the rest have automatically verified evidence but are not reviewed, and some references are imprecise. Generation was not evaluated on reviewed questions.
- The LLM judge was checked against **annotations produced by Claude (an LLM), not a human** (30 answers, kappa 0.93): this measures LLM-LLM agreement only. Citation accuracy is checked formally (valid passage numbers), not semantically.
- Small generation samples (34 and 61 questions): wide confidence intervals, especially for abstention (7 questions).
- Test questions were generated from 512-token structural chunks and reuse source vocabulary, which likely advantages BM25.
- Generation ablations (top-k, prompt variants, HyDE, small vs large LLM) were not run; a single generator, judge, encoder and reranker were used. Dollar costs are indicative.
- Corpus is dominated by INRS documents; figures and pictograms are not extracted.
## Repository layout

| Path | Content |
|---|---|
| `notebooks/rag_securite_machines.ipynb` | Full write-up (French) with the integrated source code and executed results |
| `src/` | Pipeline modules (extraction, chunking, BM25, dense, hybrid, evaluation, generation, analysis) |
| `data/sources.csv` | Source URL of each of the 40 documents |
| `data/testset/` | Candidate questions, review file, hand-validated questions |
| `results/`, `figures/` | Metrics, bootstrap tables, error taxonomy, plots |

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build; Colab already has torch
pip install -r requirements.txt
cp .env.example .env     # add GROQ_API_KEY

python -m src.download_corpus   # PDFs -> data/raw/
python -m src.extract           # text cache
python -m src.chunking          # 6 chunk configurations
python -m src.testset           # candidate questions (needs Groq)
python -m src.run_retrieval     # S1-S3 retrieval metrics
python -m src.run_generation --n 60 && python -m src.analyze report
```

The notebook [`notebooks/rag_securite_machines.ipynb`](notebooks/rag_securite_machines.ipynb) (in French) reads the cached results and contains the full write-up.
