# RAG for Industrial Machine Safety

Question answering over a specialized French document collection (INRS brochures, EU Machinery Regulation 2023/1230, Machinery Directive 2006/42/EC), built as a controlled experiment rather than a demo: every component is hand-coded, compared against baselines, and evaluated.

*RAG-based question answering over a specialized document collection.*

> **Status: preliminary results.** The test set was generated with an LLM and its evidence is verified automatically, but the questions have not all been reviewed by hand. Generation was evaluated for S0 and S1 only (free API quota exhausted); S2 and S3 are evaluated on retrieval. See *Limitations*.

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

Retrieval (structure-aware chunks, 512 tokens, 132 answerable questions):

| System | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| S1 BM25 | 0.72 | 0.92 | 0.86 |
| S2 dense | 0.53 | 0.85 | 0.72 |
| RRF only | 0.70 | 0.89 | 0.84 |
| **S3 hybrid + reranker** | **0.82** | **0.96** | **0.93** |

Generation (61 questions, 95% bootstrap CI):

| | S0 (no retrieval) | S1 (BM25 + LLM) |
|---|---|---|
| Correct answers (LLM judge) | 0.24 [0.16, 0.32] | **0.78 [0.68, 0.88]** |
| Correct abstention on unanswerable questions | 0.15 | **0.77** |
| Faithfulness to retrieved passages | n/a | 0.98 |
| Tokens per request | ~270 | ~2,100 |

Main findings: retrieval cuts hallucinations sharply (paired difference +0.54, CI [0.41, 0.68]); the reranker is the component that improves retrieval most while fusion alone does not beat BM25; structure-aware chunking beats fixed chunking; BM25 beats dense retrieval on this corpus, partly because LLM-generated questions reuse the source vocabulary.

## Limitations

- Test questions are not all hand-reviewed; some references are imprecise.
- The LLM judge was checked against **annotations produced by Claude (an LLM), not a human** (30 answers, kappa 0.93): this measures LLM-LLM agreement only.
- Generation was run for S0 and S1 only; ablations (top-k, prompt variants, HyDE, small vs large LLM) were not run.
- Test questions were generated from 512-token structural chunks and favor lexical overlap, which likely advantages BM25.
- Corpus is dominated by INRS documents; figures and pictograms are not extracted.

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

## AI assistance

Developed with Claude Code (Anthropic). Code in `src/`, draft questions, and a first draft of the notebook were produced with its help; design choices, results and analysis are the author's responsibility to understand and defend.
