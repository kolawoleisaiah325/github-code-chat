# Retrieval evaluation and observed answer errors

Run `python evaluate.py` for the offline keyword baseline, or `python evaluate.py --with-models --answers` with the installed Ollama models. The recorded JSON contains each question, retrieved files, latency, and optional generated response.

## Method

The corpus is a fictional shop with five small authored Python files. There are 16 answerable questions and four deliberately out-of-scope questions. Relevance labels identify an expected file, not a specific passage. Every method returns up to three excerpts. Hybrid ranking combines BM25 and cosine rankings using reciprocal rank fusion; this run uses `nomic-embed-text` for embeddings and `llama3.2` for answers.

| Retrieval | Expected file first (Hit@1) | Expected file within three (Hit@3) | MRR@3 | Median retrieval latency |
| --- | ---: | ---: | ---: | ---: |
| Keyword / BM25 | 93.75% (15/16) | 100% (16/16) | 0.9688 | 0.344 ms |
| Vector / cosine | 87.5% (14/16) | 100% (16/16) | 0.9375 | 23.356 ms |
| Hybrid / rank fusion | 100% (16/16) | 100% (16/16) | 1.0 | 24.870 ms |

Source: `evaluation/model-results.json`, recorded October 2, 2026. The separate offline run in `baseline-results.json` has its own latency. Latency includes the query embedding request where applicable, excludes initial indexing and answer generation, and depends on hardware, warm-up and background work.

Keyword search returned no excerpts for all four out-of-scope cases. Vector and hybrid search returned nearest neighbors for those cases. The model abstained on all four in this run, but that is not a guaranteed property of dense retrieval or generation.

## Generated answers are a separate result

The local model produced 16 responses with in-range citation numbers and four explicit abstentions. **Citation-format success is not answer accuracy.** Manual inspection found errors despite valid citation numbers:

- The unknown-coupon answer said `total_cents` returns zero. The actual zero return is in `discount_cents`; `total_cents` still returns the subtotal plus shipping when no discount applies.
- The coupon/shipping answer correctly said shipping is not discounted, but incorrectly claimed shipping is always 2,000 cents. The domestic branch can return zero or 500 cents.
- The units-count answer described the function correctly but invented its precise line location.

These findings are retained in the recorded output rather than edited away. This is why the interface exposes the actual excerpt and line range. The current validation checks citation existence; it does not verify every claim, validate model-authored line numbers, or prove that a cited passage entails the answer.

## Interpretation

This small authored corpus and authored question set are a sanity check. File-level retrieval scores on it do not establish performance on real repositories, passage retrieval accuracy, code comprehension or answer correctness. Development decisions were made while inspecting this example, so it is not an independent test set. No aggregate answer-accuracy score is claimed.

A next evaluation should freeze a separate test set on independently selected real repositories, label supporting passages, score hallucinations and appropriate abstentions, and include ambiguous follow-ups and questions requiring several files. Compare function-aware chunks and reranking with this baseline before claiming improvement.
