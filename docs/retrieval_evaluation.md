# Retrieval and Answer Evaluation

## 1. Objective and scope

This document records the current development evaluation for the Engineering Knowledge Intelligence Platform. The system retrieves evidence from a curated collection of official documentation covering Python exceptions, SQLAlchemy errors and connection pools, and PostgreSQL connection configuration.

The evaluation measures retrieval ranking quality, evidence-gate decisions, answer quality, and basic latency. The datasets are small and curated; results are useful for comparing the current configurations, not for claiming production-level generalization.

## 2. Dataset and methodology

The development answer-evaluation set contains 28 questions: 21 answerable and 7 out of scope. The development retrieval set contains 21 answerable queries.

The candidate-label file contains 346 query-candidate judgments across 21 queries: 66 relevant and 280 irrelevant. These labels were AI-assisted and have not been independently human-verified. In particular, query q10 has a high relevant-label count and should be spot-checked.

Recall is calculated over the judged candidate pool (the union of retrieved candidates), not over every passage in the full corpus. nDCG uses binary relevance labels. Consequently, these are pool-based development metrics and may change after label review or pool expansion.

The held-out split is not part of this report. It must remain unused until retrieval, prompt, and evidence-gate decisions are frozen.

## 3. Retrieval benchmark

| Retriever | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.7899 | 0.8913 | 0.8349 | 0.7499 | 0.7747 |
| BM25 | 0.7849 | 0.8988 | 0.8746 | 0.8120 | 0.8285 |
| **Hybrid** | **0.8813** | **0.9845** | **0.9206** | **0.8827** | **0.9085** |

Hybrid retrieval led across all five metrics in this candidate pool. Dense retrieval provides semantic matching, while BM25 contributes exact lexical matching for technical terms and identifiers. RRF combines their ranked lists without requiring their raw scores to be directly comparable.

## 4. Reranker comparison

The cross-encoder reranker was applied to hybrid top-10 candidates.

| Metric | Hybrid top-10 | Hybrid + reranker |
|---|---:|---:|
| Recall@5 | 0.8906 | 0.7636 |
| MRR | 0.9206 | 0.9180 |
| nDCG@5 | 0.8827 | 0.8172 |
| nDCG@10 | 0.9157 | 0.8953 |

The reranker reduced Recall@5 and both reported nDCG metrics, with no MRR improvement. It is not selected for the current baseline. This conclusion applies to the evaluated model, candidate pool, and configuration; it is not a claim that reranking is generally ineffective.

## 5. Evidence-gate evaluation

The dense-similarity threshold used for the recorded development run was **0.47**.

| Question category | Questions | Passed evidence gate | Rejected |
|---|---:|---:|---:|
| Answerable / in scope | 21 | 21 | 0 |
| Out of scope | 7 | 0 | 7 |
| **Total** | **28** | **21** | **7** |

The result shows complete separation on this particular development set. It does not establish how the threshold will behave on unseen questions, different wording, or a broader document corpus.

## 6. Answer-quality review

The development run generated answers for 21 answerable questions and abstained on 7 out-of-scope questions. The first-pass review summary was:

| Metric | Result | Qualification |
|---|---:|---|
| Expected-fact coverage | 38/42 (90.5%) | Provisional first-pass review |
| Factual correctness | 42/42 (100%) | Not independently human-validated |
| Citation support | 41/42 (97.6%) | Not independently human-validated |
| Expected vs. actual abstention agreement | 28/28 | On this development set |

The review labels were initially assistant-generated. They should not be described as independently human-validated benchmark results. A reviewer should check expected facts against the source passages and verify whether cited passages support the associated claims. A valid citation ID or a passing citation-format validator does not establish semantic entailment.

## 7. Latency and token snapshot

For the recorded run:

| Metric | Result |
|---|---:|
| Retrieval latency, median across 28 questions | 143.3 ms |
| Retrieval latency, mean across 28 questions | 142.1 ms |
| Generation latency, median across 21 answers | 4.26 s |
| Generation latency, mean across 21 answers | 6.29 s |
| Mean tokens per generated answer | 1,943.2 |
| Total tokens across generated answers | 40,807 |

These are single-run measurements, not service-level objectives. Repeated runs are needed to estimate normal variance and compare performance reliably.

## 8. Reproducibility artifacts

Selected development artifacts are stored in `results/pilot_v2/`:

- `retrieval_metrics_summary.csv` and `retrieval_metrics_per_query.csv`
- `reranker_comparison_summary.csv` and `reranker_comparison_per_query.csv`
- `reranker_latency_per_query.csv`
- `answer_eval_retrieval_dev.jsonl`
- `answer_eval_generate_dev.jsonl`
- `answer_review_summary_dev.md`

Development inputs and candidate labels are under `evaluation/dataset/`. Held-out datasets and held-out construction inputs are intentionally excluded from the public repository and must not be used for tuning.

## 9. Limitations

- The corpus contains three documentation areas and is not representative of arbitrary engineering questions.
- The retrieval benchmark has 21 queries; aggregate metrics can move noticeably when a small number of labels or queries change.
- Candidate relevance labels are AI-assisted and need independent human verification.
- Answer-review scores are provisional and need claim-level human review.
- Pool-based recall is limited by the judged candidate pool and should not be interpreted as full-corpus recall.
- The evidence gate uses a fixed similarity threshold and requires testing on more varied in-scope and out-of-scope queries.
- The reranker did not help in the current comparison; further experiments would need a justified change to the model, candidate pool, or configuration.
