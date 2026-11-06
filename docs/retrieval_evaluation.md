
# Retrieval and Answer Evaluation

## 1. Objective and scope

This document records the development evaluation for the Engineering Knowledge Intelligence Platform. The system retrieves evidence from a curated collection of official documentation covering Python exceptions, SQLAlchemy errors and connection pools, and PostgreSQL connection configuration.

The evaluation measures retrieval ranking quality, evidence-gate decisions, answer quality, and basic latency. The datasets are small and curated; results are useful for comparing the evaluated configurations, not for claiming production-level generalization.

## 2. Dataset and methodology

The development answer-evaluation set contains 28 questions: 21 answerable and 7 out of scope. The development retrieval set contains 21 answerable queries.

The consolidated candidate-label dataset contains 346 query-candidate judgments across 21 queries:

- Relevant: 67
- Irrelevant: 279

### Relevance-label assessment

The relevance assessment used a two-stage AI-assisted process:

1. ChatGPT and Gemini independently assessed all 346 candidate passages against their corresponding queries.
2. The two initial assessments agreed on 297 candidates (85.84%) and disagreed on 49 candidates.
3. A separate Gemini session adjudicated the 49 disagreements, assigning 23 as relevant and 26 as irrelevant.

The final dataset combines the 297 initially agreed labels with the 49 adjudicated labels.

These are AI-assisted and AI-adjudicated relevance judgments. They have not been independently verified by a human reviewer and should not be described as human-verified ground truth.

### Retrieval metrics

Recall is calculated over the judged candidate pool (the union of retrieved candidates), not over every passage in the full corpus. nDCG uses binary relevance labels.

The reported retrieval metrics are macro-averages across the 21 queries. Consequently, these are pool-based development metrics and may change after additional label review or candidate-pool expansion.

The held-out split is not part of this report. It must remain unused until retrieval, prompt, and evidence-gate decisions are frozen.

## 3. Retrieval benchmark — consolidated labels

The retrieval benchmark was recalculated using the consolidated 346-candidate dataset and its `final_relevant` relevance column.

| Retriever | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.7826 | 0.8916 | 0.8349 | 0.7514 | 0.7750 |
| BM25 | 0.7378 | 0.9027 | 0.8338 | 0.7810 | 0.8052 |
| **Hybrid** | **0.8577** | **0.9694** | **0.9206** | **0.8814** | **0.8957** |

Hybrid retrieval performed best across all five reported metrics.

Dense retrieval provides semantic matching, while BM25 contributes exact lexical matching for technical terms and identifiers. Reciprocal Rank Fusion (RRF) combines their ranked lists without requiring their raw retrieval scores to be directly comparable.

The results indicate that combining dense and lexical retrieval provides stronger ranking performance than either method alone on this development candidate pool.

### Comparison with the initial-label evaluation

The original retrieval evaluation used the initial AI-assisted labels (66 relevant and 280 irrelevant). After the independent assessments and subsequent adjudication, the consolidated dataset contained 67 relevant and 279 irrelevant candidates.

The retrieval ranking outputs and metric calculation methodology were not changed. Differences between the two result sets reflect the updated relevance judgments.

The original metrics are retained in `results/pilot_v2/`. The recalculated metrics are stored separately in `results/pilot_v2_final_labels/`.

## 4. Reranker comparison — earlier experiment

The cross-encoder reranker was previously evaluated on hybrid top-10 candidates.

| Metric | Hybrid top-10 | Hybrid + reranker |
|---|---:|---:|
| Recall@5 | 0.8906 | 0.7636 |
| MRR | 0.9206 | 0.9180 |
| nDCG@5 | 0.8827 | 0.8172 |
| nDCG@10 | 0.9157 | 0.8953 |

This comparison belongs to the earlier evaluation using the initial relevance-label version. It has not been recalculated using the consolidated labels and should not be directly compared with the updated Dense/BM25/Hybrid benchmark above.

In the earlier experiment, the reranker reduced Recall@5 and both reported nDCG metrics, with no MRR improvement. It was therefore not selected for the current retrieval baseline.

This conclusion applies to the evaluated model, candidate pool, and configuration; it is not a claim that reranking is generally ineffective.

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

The review labels were initially assistant-generated. They should not be described as independently human-validated benchmark results.

A reviewer should check expected facts against the source passages and verify whether cited passages support the associated claims. A valid citation ID or a passing citation-format validator does not establish semantic entailment.

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

- Original retrieval metrics: `retrieval_metrics_summary.csv` and `retrieval_metrics_per_query.csv`
- Earlier reranker comparison: `reranker_comparison_summary.csv` and `reranker_comparison_per_query.csv`
- Reranker latency: `reranker_latency_per_query.csv`
- Retrieval-only answer evaluation: `answer_eval_retrieval_dev.jsonl`
- Generated answers: `answer_eval_generate_dev.jsonl`
- Answer review: `answer_review_summary_dev.md`

The recalculated retrieval results using consolidated labels are stored in `results/pilot_v2_final_labels/`:

- `retrieval_metrics_summary.csv`
- `retrieval_metrics_per_query.csv`

Development datasets are under `evaluation/dataset/`. The consolidated full-passage relevance-label file is maintained locally. Held-out datasets and held-out construction inputs are intentionally excluded from the public repository and must not be used for tuning.

## 9. Limitations

- The corpus contains three documentation areas and is not representative of arbitrary engineering questions.
- The retrieval benchmark has 21 queries; aggregate metrics can move noticeably when a small number of labels or queries change.
- The consolidated relevance labels are AI-assisted and AI-adjudicated, not independently human-verified.
- Agreement between AI assessments does not establish label correctness.
- Answer-review scores are provisional and need claim-level human review.
- Pool-based recall is limited by the judged candidate pool and should not be interpreted as full-corpus recall.
- The evidence gate uses a fixed similarity threshold and requires testing on more varied in-scope and out-of-scope queries.
- The reranker comparison uses an earlier relevance-label version and has not been recalculated with the consolidated labels.
- Repeated latency runs are needed to understand performance variance.
