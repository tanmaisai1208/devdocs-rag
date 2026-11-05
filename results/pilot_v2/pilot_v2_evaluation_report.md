# Engineering Knowledge Intelligence — Pilot v2 Evaluation Report

**Evaluation split:** Development only
**Dataset:** Pilot v2 development set
**Questions:** 28 (21 answerable, 7 expected abstentions)
**Status:** Development evaluation complete; held-out evaluation not yet run
**Evaluation date:** 5 November 2026

---

## 1. Executive summary

Pilot v2 evaluated the retrieval, reranking, evidence-gating, answer-generation, citation-validation, and answer-quality stages of the RAG application on 28 development questions.

The main findings are:

- **Hybrid retrieval (dense + BM25 with reciprocal-rank fusion) performed best** among the three retrievers evaluated.
- **The tested cross-encoder reranker reduced ranking quality** on the current candidate set, so it should remain out of the application’s default retrieval path for now.
- **The evidence gate made the expected decision on all 28 development questions** at the tested dense-score threshold of 0.47: it passed all 21 answerable questions and rejected all 7 out-of-scope questions.
- The generation run produced **21 answers and 7 abstentions**. All 21 generated answers passed the citation-format/validation check.
- The AI-assisted answer review scored expected-fact coverage at **90.5%**, factual correctness at **100%**, and citation support at **97.6%**. Four answers had incomplete expected-fact coverage.

These are development-set results, not a final generalization claim. Retrieval relevance labels and answer-review scores were AI-assisted and have not been independently human-verified. The held-out split remains untouched and must be evaluated only after the development configuration is frozen.

## 2. Dataset and evaluation protocol

### 2.1 Development answer-evaluation set

| Split | Total questions | Answerable | Expected abstentions |
|---|---:|---:|---:|
| Pilot v2 development | 28 | 21 | 7 |

The 21 answerable questions cover Python exception behavior, SQLAlchemy connection pools and errors, PostgreSQL connection configuration, and related engineering documentation. The remaining 7 are intentionally out of scope for the indexed documentation.

### 2.2 Retrieval relevance-label set

The retrieval candidate export contained:

- 21 development retrieval queries.
- 335 unique candidate chunks.
- 346 query–candidate rows, formed from the union of dense top-10, BM25 top-10, and hybrid top-10 candidates.
- 66 candidates labeled relevant and 280 labeled irrelevant.

The relevance labels were AI-assisted. They are not independently human-verified ground truth. Query q10 had an unusually high number of relevant labels (11 of 15 candidates) and was identified as a candidate for additional spot-checking. Reported retrieval recall is therefore **pool-based recall over the judged candidate pool**, not corpus-wide recall.

### 2.3 Evaluation stages

The development evaluation covered:

1. Dense, BM25, and hybrid retrieval.
2. Hybrid candidate reranking using the tested cross-encoder.
3. Evidence-gate decisions for answerable and out-of-scope questions.
4. Answer generation and citation validation.
5. AI-assisted review of expected-fact coverage, factual correctness, citation support, abstention behavior, latency, and token usage.

## 3. Retrieval evaluation

### 3.1 Dense vs. BM25 vs. hybrid retrieval

The candidate-pool retrieval evaluation produced the following aggregate metrics:

| Metric | Dense | BM25 | Hybrid |
|---|---:|---:|---:|
| Recall@5 | 0.7899 | 0.7849 | **0.8813** |
| Recall@10 | 0.8913 | 0.8988 | **0.9845** |
| MRR | 0.8349 | 0.8746 | **0.9206** |
| nDCG@5 | 0.7499 | 0.8120 | **0.8827** |
| nDCG@10 | 0.7747 | 0.8285 | **0.9085** |

**Finding:** Hybrid retrieval ranked best on all five reported metrics. Combining semantic retrieval with lexical retrieval appears useful for this engineering-documentation query set, where some questions depend on exact API names, parameters, exception names, and error-message terms while others need semantic matching.

**Metric caveat:** Recall is calculated against relevant items found in the union candidate pool, not against every chunk in the full corpus. The binary relevance labels are AI-assisted and may contain labeling errors. These values should be reported as preliminary pool-based metrics.

### 3.2 Cross-encoder reranker evaluation

The reranker was evaluated by reranking the hybrid top-10 candidates. The comparison was:

| Metric | Hybrid top-10 baseline | Hybrid + reranker |
|---|---:|---:|
| Recall@5 | **0.8906** | 0.7636 |
| MRR | **0.9206** | 0.9180 |
| nDCG@5 | **0.8827** | 0.8172 |
| nDCG@10 | **0.9157** | 0.8953 |

The hybrid top-10 candidate set had a 1.0 candidate hit rate (at least one labeled relevant candidate was present for each query). However, the reranker moved some relevant passages down the ranking. Examples included q09 (PostgreSQL `service` configuration) and q16 (Python `SyntaxError` source-location attributes).

**Finding:** The tested reranker degraded Recall@5 and nDCG, with MRR also slightly lower. Keep hybrid retrieval as the current baseline and do not place this reranker in the default application pipeline. Reconsider reranking only after evaluating a better-suited cross-encoder, candidate text representation, or model configuration on development data.

**Comparison caveat:** The hybrid top-10 baseline values in this reranker table come from the reranker-metrics evaluation and are not identical to every value in the separate dense/BM25/hybrid candidate-pool summary above. Keep the two tables tied to their respective evaluation scripts and do not combine the figures as if they came from one identical metric run.

## 4. Evidence-gate evaluation

The answer runner used the following development configuration:

| Setting | Value |
|---|---:|
| Dense evidence threshold | 0.47 |
| Candidate retrieval count (`candidate_k`) | 10 |
| Returned context count (`top_k`) | 5 |

At this threshold, the evidence gate produced:

| Expected class | Questions | Gate decision correct | Result |
|---|---:|---:|---|
| Answerable | 21 | 21 | All passed |
| Expected abstention / out of scope | 7 | 7 | All rejected |
| **Total** | **28** | **28** | **28/28 agreement** |

**Finding:** The gate made the expected decision on every development question at threshold 0.47. This is encouraging, but the sample is small and the threshold was assessed on development data. The result does not establish 100% abstention performance on unseen questions.

## 5. Answer generation and citation validation

The completed development generation run contains:

| Outcome | Questions |
|---|---:|
| Generated answers | 21 |
| Evidence-gate abstentions | 7 |
| Total | 28 |

All 21 generated answers completed successfully and passed the citation validator. The 7 rejected questions did not call the generation model and returned the configured insufficient-evidence abstention response.

Citation validation checks the application's citation structure and referenced evidence. It does **not**, by itself, prove that every factual claim is entailed by the cited passage. Semantic citation support was assessed separately in the answer-review stage.

## 6. Answer-quality review

The consolidated answer-review CSV was cross-checked using two AI assessments. Their numerical scores and abstention decisions agreed across all 28 questions. This is an AI-assisted cross-check, not human verification.

### 6.1 Aggregate answer-quality scores

Scores use a 0–2 scale and apply to the 21 expected-answerable questions.

| Metric | Points | Percentage | Mean score |
|---|---:|---:|---:|
| Expected-fact coverage | 38/42 | **90.5%** | 1.81/2 |
| Factual correctness | 42/42 | **100.0%** | 2.00/2 |
| Citation support | 41/42 | **97.6%** | 1.95/2 |

These values reflect the submitted AI review labels. They should be described as provisional, AI-assisted review scores until a human review or a stronger independent adjudication process is completed.

### 6.2 Answers with incomplete expected-fact coverage

Four answers received a fact-coverage score of 1 rather than 2:

| Query | Missing or incomplete expected information | Scores (coverage / correctness / citation support) |
|---|---|---:|
| a04 | Did not explicitly name `pool_size` and `max_overflow` as the pool-capacity controls. | 1 / 2 / 2 |
| a10 | Omitted `sslrootcert` and `sslcert`; the expected certificate-verification details were not fully covered. | 1 / 2 / 1 |
| a17 | Explained how `except*` matches subgroups but omitted the purpose of exception groups: raising multiple unrelated exceptions together. | 1 / 2 / 2 |
| a24 | Identified unsafe sharing of `Session` / `AsyncSession` but omitted the expected consequences, including invalid object state and difficult-to-debug or driver-level errors. | 1 / 2 / 2 |

The answers were assessed as factually correct but incomplete. This distinction matters: high factual-correctness scores do not mean all requested information was included.

### 6.3 Abstention quality

| Review outcome | Questions |
|---|---:|
| Correct answer | 21 |
| Correct abstention | 7 |
| Incorrect decision | 0 |
| **Agreement** | **28/28 (100%)** |

All 7 expected out-of-scope questions were correctly abstained from, and all 21 answerable questions received answers.

## 7. Performance evaluation

The answer-review summary reported the following development-run performance:

| Metric | Result |
|---|---:|
| Retrieval latency — median | 143.3 ms |
| Retrieval latency — mean | 142.1 ms |
| Generation latency — median | 4,260.5 ms (4.26 s) |
| Generation latency — mean | 6,285.2 ms (6.29 s) |
| Total tokens across generated answers | 40,807 |
| Mean tokens per generated answer | 1,943.2 |
| Median tokens per generated answer | 1,855 |

Generation latency is recorded for the 21 answerable questions that invoked generation. The runner's 13-second request pacing delay was not included in `generation_latency_ms`; therefore, these values represent measured generation execution latency rather than total wall-clock time between requests. Retrieval latency is summarized over all 28 questions.

## 8. Development configuration decision

Based on the development results, retain the following configuration as the frozen candidate for held-out evaluation:

- Hybrid retrieval combining dense and BM25 candidates.
- Reciprocal-rank fusion (RRF) for hybrid ranking.
- `candidate_k = 10` and `top_k = 5` in the answer-evaluation runner.
- Evidence-gate dense threshold `0.47`.
- Current answer-generation and citation-validation pipeline.
- **Do not enable the tested cross-encoder reranker by default**, because it reduced retrieval ranking metrics on this development set.

Record the exact code revision, embedding model, vector collection/index state, BM25 configuration, RRF settings, evidence-gate implementation, generation model/version, prompt, and dependency versions before running held-out evaluation. If any of these change after freezing, document the change and avoid describing the held-out run as an untouched evaluation of the original configuration.

## 9. Limitations and interpretation

1. **Development-only evaluation:** All results in this report come from the pilot v2 development split. They are not held-out results and should not be presented as evidence of generalization.
2. **AI-assisted relevance labels:** The 66 relevant / 280 irrelevant retrieval labels have not been independently human-verified. Pool-based recall and ranking metrics inherit any label errors.
3. **AI-assisted answer review:** The two AI assessments agreed, but agreement between models is not equivalent to ground truth. The 100% factual-correctness result is a provisional score, not a guarantee that all answers are correct.
4. **Small sample:** Twenty-eight questions are useful for a pilot but insufficient for broad claims about domain coverage, abstention reliability, or production readiness.
5. **Pool-based recall:** Retrieval recall is measured over the judged candidate pool rather than the complete corpus.
6. **Reranker generality:** The negative reranker result applies to the tested model, candidate set, and evaluation setup. It does not prove that reranking is inherently unhelpful.
7. **Performance environment:** Latency depends on local hardware, model/API service conditions, network behavior, and concurrency. The reported values are observations from this run, not service-level guarantees.

## 10. Next steps

1. Preserve this development report and all underlying JSONL, CSV, and metric artifacts.
2. Freeze and record the development configuration and code revision.
3. Do not tune on or inspect held-out relevance labels or answer-review outcomes.
4. Run the held-out answer evaluation on the 12-question held-out split (9 answerable, 3 expected abstentions) using the frozen configuration.
5. Evaluate held-out retrieval metrics separately using the held-out candidate labels, and report the limitations of the small sample.
6. After completing the held-out run, produce a final report that presents development and held-out results separately.

---

## 11. Artifact index

Key pilot v2 artifacts currently identified:

| Artifact | Path |
|---|---|
| Development answer dataset | `evaluation/dataset/answer_eval_pilot_v2_dev.jsonl` |
| Held-out answer dataset (not yet evaluated) | `evaluation/dataset/answer_eval_pilot_v2_heldout.jsonl` |
| Development retrieval queries | `evaluation/dataset/queries_pilot_v2_dev.jsonl` |
| Held-out retrieval queries (not yet evaluated) | `evaluation/dataset/queries_pilot_v2_heldout.jsonl` |
| Retrieval relevance labels | `evaluation/dataset/pilot_v2/candidate_labels_dev.csv` |
| Candidate labels with reranker ranks | `evaluation/dataset/pilot_v2/candidate_labels_dev_with_reranker.csv` |
| Consolidated development generation results | `results/pilot_v2/answer_eval_generate_dev.jsonl` |
| Consolidated answer-review CSV | `evaluation/dataset/pilot_v2/answer_review_pilot_v2_dev_reviewed.csv` |
| Answer-review summary | `results/pilot_v2/answer_review_summary_dev.md` |
| Retrieval metrics per query | `results/pilot_v2/retrieval_metrics_per_query.csv` |
| Retrieval metrics summary | `results/pilot_v2/retrieval_metrics_summary.csv` |
| Reranker latency per query | `results/pilot_v2/reranker_latency_per_query.csv` |

**Report status:** Development evaluation consolidated. Held-out evaluation is pending.
