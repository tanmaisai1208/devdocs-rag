

\# Held-Out Evaluation — Pilot v2



\## 1. Objective and scope



This document records the held-out evaluation of the Engineering Knowledge Intelligence Platform.



The purpose of this evaluation is to assess the frozen development configuration on questions that were kept separate from development, retrieval experiments, and prompt tuning.



The held-out answer-evaluation dataset contains 12 questions:

\- 9 answerable / in-scope questions

\- 3 out-of-scope questions intended to test abstention



The evaluation measures evidence-gate behavior, answer completeness, citation support, abstention agreement, and latency.



The held-out results are reported separately from the development evaluation. No retrieval, prompt, or evidence-threshold changes were made based on these results.



\## 2. Evaluation configuration



The existing answer-evaluation runner was used without source-code modifications.



| Parameter | Value |

|---|---:|

| Retrieval top-K | 5 |

| Dense/BM25 candidate-K | 10 |

| Dense evidence threshold | 0.47 |

| Evaluation mode | Generate |

| Request delay | 1 second |

| Indexed chunks | 335 |

| Held-out questions | 12 |



The original held-out dataset was kept unchanged.



\## 3. Evidence-gate evaluation



| Question category | Questions | Gate passed | Gate rejected |

|---|---:|---:|---:|

| Answerable / in scope | 9 | 9 | 0 |

| Out of scope | 3 | 0 | 3 |

| \*\*Total\*\* | \*\*12\*\* | \*\*9\*\* | \*\*3\*\* |



The evidence gate made the expected decision for all 12 questions (12/12, 100%) on this held-out set.



All nine answerable questions passed the threshold, while all three out-of-scope questions were rejected.



This result is encouraging but is based on a small curated dataset. It does not establish generalization to unseen questions or broader engineering documentation.



An important limitation was observed in query a35. The maximum dense similarity score was approximately 0.5655, exceeding the 0.47 threshold, although the retrieved passages did not adequately explain the requested `target\_session\_attrs` behavior.



This demonstrates that a maximum similarity threshold alone does not guarantee that retrieved evidence is substantively sufficient to answer a question.



\## 4. Answer generation and abstention



Gemini generation was attempted for all nine answerable questions.



Eight questions produced non-abstained answers. Query a35 passed the evidence gate but the generator abstained because it could not find sufficient information.



All three out-of-scope questions were rejected by the evidence gate. Generation was skipped for these questions, avoiding unnecessary model calls.



| Question category | Expected abstentions | Actual abstentions |

|---|---:|---:|

| Answerable / in scope | 0 | 1 |

| Out of scope | 3 | 3 |

| \*\*Total\*\* | \*\*3\*\* | \*\*4\*\* |



Overall abstention agreement was 11/12 (91.7%).



The unexpected abstention on a35 is an answer-generation/evidence-coverage issue rather than an evidence-gate rejection.



\## 5. Expected-fact coverage



The nine answerable questions contain 24 expected facts in total.



The generated answers were assessed against those expected facts.



| Query | Expected facts | Facts addressed | Coverage |

|---|---:|---:|---:|

| a29 | 2 | 2 | 100% |

| a30 | 2 | 2 | 100% |

| a31 | 3 | 2 | 66.7% |

| a32 | 2 | 2 | 100% |

| a33 | 3 | 3 | 100% |

| a34 | 3 | 3 | 100% |

| a35 | 3 | 0 | 0% |

| a36 | 3 | 1 | 33.3% |

| a37 | 3 | 3 | 100% |

| \*\*Total\*\* | \*\*24\*\* | \*\*18\*\* | \*\*75.0%\*\* |



Among the eight non-abstained answers, 18 of 21 expected facts were addressed (85.7%).



\### Notable answer-quality findings



\*\*a31 — Python SyntaxError locations\*\*



The answer correctly explained `end\_lineno` and `end\_offset`, but omitted that these attributes were introduced in Python 3.10. This information was present in the top-ranked retrieved passage.



\*\*a35 — PostgreSQL target\_session\_attrs\*\*



The retrieved passages discussed `load\_balance\_hosts` and mentioned `target\_session\_attrs` in passing, but did not adequately explain the parameter's modes or selection behavior. The generator abstained rather than attempting an unsupported answer.



\*\*a36 — PostgreSQL channel\_binding\*\*



The answer correctly explained `require`, `prefer`, and `disable`, but omitted the SSL/SCRAM compatibility requirements and default behavior. All these details were present in the top-ranked retrieved passage.



The a31 and a36 findings indicate incomplete use of available evidence during answer generation. The a35 finding indicates inadequate retrieval evidence despite the evidence gate passing.



\## 6. Citation and factual-support review



All eight non-abstained answers had a `valid` citation-format status.



The cited chunk IDs were matched against the retrieved passages, and the cited source excerpts were inspected in relation to the generated claims.



No obvious unsupported claims were identified in this manual review of the supplied excerpts. However, this is a qualitative assessment, not an independently human-verified factuality benchmark.



Citation-format validation establishes that citation references are structurally valid. It does not, by itself, establish semantic entailment or complete answer coverage.



\## 7. Automated expected-fact similarity audit



The existing local fact-coverage audit was run against the held-out output.



It processed:

\- 9 answerable queries

\- 24 expected facts

\- 506 passage snippets



The best-matching top-five passage snippet met the following cosine-similarity thresholds:



| Similarity threshold | Facts with a candidate match |

|---|---:|

| 0.40 | 23/24 (95.8%) |

| 0.45 | 23/24 (95.8%) |

| 0.50 | 23/24 (95.8%) |

| 0.55 | 22/24 (91.7%) |

| 0.60 | 21/24 (87.5%) |



The fact associated with the lowest similarity was the list of supported `target\_session\_attrs` modes in a35, with a best similarity of approximately 0.3423.



These are semantic-similarity screening results, not verified factual-support measurements. Similarity does not prove that a passage contains sufficient evidence or entails an expected fact.



Only five passages were retrieved per query. The audit's top-13 fields therefore do not represent an independently retrieved top-13 result.



\## 8. Latency snapshot



| Metric | Result |

|---|---:|

| Retrieval latency, mean across 12 questions | 121.85 ms |

| Retrieval latency, median across 12 questions | 132.09 ms |

| Generation latency, mean across 9 generation attempts | 16.95 s |

| Generation latency, median across 9 generation attempts | 6.44 s |

| Maximum generation latency | 78.69 s |



Generation latency varied considerably. Query a36 had the longest generation latency at approximately 78.69 seconds.



Out-of-scope questions rejected by the evidence gate did not call Gemini and therefore had zero generation latency.



These are single-run observations, not service-level objectives or performance guarantees.



\## 9. Overall findings



The held-out evaluation produced the following results:



\- Evidence gate: 12/12 expected decisions.

\- Non-abstained answers: 8 of 9 answerable questions.

\- Expected-fact coverage: 18/24 (75.0%).

\- Answer-level abstention agreement: 11/12 (91.7%).

\- Citation-format validation: all eight non-abstained answers passed.

\- Manual cited-passage review: no obvious unsupported claims identified in the supplied excerpts.



The evaluation exposed two answer-completeness omissions and one retrieval-evidence limitation.



The evidence gate correctly rejected all three out-of-scope questions, but its score threshold did not guarantee sufficient semantic evidence for every in-scope question.



\## 10. Limitations



\- The held-out dataset contains only 12 questions and is not representative of arbitrary engineering queries.

\- Expected facts and source-section labels were proposed during benchmark construction and have not been independently human-verified.

\- Expected-fact coverage is a manual assessment against those expected facts, not a comprehensive measure of factual correctness.

\- Citation-format validation does not establish semantic entailment.

\- Automated embedding similarity is a screening aid, not a factual-support evaluator.

\- Latency measurements come from one run and show substantial generation-time variation.

\- No separate held-out retrieval relevance-label pool was created, so held-out Recall@K, MRR, and nDCG are not reported.



The held-out results are retained as an evaluation record. The observations should inform a future development cycle, not trigger parameter or prompt changes against this same held-out set.
