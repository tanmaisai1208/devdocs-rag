# RAG answer-review summary

- **Input file:** `evaluation\dataset\pilot_v2\answer_review_pilot_v2_dev_reviewed.csv`
- **Questions in CSV:** 28
- **Expected-answerable questions:** 21
- **Expected-abstention questions:** 7
- **Rows with unknown `should_abstain`:** 0

> **Important:** This report calculates summaries from the CSV's review labels. It does not independently validate those labels. If the CSV contains assistant-generated first-pass reviews, have a human check them before reporting these values as benchmark results.

## Answer-review scores (provisional)

Scores use a 0–2 scale and are summarized on expected-answerable questions only. Blank values are excluded.

| Metric | Summary |
|---|---|
| Expected-fact coverage | 38/42 points (90.5%), mean 1.81/2 across 21 reviewed answers |
| Factual correctness | 42/42 points (100.0%), mean 2.00/2 across 21 reviewed answers |
| Citation support | 41/42 points (97.6%), mean 1.95/2 across 21 reviewed answers |

## Abstention and execution checks

- **Expected vs. actual abstention agreement:** 28/28 questions
- **Review abstention labels:** {'correct_answer': 21, 'correct_abstention': 7}

## Performance

- **Retrieval latency:** median 143.3 ms; mean 142.1 ms; n=28
- **Generation latency (answerable questions with a recorded generation):** median 4,260.5 ms; mean 6,285.2 ms; n=21
- **Tokens per generated answer:** total 40,807; mean 1,943.2; median 1,855; n=21

## Per-question review

| ID | Should abstain | Did abstain | Fact coverage | Factual correctness | Citation support | Abstention review |
|---|---:|---:|---:|---:|---:|---|
| a01 | False | False | 2 | 2 | 2 | correct_answer |
| a02 | False | False | 2 | 2 | 2 | correct_answer |
| a03 | False | False | 2 | 2 | 2 | correct_answer |
| a04 | False | False | 1 | 2 | 2 | correct_answer |
| a05 | False | False | 2 | 2 | 2 | correct_answer |
| a06 | False | False | 2 | 2 | 2 | correct_answer |
| a07 | False | False | 2 | 2 | 2 | correct_answer |
| a08 | False | False | 2 | 2 | 2 | correct_answer |
| a09 | False | False | 2 | 2 | 2 | correct_answer |
| a10 | False | False | 1 | 2 | 1 | correct_answer |
| a11 | False | False | 2 | 2 | 2 | correct_answer |
| a12 | False | False | 2 | 2 | 2 | correct_answer |
| a13 | True | True | NA | NA | NA | correct_abstention |
| a14 | True | True | NA | NA | NA | correct_abstention |
| a15 | True | True | NA | NA | NA | correct_abstention |
| a16 | False | False | 2 | 2 | 2 | correct_answer |
| a17 | False | False | 1 | 2 | 2 | correct_answer |
| a18 | False | False | 2 | 2 | 2 | correct_answer |
| a19 | False | False | 2 | 2 | 2 | correct_answer |
| a20 | False | False | 2 | 2 | 2 | correct_answer |
| a21 | False | False | 2 | 2 | 2 | correct_answer |
| a22 | False | False | 2 | 2 | 2 | correct_answer |
| a23 | False | False | 2 | 2 | 2 | correct_answer |
| a24 | False | False | 1 | 2 | 2 | correct_answer |
| a25 | True | True | NA | NA | NA | correct_abstention |
| a26 | True | True | NA | NA | NA | correct_abstention |
| a27 | True | True | NA | NA | NA | correct_abstention |
| a28 | True | True | NA | NA | NA | correct_abstention |

## Recommended interpretation

Treat these numbers as a snapshot of the current review sheet. Keep the original generation JSONL and this CSV as separate artifacts. After manually verifying the review labels, rerun this script to regenerate the summary.
