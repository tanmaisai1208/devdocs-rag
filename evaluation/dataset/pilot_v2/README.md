# Pilot v2 development evaluation files

This directory contains development-side additions and review artifacts for the 28-question answer evaluation and 21-query retrieval benchmark.

## Development files

- `answer_eval_dev_additions.jsonl`: 13 new development answer-evaluation records added to the original 15-question dataset.
- `queries_dev_additions.jsonl`: 9 new development retrieval queries added to the original 12-query dataset.
- `answer_review_pilot_v2_dev_reviewed.csv`: development answer-review sheet used for the current provisional summary.
- `candidate_labels_dev.csv`: query-candidate relevance judgments used for the current retrieval comparison. These labels are AI-assisted and have not been independently human-verified.

The combined development datasets are stored in `evaluation/dataset/answer_eval_pilot_v2_dev.jsonl` and `evaluation/dataset/queries_pilot_v2_dev.jsonl`.

## Evaluation caution

The answer-review scores and candidate relevance labels are provisional. Independently verify expected facts, claim-to-citation support, and passage-level relevance before describing these results as human-validated.

Held-out data and local split-construction materials are not included in the public repository. Do not use held-out questions, labels, or outputs for development tuning. The reports in `results/pilot_v2/` cover development data only.
