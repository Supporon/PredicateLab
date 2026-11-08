# PredicateLab report — run_20261108_085937

- domain: `tsil`
- protocol hash: `e1ea38ce142cf375`
- identity hash: `cce8e4e35d012f70`
- snapshot: `snap_96cc5fb17be56224`
- data suitability: **ok**
- label expression: `Ref($close, -2) / Ref($close, -1) - 1` (2 future trading days)
- seeds: `[2025, 2026]`

## Evidence counts (status / state / stage)

| status | count |
|---|---|
| insufficient_evidence | 8 |
| invalid_structure | 5 |
| screened_out | 27 |

| state | count |
|---|---|
| executable | 27 |
| insufficient | 13 |

| stage | count |
|---|---|
| discovery | 40 |

> Discovery-positive means `exploratory_positive`, never `independently_confirmed`. Confirmation and final results are listed separately.

## Rounds

### Round 0 (round_069afe5d99a10a54) — `completed`

- proposed: 20, bound: 20, validated: 0, rejected: 0, insufficient: 4, screened_out: 11

| candidate | status | state | effect | CI | p | q | stability |
|---|---|---|---|---|---|---|---|
| cand_9cca36d | insufficient_evidence | insufficient | -0.0027 | [-0.0069, +0.0012] | 0.2801 | 0.3734 | 0.25 |
| cand_46403aa | insufficient_evidence | insufficient | -0.0007 | [-0.0071, +0.0046] | 0.2542 | 0.3734 | 0.50 |
| cand_5ca672d | insufficient_evidence | insufficient | -0.0037 | [-0.0087, +0.0007] | 0.1866 | 0.3734 | 0.00 |
| cand_fde5103 | insufficient_evidence | insufficient | -0.0024 | [-0.0076, +0.0027] | 0.5528 | 0.5528 | 0.25 |

### Round 1 (round_b6c8357f5a85e5f2) — `completed`

- proposed: 20, bound: 20, validated: 0, rejected: 0, insufficient: 4, screened_out: 16

| candidate | status | state | effect | CI | p | q | stability |
|---|---|---|---|---|---|---|---|
| cand_49c96c3 | insufficient_evidence | insufficient | -0.0006 | [-0.0056, +0.0029] | 0.4847 | 0.5801 | 0.25 |
| cand_9200349 | insufficient_evidence | insufficient | -0.0027 | [-0.0079, +0.0025] | 0.5241 | 0.5801 | 0.25 |
| cand_a00c6e5 | insufficient_evidence | insufficient | -0.0014 | [-0.0070, +0.0025] | 0.5801 | 0.5801 | 0.50 |
| cand_7e3f3f2 | insufficient_evidence | insufficient | -0.0031 | [-0.0083, +0.0013] | 0.3011 | 0.5801 | 0.25 |
