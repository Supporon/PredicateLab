# PredicateLab report — run_20261108_090135

- domain: `tsil`
- protocol hash: `143ea89bc8f2857d`
- identity hash: `46237f7af1a54444`
- snapshot: `snap_96cc5fb17be56224`
- data suitability: **ok**
- label expression: `Ref($close, -2) / Ref($close, -1) - 1` (2 future trading days)
- seeds: `[2025, 2026]`

## Evidence counts (status / state / stage)

| status | count |
|---|---|
| insufficient_evidence | 7 |
| rejected | 1 |
| screened_out | 32 |

| state | count |
|---|---|
| executable | 32 |
| insufficient | 7 |
| rejected | 1 |

| stage | count |
|---|---|
| discovery | 40 |

> Discovery-positive means `exploratory_positive`, never `independently_confirmed`. Confirmation and final results are listed separately.

## Rounds

### Round 0 (round_b0028a4a87215549) — `completed`

- proposed: 20, bound: 20, validated: 0, rejected: 1, insufficient: 3, screened_out: 16

| candidate | status | state | effect | CI | p | q | stability |
|---|---|---|---|---|---|---|---|
| cand_0b36af3 | insufficient_evidence | insufficient | -0.0041 | [-0.0098, +0.0006] | 0.0985 | 0.197 | 0.50 |
| cand_85651da | insufficient_evidence | insufficient | -0.0012 | [-0.0078, +0.0029] | 0.5025 | 0.5025 | 0.50 |
| cand_d419cd6 | rejected | rejected | -0.0024 | [-0.0106, +0.0022] | 0.08693 | 0.197 | 0.75 |
| cand_9f6a64b | insufficient_evidence | insufficient | -0.0034 | [-0.0075, +0.0009] | 0.1648 | 0.2198 | 0.25 |

### Round 1 (round_1ef72b8153b851c7) — `completed`

- proposed: 20, bound: 20, validated: 0, rejected: 0, insufficient: 4, screened_out: 16

| candidate | status | state | effect | CI | p | q | stability |
|---|---|---|---|---|---|---|---|
| cand_18836b9 | insufficient_evidence | insufficient | -0.0023 | [-0.0078, +0.0020] | 0.4433 | 0.5642 | 0.50 |
| cand_9e43404 | insufficient_evidence | insufficient | -0.0021 | [-0.0069, +0.0033] | 0.5642 | 0.5642 | 0.25 |
| cand_743d2b5 | insufficient_evidence | insufficient | -0.0022 | [-0.0063, +0.0017] | 0.3273 | 0.5642 | 0.50 |
| cand_8264f78 | insufficient_evidence | insufficient | -0.0022 | [-0.0073, +0.0017] | 0.5232 | 0.5642 | 0.00 |
