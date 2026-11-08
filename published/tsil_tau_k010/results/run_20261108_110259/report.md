# PredicateLab report — run_20261108_110259

- domain: `tsil`
- protocol hash: `dca3763628dcf96c`
- identity hash: `de242309dcaab238`
- snapshot: `snap_96cc5fb17be56224`
- data suitability: **ok**
- label expression: `Ref($close, -2) / Ref($close, -1) - 1` (2 future trading days)
- seeds: `[2025, 2026]`

## Evidence counts (status / state / stage)

| status | count |
|---|---|
| insufficient_evidence | 4 |
| screened_out | 16 |

| state | count |
|---|---|
| executable | 16 |
| insufficient | 4 |

| stage | count |
|---|---|
| discovery | 20 |

> Discovery-positive means `exploratory_positive`, never `independently_confirmed`. Confirmation and final results are listed separately.

## Rounds

### Round 0 (round_3de2797b255e1ee3) — `completed`

- proposed: 20, bound: 20, validated: 0, rejected: 0, insufficient: 4, screened_out: 16

| candidate | status | state | effect | CI | p | q | stability |
|---|---|---|---|---|---|---|---|
| cand_7839cc5 | insufficient_evidence | insufficient | -0.0010 | [-0.0064, +0.0015] | 0.4969 | 0.6625 | 0.75 |
| cand_d907358 | insufficient_evidence | insufficient | -0.0009 | [-0.0058, +0.0017] | 0.6985 | 0.6985 | 0.25 |
| cand_a1924c4 | insufficient_evidence | insufficient | -0.0028 | [-0.0071, +0.0012] | 0.2659 | 0.5317 | 0.50 |
| cand_71b00c2 | insufficient_evidence | insufficient | -0.0007 | [-0.0056, +0.0012] | 0.257 | 0.5317 | 0.75 |
