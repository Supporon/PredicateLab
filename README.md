# PredicateLab

A generic predicate discovery and evidence-validation platform.

The engine is domain-agnostic: a domain adapter supplies a data contract, a
candidate generator, and a fit/evaluate implementation, while the core handles
protocol execution, statistics, and evidence bookkeeping.

## Layout

- `predicatelab/core/` — domain-agnostic engine (schema, adapter, data, engine,
  hashing, protocol, stats, store)
- `predicatelab/domains/` — domain adapters: `tsil` (implemented end to end),
  `tabular` (generic reference adapter), `mineral` and `image` (extension
  contracts only)
- `configs/` — experiment configurations
- `docs/` — design notes
- `tests/` — tests
- `TSIL-main/` — upstream TSIL baseline (temporal stock investment benchmark)
- `artifacts/` — run outputs (ignored by git)

## Usage

```bash
python -m predicatelab.cli domains
python -m predicatelab.cli validate-config --config configs/tsil_lstm_quick.yaml
python -m predicatelab.cli discover --config configs/tsil_lstm_quick.yaml
```
