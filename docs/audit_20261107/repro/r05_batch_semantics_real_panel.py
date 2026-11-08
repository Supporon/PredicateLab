"""R05: quantify the composition of actual shuffled TSIL training batches.

A market / cross-stock predicate needs a defensible relation between a batch and
an intended cross-section. This reproducer follows worker._build_dataset on the
f2020 discovery fold and examines target (instrument, date) points in the first
four actual shuffled training batches.  It does not fit/train a model.

Usage: CUDA_VISIBLE_DEVICES= python -I r05_batch_semantics_real_panel.py <src> <qlib_provider>
"""
import json
import sys
from collections import Counter
from pathlib import Path

src = Path(sys.argv[1]).resolve()
provider = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(src))
sys.path.insert(0, str(src / "TSIL-main"))

import numpy as np
import qlib
import yaml

from predicatelab.core.protocol import validate_label_isolation
from predicatelab.domains.tsil.adapter import TSILAdapter
from predicatelab.domains.tsil import worker

cfg = yaml.safe_load((src / "configs" / "tsil_lstm_full.yaml").read_text())
cfg["data"]["provider_uri"] = str(provider)
cfg["data"]["tsil_root"] = str(src / "TSIL-main")
ad = TSILAdapter(cfg)
proto = ad.build_protocol(cfg)
contract = ad.data_contract(cfg)
folds, _, _ = validate_label_isolation(proto.folds, proto.confirmation_folds,
                                       proto.holdout_fold, ad.trading_calendar(cfg),
                                       contract.label_definition, contract.label_window)
fold = next(f for f in folds if f.fold_id == "f2020")
qlib.init(provider_uri=str(provider), region="cn")
job = {"fold": fold.model_dump(mode="json"), "data": ad._data_payload(),
       "provider_uri": str(provider), "variant": "baseline", "descriptor": {},
       "training": {"early_stop_frac": 0.1}}
dataset, _ = worker._build_dataset(job, batch_size=800, shuffle=True)
train = dataset.prepare("train")
np.random.seed(2025)  # QniverseModel(seed=2025) does this before train iteration
rows = []
for j, b in zip(range(4), train):
    idx = train.restore_index(b["index"])
    dates = [str(x[1])[:10] for x in idx]
    instruments = [str(x[0]) for x in idx]
    c = Counter(dates)
    rows.append({
        "batch": j,
        "n_samples": len(idx),
        "n_target_dates": len(c),
        "target_date_min": min(dates), "target_date_max": max(dates),
        "largest_same_date_group": max(c.values()),
        "median_samples_per_date": int(np.median(list(c.values()))),
        "dates_with_at_least_3_samples": sum(v >= 3 for v in c.values()),
        "n_instruments": len(set(instruments)),
    })
print(json.dumps({"fold": fold.model_dump(mode="json"), "batch_size": 800,
                  "shuffle": True, "batches": rows}, indent=2))
