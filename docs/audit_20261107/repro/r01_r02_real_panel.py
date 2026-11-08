"""R01/R02 prevalence on a real qlib CSI300 panel through the *actual* worker path.

Builds the f2020 discovery fold exactly as the engine would (protocol ->
effective fold -> label isolation -> worker._build_dataset -> MTSDatasetH) and
measures, per segment:
  * samples whose seq_len window contains rows of a different instrument;
  * of those, windows whose foreign rows are dated AFTER the target date;
  * samples dropped by eval-mode drop_last (batch_size=800) and their instruments.
No model is trained.  Only data up to the f2020 validation end (+2 label days)
is read; no confirmation (2023) or holdout (2024+) dates are touched.

Usage (from audit dir):
  CUDA_VISIBLE_DEVICES= venv/bin/python -I repro/r01_r02_real_panel.py <src_copy> <qlib_subset>
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

src = Path(sys.argv[1]).resolve()
provider = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(src))
sys.path.insert(0, str(src / "TSIL-main"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

cfg = yaml.safe_load((src / "configs" / "tsil_lstm_full.yaml").read_text())
cfg["data"]["provider_uri"] = str(provider)
cfg["data"]["tsil_root"] = str(src / "TSIL-main")

from predicatelab.core.protocol import validate_label_isolation  # noqa: E402
from predicatelab.domains.tsil.adapter import TSILAdapter  # noqa: E402
from predicatelab.domains.tsil import worker  # noqa: E402

ad = TSILAdapter(cfg)
proto = ad.build_protocol(cfg)
contract = ad.data_contract(cfg)
folds, _conf, _hold = validate_label_isolation(
    proto.folds, proto.confirmation_folds, proto.holdout_fold,
    ad.trading_calendar(cfg), contract.label_definition, contract.label_window)
fold = next(f for f in folds if f.fold_id == "f2020")

import qlib  # noqa: E402
qlib.init(provider_uri=str(provider), region="cn")

job = {"fold": fold.model_dump(mode="json"), "data": ad._data_payload(),
       "provider_uri": str(provider), "variant": "baseline", "descriptor": {},
       "training": {"early_stop_frac": 0.1}}
segments, split_meta = worker._split_segments(job)
dataset, _ = worker._build_dataset(job, batch_size=800, shuffle=False)

idx = dataset._index
inst = idx.get_level_values(0).to_numpy()
dt = idx.get_level_values(1).to_numpy()
out = {"fold": fold.model_dump(mode="json"), "segments": segments, "split_meta": split_meta,
       "n_rows_learn_frame": int(len(idx)), "n_instruments": int(len(set(inst)))}

for seg in ("train", "valid", "test"):
    part = dataset.prepare(seg)
    sl = part.batch_slices
    foreign = future = 0
    gap_windows = 0
    foreign_insts = Counter()
    for s in sl:
        t = s.stop - 1
        w_inst = inst[s.start:s.stop]
        mask = w_inst != inst[t]
        if mask.any():
            foreign += 1
            foreign_insts[str(inst[t])] += 1
            if (dt[s.start:s.stop][mask] > dt[t]).any():
                future += 1
        else:
            d = dt[s.start:s.stop]
            # same-instrument window spanning a membership/NaN-label gap (> 10 calendar days)
            if len(d) > 1 and (np.diff(d).astype("timedelta64[D]").astype(int) > 10).any():
                gap_windows += 1
    n = len(sl)
    part.eval()
    emitted = sum(len(b["index"]) for b in part)
    dropped_rows = list(range(emitted, n))
    dropped_insts = sorted({str(inst[sl[i].stop - 1]) for i in dropped_rows})
    dropped_dates = sorted({str(dt[sl[i].stop - 1])[:10] for i in dropped_rows})
    out[seg] = {
        "n_samples": int(n),
        "windows_with_foreign_instrument_rows": int(foreign),
        "share_foreign": round(foreign / n, 5) if n else None,
        "foreign_windows_with_future_dated_rows": int(future),
        "n_target_instruments_affected": len(foreign_insts),
        "same_instrument_windows_spanning_gap_gt10d": int(gap_windows),
        "eval_emitted": int(emitted),
        "eval_dropped": int(n - emitted),
        "eval_dropped_instruments": dropped_insts[:10] + (["..."] if len(dropped_insts) > 10 else []),
        "eval_n_dropped_instruments": len(dropped_insts),
        "eval_n_dates_touched_by_drop": len(dropped_dates),
    }
print(json.dumps(out, indent=2, default=str))
