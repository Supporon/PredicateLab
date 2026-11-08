"""R01: windows built by TSIL `_create_ts_slices` cross instrument boundaries.
R02: evaluation mode (`eval()` sets drop_last=True) silently drops the tail batch.

Spec used as the oracle (written independently, not derived from the code):
  * a feature window used to predict (instrument i, date t) may only contain
    rows of instrument i (rows of other instruments must not enter);
  * evaluation must emit exactly one prediction per eligible sample.

Usage: python -I r01_r02_dataset_slicing.py <path/to/TSIL-main/src/dataset_ori_onehot.py>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ast_load import load_functions, load_methods, sha256  # noqa: E402

path = sys.argv[1]
ns = {"np": np, "pd": pd}
fns = load_functions(path, ["_create_ts_slices"], ns)
create = fns["_create_ts_slices"]
meths = load_methods(path, "MTSDatasetH", ["_get_slices", "__len__", "__iter__", "eval", "train"], ns)

out = {"audited_file": path, "sha256": sha256(path)}

# ---------------- R01: synthetic panel, 3 instruments, staggered entry ----------------
dates = pd.bdate_range("2020-01-01", periods=30)
rows = []
for inst, first in (("A", 0), ("B", 22), ("C", 25)):   # B and C join late (index entry / IPO)
    for d in dates[first:]:
        rows.append((inst, d))
idx = pd.MultiIndex.from_tuples(rows, names=["instrument", "datetime"])
seq_len = 5
slices = create(idx, seq_len)
inst_arr = idx.get_level_values(0).to_numpy()
date_arr = idx.get_level_values(1).to_numpy()
violations = []
for s in slices:
    target = inst_arr[s.stop - 1]
    window_insts = inst_arr[s.start:s.stop]
    if (window_insts != target).any():
        foreign = [(str(i), str(pd.Timestamp(d).date())) for i, d in
                   zip(window_insts, date_arr[s.start:s.stop]) if i != target]
        violations.append({"target": (str(target), str(pd.Timestamp(date_arr[s.stop - 1]).date())),
                           "foreign_rows": foreign})
out["R01"] = {
    "n_samples": int(len(slices)),
    "n_windows_with_foreign_instrument_rows": len(violations),
    "expected": 0,
    "examples": violations[:3],
    "note": "B's first windows contain A's LAST rows (A's latest dates, i.e. later than B's target date)",
}

# ---------------- R02: evaluation drops the trailing partial batch -----------------------
class Stub:  # minimal object carrying the attributes the audited methods read
    pass

for name, fn in meths.items():
    setattr(Stub, name, fn)

N, F, B = 2 * 800 + 123, 4, 800       # 1723 eligible samples, batch 800
st = Stub()
st._data = np.random.default_rng(0).normal(size=(N + 10, F + 1)).astype("float32")
st._label = np.arange(N + 10, dtype="float32")
st.use_timestamp = False
st.pin_memory = False
st.seq_len = 1
st.horizon = 1
st.batch_slices = np.array([slice(i, i + 1) for i in range(N)])
st.params = (B, False, True)          # train params as in MTSDatasetH.__init__
st.batch_size, st.drop_last, st.shuffle = st.params

import torch  # noqa: E402  (methods build tensors)
ns["torch"] = torch
ns["_to_tensor"] = lambda x: torch.tensor(x, dtype=torch.float)
ns["device"] = "cpu"

st.eval()
seen = []
for batch in st:
    seen.extend(batch["index"].tolist())
out["R02"] = {
    "eligible_samples": N,
    "emitted_in_eval": len(seen),
    "dropped": N - len(seen),
    "dropped_index_range": [min(set(range(N)) - set(seen)), max(set(range(N)) - set(seen))]
    if N != len(seen) else None,
    "eval_drop_last": bool(st.drop_last),
    "expected_dropped": 0,
    "note": "batch_slices are ordered by instrument then date, so the dropped tail is "
            "always the alphabetically last instrument(s)",
}
print(json.dumps(out, indent=2, default=str))
