#!/usr/bin/env python
"""Block until all five tau-sweep runs reach a terminal round state.

Prints the final progress table, then exits 0.  Exits 1 on timeout.
Writes nothing into the repo outside artifacts/.
"""

from __future__ import annotations

import pathlib
import sqlite3
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
ARMS = ["k005", "k010", "k020", "k050", "learned"]
TERMINAL_RUN = {"completed", "failed", "aborted"}
TIMEOUT_S = 150 * 60


def state(slug: str) -> tuple[str, list[tuple]]:
    db = REPO / "artifacts" / f"tsil_tau_{slug}" / "predicatelab.sqlite"
    if not db.exists():
        return ("missing", [])
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
        row = con.execute(
            "select run_id, status from runs order by started_at desc limit 1"
        ).fetchone()
        if not row:
            return ("empty", [])
        rounds = con.execute(
            "select round_index, status, n_validated from rounds where run_id=?"
            " order by round_index", (row[0],),
        ).fetchall()
        con.close()
        return (row[1], rounds)
    except Exception as exc:  # db mid-write
        return (f"unreadable:{exc}", [])


def main() -> int:
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        states = {s: state(s) for s in ARMS}
        done = all(
            st in TERMINAL_RUN or (rounds and all(r[1] != "running" for r in rounds))
            for st, rounds in states.values()
        )
        if done:
            print("ALL TAU ARMS FINISHED")
            print(" ".join(f"{s}={states[s][0]}" for s in ARMS))
            subprocess.run(
                [sys.executable, str(REPO / "artifacts" / "exp_logs" / "progress_tau.py")],
                check=False,
            )
            for s in ARMS:
                print(f"--- {s}")
                log = REPO / "artifacts" / "exp_logs" / f"tau_{s}.log"
                if log.exists():
                    print(log.read_text().strip())
            return 0
        time.sleep(60)
    print("TIMEOUT waiting for tau arms")
    for s in ARMS:
        print(s, state(s))
    return 1


if __name__ == "__main__":
    sys.exit(main())
