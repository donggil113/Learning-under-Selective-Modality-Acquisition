"""Parse PhysioNet/CinC Challenge 2012 records into a long table and per-stay summaries.

Documented facts this parser relies on (challenge-2012 v1.0.0 project page):
  * 12,000 adult ICU stays, stays < 48 h excluded; sets A, B, C of 4,000 each.
  * Every observation is time-stamped relative to ICU admission and lies in the
    first 48 h, so the prediction cutoff is t = 48 h for every stay.
  * "-1 indicates missing or unknown data"; a variable absent from a file was
    "not recorded" in the first 48 h -- the page does NOT say the test was not
    performed, so absence is treated as record-availability M, never as A.
  * Label: In-hospital_death (0/1) in Outcomes-{a,b,c}.txt.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(os.environ.get("N1_C2012", Path(__file__).resolve().parents[2] / "data/raw/challenge2012"))
CACHE = Path(__file__).resolve().parents[2] / "data/cache"

DESCRIPTORS = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]


def read_long() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "c2012_long.parquet"
    if out.exists():
        return pd.read_parquet(out)
    rows = []
    for s in "abc":
        d = RAW / f"set-{s}"
        for fn in sorted(os.listdir(d)):
            rid = int(fn.split(".")[0])
            df = pd.read_csv(d / fn)
            df["RecordID_file"] = rid
            df["set"] = s
            rows.append(df)
    long = pd.concat(rows, ignore_index=True)
    hh, mm = long["Time"].str.split(":", expand=True).astype(int).T.values
    long["t_hours"] = hh + mm / 60.0
    long = long.rename(columns={"RecordID_file": "rid"})
    long = long[["rid", "set", "t_hours", "Parameter", "Value"]]
    long.to_parquet(out)
    return long


def read_outcomes() -> pd.DataFrame:
    oc = pd.concat([pd.read_csv(RAW / f"Outcomes-{s}.txt").assign(set=s) for s in "abc"], ignore_index=True)
    return oc.rename(columns={"RecordID": "rid"})
