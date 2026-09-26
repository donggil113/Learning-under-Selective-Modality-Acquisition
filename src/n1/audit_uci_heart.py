"""Why UCI Heart Disease is BLOCKED_REALDATA for acquisition claims.

Labels are documented (num = angiographic disease status, > 50 % narrowing),
but missing values are only described as "Several. Distinguished with value
-9.0" -- the documentation never says whether an absent value means the test
was not performed, not transcribed, or lost ("my original copy of the database
appears to be corrupted", WARNING file).  Switzerland additionally codes serum
cholesterol as 0 for every patient, an undocumented missing code.  Missingness
is almost entirely site-level (structural), not per-patient acquisition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/uci_heart"
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
SITES = {"cleveland": "processed.cleveland.data", "hungarian": "processed.hungarian.data",
         "switzerland": "processed.switzerland.data", "va": "processed.va.data"}


def run():
    out = {}
    for s, f in SITES.items():
        d = pd.read_csv(RAW / f, header=None, names=COLS, na_values=["?", "-9", "-9.0"])
        miss = d.isna().mean().round(3).to_dict()
        miss["chol==0"] = float((d.chol == 0).mean().round(3))
        out[s] = {"n": int(len(d)), "disease_rate(num>0)": float((d.num > 0).mean().round(3)), "missing_frac": miss}
    dst = ROOT / "results/audit"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "uci_heart_missingness.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = run()
    tab = pd.DataFrame({s: {**{"n": v["n"], "disease": v["disease_rate(num>0)"]}, **{k: v["missing_frac"][k] for k in ["chol", "chol==0", "fbs", "slope", "ca", "thal"]}} for s, v in r.items()}).T
    print(tab.to_markdown())
