"""Acquisition (A) versus record availability (M) on the open MIMIC-IV demo.

MIMIC-IV Clinical Database Demo v2.2 (100 patients, ODbL 1.0, no credentialing).
Unit: ICU stay (icu/icustays).  Cutoffs: intime + 24 h and + 48 h.

For each laboratory panel three indicators are computed per stay:
  collected  every component has a specimen with charttime in [intime, cutoff]
             (what a retrospective extract keyed on charttime reports)
  available  same, and the result was stored (storetime) by the cutoff
             (what a model running at the cutoff can actually read)
  ordered    (imaging / cardiology only) a NEW POE order of that subtype in the
             window that was not later discontinued (transaction_type == "New",
             discontinued_by_poe_id empty).  An order is not proof the exam was
             performed; the demo holds no image / waveform / report files at all,
             so performance cannot be verified here.
Modules absent from the demo (MIMIC-CXR, MIMIC-IV-ECG, notes) make every stay's
image / waveform "file" absent although the order table shows the tests were
ordered: a file-presence rule would code them as not acquired.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/mimic_iv_demo"
PANELS = {"ABG": [50820, 50821, 50818], "LACT": [50813], "LIVER": [50861, 50878, 50863, 50885], "TROP_T": [51003]}
ORDERS = {("Radiology", "General Xray"): "xray_ordered", ("Cardiology", "ECG"): "ecg_ordered",
          ("Cardiology", "Echo"): "echo_ordered"}


def run():
    icu = pd.read_csv(RAW / "icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    le = pd.read_csv(RAW / "hosp/labevents.csv.gz", parse_dates=["charttime", "storetime"],
                     usecols=["subject_id", "hadm_id", "itemid", "charttime", "storetime"])
    poe = pd.read_csv(RAW / "hosp/poe.csv.gz", parse_dates=["ordertime"])
    le = le[le.itemid.isin(sum(PANELS.values(), []))]
    out = {"n_stays": int(len(icu)), "n_patients": int(icu.subject_id.nunique()), "cutoffs": {}}
    for h in (24, 48):
        rows = []
        for _, s in icu.iterrows():
            cut = s.intime + pd.Timedelta(hours=h)
            L = le[(le.subject_id == s.subject_id) & (le.charttime >= s.intime) & (le.charttime <= cut)]
            r = {"stay_id": s.stay_id}
            for p, items in PANELS.items():
                col = all((L.itemid == i).any() for i in items)
                av = all(((L.itemid == i) & (L.storetime <= cut)).any() for i in items)
                r[f"{p}_collected"], r[f"{p}_available"] = col, av
            O = poe[(poe.subject_id == s.subject_id) & (poe.ordertime >= s.intime) & (poe.ordertime <= cut)
                    & (poe.transaction_type == "New") & poe.discontinued_by_poe_id.isna()]
            for (t, st), name in ORDERS.items():
                r[name] = bool(((O.order_type == t) & (O.order_subtype == st)).any())
            rows.append(r)
        df = pd.DataFrame(rows)
        res = {}
        for p in PANELS:
            c, a = df[f"{p}_collected"], df[f"{p}_available"]
            res[p] = {"collected": int(c.sum()), "available_at_cutoff": int(a.sum()),
                      "collected_not_available": int((c & ~a).sum()),
                      "frac_of_collected_not_available": float((c & ~a).sum() / max(c.sum(), 1))}
        for name in ORDERS.values():
            # the demo ships only hosp/ and icu/: no MIMIC-CXR, MIMIC-IV-ECG or echo report files exist
            res[name] = {"stays_with_new_undiscontinued_order": int(df[name].sum()), "stays_with_file_in_demo": 0}
        out["cutoffs"][f"{h}h"] = res
    # storetime lag distribution for the panels (hours)
    lag = (le.storetime - le.charttime).dt.total_seconds() / 3600
    out["storetime_lag_hours"] = {p: {q: float(np.nanpercentile(lag[le.itemid.isin(i)], q)) for q in (50, 90, 99)}
                                  for p, i in PANELS.items()}
    dst = ROOT / "results/audit"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "mimic_iv_demo_A_vs_M.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
