"""Per-stay feature matrix, record-availability masks and label for Challenge 2012.

Eligible population E (fixed, nothing conditioned on availability):
  all 12,000 stays of sets A, B, C -- the challenge's own inclusion rule
  (adult ICU stays of >= 48 h; DNR/CMO not excluded).  No stay is dropped for
  missing data.
Prediction cutoff: 48 h after ICU admission (every observation lies in [0, 48]).
Label: In-hospital_death.

Modalities ("panels") whose availability is selective:
  ABG   = pH, PaO2, PaCO2            (arterial blood gas)
  ALINE = SysABP, DiasABP, MAP       (invasive arterial pressure, i.e. an arterial line)
  LACT  = Lactate
  LIVER = ALT, AST, ALP, Bilirubin   (hepatic panel)
M[k] = 1 iff every component of panel k has at least one valid (>= 0) value in
[0, 48] h.  A partially recorded panel is coarsened to M[k] = 0 and its values
are discarded (documented; 0.5 %, 0.4 %, 0 %, 3.7 % of stays).

M is RECORD availability at the cutoff.  The challenge page says variables were
"recorded at least once" and that "not all variables are available in all
cases"; it does not say an absent variable was not measured.  Absence is
therefore never read as "test not performed" (A = 0).  Structural examples in
these very files: MechVent only ever takes the value 1 (so its absence cannot be
read as 0 = not ventilated), and Troponin-I vs Troponin-T availability tracks an
assay change rather than a clinical decision.

Base covariates X0 (recorded for ~94-98 % of stays): age, sex, ICU type, and
48-h summaries of routine vitals and labs.  Their occasional absence is kept as
NaN and handled identically by every method (source-median fill + group
indicators); it is reported, not excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .load_c2012 import CACHE, read_long, read_outcomes

PANELS = {
    "ABG": ["pH", "PaO2", "PaCO2"],
    "ALINE": ["SysABP", "DiasABP", "MAP"],
    "LACT": ["Lactate"],
    "LIVER": ["ALT", "AST", "ALP", "Bilirubin"],
}
# summaries used for each panel's variables
PANEL_SUMMARIES = {
    "pH": ["min", "last"], "PaO2": ["min", "last"], "PaCO2": ["max", "last"],
    "SysABP": ["min", "mean"], "DiasABP": ["min", "mean"], "MAP": ["min", "mean"],
    "Lactate": ["max", "last"],
    "ALT": ["max"], "AST": ["max"], "ALP": ["max"], "Bilirubin": ["max"],
}
BASE_TS = {
    "HR": ["min", "max", "last"], "Temp": ["min", "max", "last"], "GCS": ["min", "last"],
    "BUN": ["max", "last"], "Creatinine": ["max", "last"], "HCT": ["min", "last"],
    "Platelets": ["min", "last"], "WBC": ["max", "last"], "Na": ["min", "max"], "K": ["min", "max"],
    "HCO3": ["min", "last"], "Glucose": ["max", "last"], "Mg": ["last"],
}
BASE_GROUPS = {"vitals": ["HR", "Temp", "GCS"], "labs": ["BUN", "Creatinine", "HCT", "Platelets", "WBC",
                                                        "Na", "K", "HCO3", "Glucose", "Mg"], "urine": ["Urine"]}
LOG = {"Lactate", "ALT", "AST", "ALP", "Bilirubin", "BUN", "Creatinine", "Glucose", "WBC", "Platelets", "Urine"}


@dataclass
class Cohort:
    X: np.ndarray                 # (n, p) NaN where a panel is not recorded / base value absent
    cols: list[str]
    base_idx: list[int]
    panel_idx: list[list[int]]    # per panel k, column indices
    panels: list[str]
    M: np.ndarray                 # (n, K) bool record availability at the cutoff
    y: np.ndarray                 # (n,) in-hospital death
    set: np.ndarray               # 'a' | 'b' | 'c'
    icu: np.ndarray               # 1 CCU, 2 CSRU, 3 MICU, 4 SICU
    rid: np.ndarray
    base_missing: np.ndarray      # (n,) any base covariate absent
    meta: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.panels)

    def pattern(self, M=None):
        M = self.M if M is None else M
        return (M.astype(int) * (1 << np.arange(self.K))).sum(1)


def _summaries(long: pd.DataFrame, var: str, stats: list[str]) -> pd.DataFrame:
    d = long[long.Parameter == var]
    g = d.sort_values("t_hours").groupby("rid").Value
    out = {}
    for s in stats:
        out[f"{var}_{s}"] = getattr(g, s)() if s != "last" else g.last()
    return pd.DataFrame(out)


def build(cache: bool = True) -> Cohort:
    out = CACHE / "c2012_cohort.npz"
    if cache and out.exists():
        z = np.load(out, allow_pickle=True)
        d = {k: (z[k].item() if z[k].dtype == object and z[k].shape == () else z[k]) for k in z.files}
        d["cols"] = [str(c) for c in d["cols"]]
        d["panels"] = [str(c) for c in d["panels"]]
        d["base_idx"] = [int(i) for i in d["base_idx"]]
        d["panel_idx"] = [[int(i) for i in p] for p in d["panel_idx"]]
        return Cohort(**d)
    long = read_long()
    oc = read_outcomes().set_index("rid")
    rids = oc.index.values
    desc = long[long.t_hours == 0].pivot_table(index="rid", columns="Parameter", values="Value", aggfunc="first")
    valid = long[(long.Value >= 0) & ~long.Parameter.isin(["RecordID", "Age", "Gender", "Height", "ICUType"])]

    frames = [desc.reindex(rids)[["Age", "Gender"]].where(lambda d: d >= 0)]
    icu = desc.reindex(rids)["ICUType"].astype(int).values
    frames.append(pd.DataFrame({f"icu{k}": (icu == k).astype(float) for k in (1, 2, 3, 4)}, index=rids))
    for v, st in BASE_TS.items():
        frames.append(_summaries(valid, v, st).reindex(rids))
    urine = valid[valid.Parameter == "Urine"].groupby("rid").Value.sum().rename("Urine_sum").reindex(rids)
    frames.append(urine.to_frame())
    base = pd.concat(frames, axis=1)
    base_cols = list(base.columns)

    avail = valid.groupby(["rid", "Parameter"]).size().unstack(fill_value=0).reindex(rids, fill_value=0) > 0
    M = np.stack([avail[v].all(axis=1).values for v in PANELS.values()], axis=1)
    pframes = []
    panel_cols = []
    for k, (p, vs) in enumerate(PANELS.items()):
        cols_k = []
        for v in vs:
            f = _summaries(valid, v, PANEL_SUMMARIES[v]).reindex(rids)
            f[~M[:, k]] = np.nan  # coarsen partial panels to "not recorded"
            pframes.append(f)
            cols_k += list(f.columns)
        panel_cols.append(cols_k)
    allf = pd.concat([base] + pframes, axis=1)
    for c in allf.columns:
        if c.split("_")[0] in LOG:
            allf[c] = np.log1p(allf[c].clip(lower=0))
    cols = list(allf.columns)
    base_idx = [cols.index(c) for c in base_cols]
    panel_idx = [[cols.index(c) for c in pc] for pc in panel_cols]
    X = allf.values.astype(float)
    base_missing = np.isnan(X[:, base_idx]).any(1)
    coh = Cohort(X=X, cols=cols, base_idx=base_idx, panel_idx=panel_idx, panels=list(PANELS), M=M,
                 y=oc["In-hospital_death"].values.astype(int), set=oc["set"].values.astype(str), icu=icu,
                 rid=rids, base_missing=base_missing,
                 meta={"cutoff_hours": 48, "n_eligible": len(rids), "label": "In-hospital_death"})
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(out, **{k: (np.array(v, dtype=object) if isinstance(v, (list, dict)) else v)
                         for k, v in coh.__dict__.items()})
    return coh


if __name__ == "__main__":
    c = build(cache=False)
    print(c.X.shape, c.M.mean(0), c.y.mean(), c.base_missing.mean())
    print("complete cases", c.M.all(1).mean(), "mort", c.y[c.M.all(1)].mean())
