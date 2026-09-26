# N1 — When Does Modality Dropout Predict Deployment Risk?

*Learning under Selective Modality Acquisition*

**Question.** A model is evaluated with artificial modality dropout on a fully observed
population. When does that evaluation explain the model ranking on a deployment
population whose modalities are missing naturally, because they were acquired
selectively?

**Verdict and full report:** [`docs/REPORT.md`](docs/REPORT.md). The decision criteria
were fixed before the aggregate was read: [`docs/DECISION_CRITERIA.md`](docs/DECISION_CRITERIA.md).

## Layout

```
src/n1/
  theory.py            exact counterexamples (CE1, CE2), gap decomposition, sharp
                       bounds on a risk difference (+ prevalence LP, breakdown Γ);
                       `python -m src.n1.theory` certifies every claim numerically
  load_c2012.py        PhysioNet/CinC Challenge 2012 parser
  features_c2012.py    eligible population (all 12,000 stays), 48 h cutoff, 4 panels,
                       record-availability mask M (never read as "not performed")
  models.py            25 candidates: complete-case and natural-missingness strategies,
                       dropout, mask-aware, imputation, observed-only, reweighting,
                       Zhou et al. (2023) Alg. 1 and v-MAR baselines, same data and budget
  evaluators.py        selection rules from {labeled complete cases V, unlabeled
                       deployment sample U}; acquisition-aware bounds
  run_c2012.py         real-data study (6 role orders x seeds; cc and cross-ICU designs)
  run_semisynth.py     real features and labels with a KNOWN acquisition mechanism
  audit_mimic_demo.py  acquisition vs record availability on the open MIMIC-IV demo
  audit_uci_heart.py   why UCI Heart is BLOCKED_REALDATA
  summarize.py         all tables -> results/summary/{SUMMARY.md,summary.json}
  figures.py           results/figures/*.png
tests/                 theory and pipeline tests (incl. "unobserved panels are never read")
scripts/download_data.sh   open-access data with SHA-256 checks (data/ is git-ignored)
```

## Reproduce

```bash
pip install numpy scipy pandas scikit-learn statsmodels matplotlib joblib pyarrow tabulate pytest
scripts/download_data.sh
python -m src.n1.theory
python -m pytest -q tests
OMP_NUM_THREADS=1 python -m src.n1.run_c2012 --design cc  --seeds 2 --boot 200 --jobs 4
OMP_NUM_THREADS=1 python -m src.n1.run_c2012 --design icu --seeds 1 --boot 200 --jobs 4
OMP_NUM_THREADS=1 python -m src.n1.run_semisynth --reps 8 --jobs 4
python -m src.n1.audit_mimic_demo && python -m src.n1.audit_uci_heart
python -m src.n1.summarize && python -m src.n1.figures
```

## Data licenses

* PhysioNet/CinC Challenge 2012 v1.0.0: ODC-By 1.0.
* MIMIC-IV Clinical Database Demo v2.2: ODbL 1.0.
* UCI Heart Disease: CC BY 4.0. Principal investigators: A. Janosi, W. Steinbrunn,
  M. Pfisterer, R. Detrano.

No patient-level data or derived patient rows are committed. The per-run prediction arrays
(`results/c2012/arrays_*.npz`, about 2.4 MB each) are git-ignored and are re-created by `run_c2012`.
