# Decision criteria (fixed before the aggregate results were read)

Written after the literature synthesis and after a preview of the first 6 of 12
complete-case runs. It was written before the full aggregate, the cross-ICU
design, and the semi-synthetic runs were available. The criteria below do not
depend on any number that was still pending.

| Label | Condition |
|---|---|
| **GO** | All three hold: (a) the candidate method makes at least a *moderate* contribution that verified prior work does not already make; (b) on real data, artificial-dropout evaluation produces bootstrap-significant ranking reversals in most runs that neither class-balance adjustment nor sample-selection adjustment explains; (c) at equal information (V + U, no target labels), the candidate method decides more pairs correctly than the best baseline evaluator. |
| **NO_GO** | (a) fails. The method is an instance of prior work (the user's rule: do not cover this with a new loss). |
| **BLOCKED_REALDATA** | The acquisition meaning (test performed vs record present) or the label of the real data cannot be confirmed from its documentation. Recorded per claim, not per project. |

Applied per claim:

* A claim about **record availability M at the cutoff** is testable on
  PhysioNet 2012. That dataset documents "recorded at least once in the first
  48 h", a fixed cutoff, and a labeled outcome.
* A claim about **clinician acquisition A**, e.g. "tests are ordered for
  sicker patients", is BLOCKED_REALDATA on PhysioNet 2012 and UCI Heart. Neither
  says whether an absent value was not performed. The open MIMIC-IV demo shows
  that file presence and orders disagree.
* Credentialed data (MIMIC-IV full, MIMIC-CXR, MIMIC-IV-ECG/Note, eICU full) are
  BLOCKED_REALDATA: this project has no credentialed account or data use agreement.
