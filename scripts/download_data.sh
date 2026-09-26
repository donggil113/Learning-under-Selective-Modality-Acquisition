#!/usr/bin/env bash
# Re-create data/raw from open-access sources and verify SHA-256.
#   PhysioNet/CinC Challenge 2012 v1.0.0   ODC-By 1.0   (main real-data study)
#   MIMIC-IV Clinical Database Demo v2.2   ODbL 1.0     (A-vs-M audit only)
#   UCI Heart Disease (id 45)              CC BY 4.0    (BLOCKED_REALDATA audit only)
# Credentialed sources (MIMIC-IV full, MIMIC-CXR, MIMIC-IV-ECG/Note, eICU full) are
# NOT downloaded: no credentialed account/DUA exists in this project -> BLOCKED_REALDATA.
set -euo pipefail
cd "$(dirname "$0")/.."
R=data/raw
mkdir -p $R/challenge2012 $R/uci_heart $R/mimic_iv_demo/hosp $R/mimic_iv_demo/icu
PN=https://physionet.org/files
for f in set-a.tar.gz set-b.tar.gz set-c.tar.gz Outcomes-a.txt Outcomes-b.txt Outcomes-c.txt; do
  curl -sSf -o $R/challenge2012/$f $PN/challenge-2012/1.0.0/$f
done
(cd $R/challenge2012 && sha256sum -c - <<'SUMS'
8cb250f179cd0952b4b9ebcf8954b63d70383131670fac1cfee13deaa13ca920  set-a.tar.gz
b1637a2a423a8e76f8f087896cfc5fdf28f88519e1f4e874fbda69b2a64dac30  set-b.tar.gz
a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1  set-c.tar.gz
2613ea60ccda29f87571a7d6b09ad130858a8ccd66325bd073365925026883c2  Outcomes-a.txt
1305f29734da374a809483229a99f33a9230aa8d4fa74b1a2d6ad788136423d5  Outcomes-b.txt
0b32dcb7753c0f18f2f5fde498a28618b8879c5e527ce0dae8ad120dd7e205c9  Outcomes-c.txt
SUMS
for t in set-a set-b set-c; do tar xzf $t.tar.gz; done)
curl -sSf -o $R/uci_heart/heart+disease.zip https://archive.ics.uci.edu/static/public/45/heart+disease.zip
(cd $R/uci_heart && echo "b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc  heart+disease.zip" | sha256sum -c - && unzip -o -q heart+disease.zip)
for f in SHA256SUMS.txt hosp/labevents.csv.gz hosp/d_labitems.csv.gz hosp/admissions.csv.gz hosp/patients.csv.gz \
         hosp/poe.csv.gz hosp/poe_detail.csv.gz icu/icustays.csv.gz icu/d_items.csv.gz; do
  curl -sSf -o $R/mimic_iv_demo/$f $PN/mimic-iv-demo/2.2/$f
done
(cd $R/mimic_iv_demo && grep -E "hosp/(labevents|d_labitems|admissions|patients|poe|poe_detail)\.csv\.gz|icu/(icustays|d_items)\.csv\.gz" SHA256SUMS.txt | sha256sum -c -)
echo "data ready"
