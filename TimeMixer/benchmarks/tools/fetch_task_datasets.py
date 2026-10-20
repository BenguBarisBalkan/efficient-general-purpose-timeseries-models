"""
Fetch the task datasets that aren't shipped with this repo, into ../../dataset/.

Downloads the two FULLY-OPEN datasets used to validate the new SparseTSF tasks:
  • PSM             (anomaly detection) -> dataset/PSM/{train,test,test_label}.csv
  • JapaneseVowels  (classification)    -> dataset/JapaneseVowels/JapaneseVowels_{TRAIN,TEST}.ts

Idempotent: files already present are skipped. Stdlib only (urllib + zipfile).

    venv/Scripts/python.exe benchmarks/tools/fetch_task_datasets.py

── MSL and SMAP: built from a manual download ──
NASA's original link is dead, so download the Kaggle mirror of NASA's release by hand
(patrickfleith/nasa-anomaly-detection-dataset-smap-msl; Kaggle needs a login) and leave the zip at
dataset/archive.zip or dataset/_raw/nasa_smap_msl.zip. build_msl_smap() rebuilds the TSLib-format
arrays from it and refuses to write them unless their sizes equal the published ones.

── Datasets this script does NOT fetch (documented so the loaders can be satisfied manually) ──
The TimesNet/TSLib *preprocessed* bundle is distributed via Google Drive / Tsinghua Cloud (not
reliably scriptable). Files with these exact names work with data_provider/data_loader.py as-is:

  dataset/MSL/   MSL_train.npy   MSL_test.npy   MSL_test_label.npy
  dataset/SMAP/  SMAP_train.npy  SMAP_test.npy  SMAP_test_label.npy
  dataset/SMD/   SMD_train.npy   SMD_test.npy   SMD_test_label.npy
  dataset/SWaT/  swat_train2.csv swat2.csv        (SWaT is GATED — request access at iTrust/SUTD)

Source for the bundle: the "Time-Series-Library" / TimesNet dataset release
(github.com/thuml/Time-Series-Library — see its README "Datasets" link).

Extra UEA classification sets are drop-in: download "<Name>.zip" from
https://timeseriesclassification.com/aeon-toolkit/<Name>.zip and unzip the non-`_eq_`
<Name>_TRAIN.ts / <Name>_TEST.ts into dataset/<Name>/.
"""

import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2].parent / "dataset"

PSM_BASE = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"
PSM_FILES = ["train.csv", "test.csv", "test_label.csv"]

UEA_URL_TMPL = "https://timeseriesclassification.com/aeon-toolkit/{}.zip"

# The 10 standard UEA multivariate sets used by TSLib / TimesNet / TimeMixer++ for classification.
UEA_NAMES = [
    "EthanolConcentration", "FaceDetection", "Handwriting", "Heartbeat", "JapaneseVowels",
    "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
    "UWaveGestureLibrary",
]


def _get(url: str) -> bytes:
    """Fetch bytes. Tries urllib, then falls back to curl.

    timeseriesclassification.com rejects urllib's requests with HTTP 403 regardless of
    User-Agent, but serves curl fine — so curl is the working transport for the UEA archives.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    except Exception:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            dst = os.path.join(td, "dl.bin")
            p = subprocess.run(["curl", "-sSL", "-m", "600", "-o", dst, url],
                               capture_output=True, text=True)
            if p.returncode != 0 or not os.path.exists(dst):
                raise RuntimeError(f"curl failed ({p.returncode}): {p.stderr.strip()[:200]}")
            with open(dst, "rb") as f:
                return f.read()


def fetch_psm() -> None:
    out = DATASET_ROOT / "PSM"
    out.mkdir(parents=True, exist_ok=True)
    for f in PSM_FILES:
        dst = out / f
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] PSM/{f}")
            continue
        print(f"[get ] PSM/{f} ...", flush=True)
        dst.write_bytes(_get(f"{PSM_BASE}/{f}"))
    print(f"[ok  ] PSM -> {out}")


def fetch_uea(names=None) -> None:
    for name in (names or UEA_NAMES):
        out = DATASET_ROOT / name
        train, test = out / f"{name}_TRAIN.ts", out / f"{name}_TEST.ts"
        if train.exists() and test.exists():
            print(f"[skip] {name} (.ts present)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"[get ] {name}.zip ...", flush=True)
        try:
            z = zipfile.ZipFile(io.BytesIO(_get(UEA_URL_TMPL.format(name))))
        except Exception as e:                      # noqa: BLE001 - report and keep going
            print(f"[FAIL] {name}: {e}")
            continue
        # keep only the standard (non-_eq_) .ts pair — UEAloader globs by a TRAIN/TEST name filter
        # and loads the FIRST match, so extra variants (e.g. *_eq_*) must not be present.
        for n in z.namelist():
            base = os.path.basename(n)
            if base in (f"{name}_TRAIN.ts", f"{name}_TEST.ts"):
                (out / base).write_bytes(z.read(n))
        z.close()
        got = train.exists() and test.exists()
        print(f"[{'ok  ' if got else 'FAIL'}] {name} -> {out}")


SMD_API = "https://api.github.com/repos/NetManAIOps/OmniAnomaly/contents/ServerMachineDataset/{}"
SMD_RAW = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/{}/{}"


def fetch_smd() -> None:
    """Build SMD_{train,test,test_label}.npy by concatenating the 28 machine files.

    Upstream SMD ships as per-machine CSV text (OmniAnomaly). data_loader.SMDSegLoader expects the
    single concatenated .npy arrays used by TimesNet/TSLib, so we assemble them here (machines in
    sorted order, which is the convention those preprocessed bundles follow).
    """
    import json
    import numpy as np

    out = DATASET_ROOT / "SMD"
    targets = {k: out / f"SMD_{k}.npy" for k in ("train", "test", "test_label")}
    if all(p.exists() for p in targets.values()):
        print("[skip] SMD (.npy present)")
        return
    out.mkdir(parents=True, exist_ok=True)

    listing = json.loads(_get(SMD_API.format("train")).decode())
    machines = sorted(e["name"] for e in listing if e["name"].endswith(".txt"))
    print(f"[get ] SMD: {len(machines)} machines ...", flush=True)

    parts = {"train": [], "test": [], "test_label": []}
    for i, m in enumerate(machines, 1):
        for split in ("train", "test", "test_label"):
            raw = _get(SMD_RAW.format(split, m)).decode()
            arr = np.array([[float(x) for x in line.split(",")]
                            for line in raw.strip().splitlines()], dtype=np.float32)
            parts[split].append(arr)
        print(f"        [{i}/{len(machines)}] {m}", flush=True)

    for split, p in targets.items():
        data = np.concatenate(parts[split], axis=0)
        np.save(p, data)
        print(f"[ok  ] SMD_{split}.npy {data.shape}")


# NASA's telemanom release (Hundman et al., 2018): per-channel train/<chan>.npy, test/<chan>.npy and
# labeled_anomalies.csv. The original S3 link now returns 403; a mirror is the Kaggle dataset
# patrickfleith/nasa-anomaly-detection-dataset-smap-msl. Download it manually (Kaggle needs a login)
# and leave the zip, or its extracted folder, at one of these paths.
NASA_SOURCES = [DATASET_ROOT / "_raw" / "nasa_smap_msl.zip", DATASET_ROOT / "_raw" / "nasa_smap_msl",
                DATASET_ROOT / "archive.zip"]
# Rebuilt sizes must equal the published ones (train steps, test steps, dims), or nothing is written.
NASA_PUBLISHED = {"MSL": (58317, 73729, 55), "SMAP": (135183, 427617, 25)}


def build_msl_smap() -> None:
    """Build {MSL,SMAP}_{train,test,test_label}.npy from NASA's per-channel release.

    Same procedure as the OmniAnomaly preprocessing behind the TimesNet/TSLib bundle: channels in
    sorted order, filtered by spacecraft, SMAP's P-2 left out, train and test arrays concatenated
    along time, and the labelled anomaly ranges (inclusive) set to 1. Verified by the sizes above.
    Arrays are read with allow_pickle=False; nothing from the archive is executed.
    """
    import ast
    import csv
    import numpy as np

    targets = {sc: {k: DATASET_ROOT / sc / f"{sc}_{k}.npy" for k in ("train", "test", "test_label")}
               for sc in NASA_PUBLISHED}
    if all(p.exists() for t in targets.values() for p in t.values()):
        print("[skip] MSL, SMAP (.npy present)")
        return
    src = next((p for p in NASA_SOURCES if p.exists()), None)
    if src is None:
        print("[miss] MSL, SMAP: NASA release not found at " + ", ".join(str(p) for p in NASA_SOURCES))
        return

    if src.is_dir():
        def read(rel):
            return (src / rel).read_bytes()
    else:
        archive = zipfile.ZipFile(src)
        by_suffix = {}
        for name in archive.namelist():
            parts = name.split("/")
            if parts[-1] == "labeled_anomalies.csv":
                by_suffix["labeled_anomalies.csv"] = name
            elif len(parts) >= 2 and parts[-2] in ("train", "test") and parts[-1].endswith(".npy"):
                by_suffix[parts[-2] + "/" + parts[-1]] = name

        def read(rel):
            return archive.read(by_suffix[rel])

    def load(rel):
        return np.load(io.BytesIO(read(rel)), allow_pickle=False)

    table = list(csv.reader(io.StringIO(read("labeled_anomalies.csv").decode())))[1:]
    rows = sorted(table, key=lambda r: r[0])
    for sc, (n_train, n_test, dims) in NASA_PUBLISHED.items():
        info = [r for r in rows if r[1] == sc and r[0] != "P-2"]
        train = np.concatenate([load(f"train/{r[0]}.npy") for r in info], axis=0)
        test = np.concatenate([load(f"test/{r[0]}.npy") for r in info], axis=0)
        labels = []
        for r in info:
            lab = np.zeros(int(r[4]), dtype=np.float32)
            for a, b in ast.literal_eval(r[2]):
                lab[a:b + 1] = 1.0
            labels.append(lab)
        labels = np.concatenate(labels)
        got = (train.shape[0], test.shape[0], train.shape[1])
        if got != (n_train, n_test, dims) or labels.shape[0] != n_test:
            raise SystemExit(f"[fail] {sc}: rebuilt sizes {got} differ from published "
                             f"{(n_train, n_test, dims)}; nothing written")
        out = DATASET_ROOT / sc
        out.mkdir(parents=True, exist_ok=True)
        for split, arr in (("train", train), ("test", test), ("test_label", labels)):
            np.save(targets[sc][split], arr)
        print(f"[ok  ] {sc}: train {train.shape}, test {test.shape}, "
              f"anomalies {100 * labels.mean():.2f}% of test")


if __name__ == "__main__":
    print(f"dataset root: {DATASET_ROOT}")
    fetch_psm()
    fetch_uea()
    fetch_smd()
    build_msl_smap()
    print("\nDone. SWaT still needs manual sourcing (gated) — see this file's header.")
