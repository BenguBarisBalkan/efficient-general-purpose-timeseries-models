"""
Fetch the task datasets that aren't shipped with this repo, into ../../dataset/.

Downloads the two FULLY-OPEN datasets used to validate the new SparseTSF tasks:
  • PSM             (anomaly detection) -> dataset/PSM/{train,test,test_label}.csv
  • JapaneseVowels  (classification)    -> dataset/JapaneseVowels/JapaneseVowels_{TRAIN,TEST}.ts

Idempotent: files already present are skipped. Stdlib only (urllib + zipfile).

    venv/Scripts/python.exe benchmarks/tools/fetch_task_datasets.py

── Datasets this script does NOT fetch (documented so the loaders can be satisfied manually) ──
The other anomaly sets need the TimesNet/TSLib *preprocessed* .npy bundle, which is only
distributed via Google Drive / Tsinghua Cloud (not reliably scriptable). Drop the files with these
exact names into the matching folder and they work with data_provider/data_loader.py as-is:

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


if __name__ == "__main__":
    print(f"dataset root: {DATASET_ROOT}")
    fetch_psm()
    fetch_uea()
    fetch_smd()
    print("\nDone. SMAP/MSL/SWaT still need manual sourcing — see this file's header.")
