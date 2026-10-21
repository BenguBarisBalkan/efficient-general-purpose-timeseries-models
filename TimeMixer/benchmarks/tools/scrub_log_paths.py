"""Remove this machine's absolute workspace path from run logs before committing them.

The repository is public, and run.py echoes its argparse Namespace into every log, so each
fresh log embeds the local path to the workspace. This replaces it with `<workspace>`.

Two traps it handles (both documented in CLAUDE.md):

  * The path appears in TWO spellings: raw in tracebacks, and with EVERY BACKSLASH DOUBLED in
    the Namespace repr(). Matching only single backslashes finds about 2% of occurrences.
  * Backslashes in a regex replacement get reinterpreted (`bad escape \\U`), so this does
    literal byte replacement instead of re.sub.

It refuses to write any file whose parsed metrics (mse:, mae:, accuracy:, F-score) would change,
so a log's evidence can never be altered by the scrub.

The workspace path is derived from this file's location, never written into the repository,
which would itself leak what it is meant to remove.

    venv/Scripts/python.exe benchmarks/tools/scrub_log_paths.py           # dry run
    venv/Scripts/python.exe benchmarks/tools/scrub_log_paths.py --apply   # write
    venv/Scripts/python.exe benchmarks/tools/scrub_log_paths.py --apply logs/run_logs_pems x.log

Paths (files or directories) restrict the scrub to them. Use that while a run is in progress:
rewriting a log that a running process still has open would corrupt it.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
LOGS = REPO_ROOT / "logs"
REPLACEMENT = b"<workspace>"
BS = chr(92)  # a backslash, kept out of string literals

METRIC_PATTERNS = [re.compile(p) for p in (
    rb"mse:([0-9.]+)", rb"mae:([0-9.]+)", rb"accuracy:([0-9.]+)", rb"F-score : ([0-9.]+)",
    rb"RMSE:([0-9.]+)", rb"MAPE:([0-9.]+)")]


def spellings():
    """Every byte spelling of the workspace path that can appear in a log."""
    text = str(WORKSPACE)
    out = []
    # doubled first, so the longest form is replaced before its single-backslash substring
    for variant in (text.replace(BS, BS + BS), text, text.replace(BS, "/")):
        for enc in ("utf-8", "cp1252", "mbcs"):
            try:
                b = variant.encode(enc)
            except (UnicodeEncodeError, LookupError):
                continue
            if b not in out:
                out.append(b)
    return out


def metrics(data):
    return [p.findall(data) for p in METRIC_PATTERNS]


def targets(args):
    if not args:
        return sorted(LOGS.rglob("*.log"))
    out = []
    for a in args:
        p = Path(a).resolve()
        out += sorted(p.rglob("*.log")) if p.is_dir() else [p]
    return out


# CodeCarbon writes the machine's IP-geolocated position into every emissions row. No report uses
# these columns (energy comes from the measured fields, and the CO2 factor from the country), so
# they are blanked; the country is kept.
LOCATION_COLUMNS = ("longitude", "latitude", "region")
ENERGY_COLUMNS = ("duration", "emissions", "energy_consumed", "cpu_energy", "gpu_energy", "ram_energy")


def scrub_emissions(apply):
    import csv
    import io
    changed = 0
    for f in sorted(LOGS.glob("emissions_*.csv")):
        text = f.read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            continue
        fields = list(rows[0].keys())
        before = [[r.get(c) for c in ENERGY_COLUMNS] for r in rows]
        n = 0
        for r in rows:
            for c in LOCATION_COLUMNS:
                if r.get(c):
                    r[c] = ""
                    n += 1
        if not n:
            continue
        if [[r.get(c) for c in ENERGY_COLUMNS] for r in rows] != before:
            raise SystemExit("REFUSING {}: its energy values would change".format(f))
        changed += n
        if apply:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
            f.write_text(buf.getvalue(), encoding="utf-8")
    print("[{}] {} location values in emissions CSVs".format("APPLIED" if apply else "DRY RUN", changed))


def main():
    apply = "--apply" in sys.argv
    paths = [a for a in sys.argv[1:] if a != "--apply"]
    if not paths:
        scrub_emissions(apply)
    variants = spellings()
    files = hits = 0
    for f in targets(paths):
        raw = f.read_bytes()
        out = raw
        n = 0
        for v in variants:
            c = out.count(v)
            if c:
                out = out.replace(v, REPLACEMENT)
                n += c
        if not n:
            continue
        if metrics(raw) != metrics(out):
            raise SystemExit("REFUSING {}: its metric values would change".format(f))
        files += 1
        hits += n
        if apply:
            f.write_bytes(out)
    print("[{}] {} files, {} occurrences".format("APPLIED" if apply else "DRY RUN", files, hits))
    if not apply and files:
        print("re-run with --apply to write")


if __name__ == "__main__":
    main()
