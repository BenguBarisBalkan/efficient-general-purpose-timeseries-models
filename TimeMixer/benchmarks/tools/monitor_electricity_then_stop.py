import subprocess
import time
from pathlib import Path


# This script lives in TimeMixer/benchmarks/, so REPO_ROOT is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "logs" / "run_logs" / "Electricity_pred96.log"
RUNNER_PID = 26148


def has_final_metrics() -> bool:
    if not LOG_PATH.exists():
        return False
    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    return "mse:" in text and "mae:" in text


def main() -> None:
    while True:
        if has_final_metrics():
            subprocess.run(
                ["python", str(REPO_ROOT / "benchmarks" / "run_all_long_term_benchmarks.py"), "--update-from-logs-only"],
                cwd=str(REPO_ROOT),
                check=False,
            )
            subprocess.run(["taskkill", "/PID", str(RUNNER_PID), "/T", "/F"], check=False)
            break
        time.sleep(60)


if __name__ == "__main__":
    main()

