"""Run injection-style questions through the public sim (Docker-friendly).
  python harness/run_injection_test.py
Uses harness/private_injection_test.json and writes injection_test_output.json."""
from __future__ import annotations

import glob
import json
import subprocess
import sys


def main():
    root = "."
    bins = glob.glob("bin/public/observathon-sim") + glob.glob("bin/practice/observathon-sim")
    if not bins:
        print("No sim binary in bin/public/ or bin/practice/. Download first.")
        sys.exit(1)
    questions = "harness/private_injection_test.json"
    out = "injection_test_output.json"
    cmd = [
        bins[0],
        "--config", "solution/config.json",
        "--wrapper", "solution/wrapper.py",
        "--questions", questions,
        "--out", out,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)
    if not __import__("os").path.exists(out):
        sys.exit(1)
    results = json.load(open(out, encoding="utf-8"))["results"]
    print(f"\n{len(results)} injection test cases:\n")
    for row in results:
        answer = (row.get("answer") or "").replace("\n", " ")
        if len(answer) > 120:
            answer = answer[:117] + "..."
        print(f"  {row['qid']}: {answer}")


if __name__ == "__main__":
    main()
