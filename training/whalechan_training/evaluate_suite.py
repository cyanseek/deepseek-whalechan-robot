"""Run the three documented CPU policy evaluations and retain every outcome."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


CASES = (
    ("default", ()),
    ("faster", ("--speed", "0.065")),
    ("immediate", ("--start-immediately",)),
)
METRICS = (
    "sim_seconds", "failure", "policy_sha256", "confirmed_steps_by_foot",
    "alternating_transitions", "walking_slip_m", "stops", "stop_windows",
    "max_tilt_deg", "max_heading_error_deg", "forward_displacement_m",
)


def run_suite(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "policy_bundle": str(run),
        "qualified": False,
        "cases": [{"name": name, "status": "pending"} for name, _ in CASES],
    }

    def save():
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    save()
    for case, (name, options) in zip(summary["cases"], CASES):
        command = [sys.executable, "-m", "whalechan_training.evaluate", str(run),
                   "--out", str(out / name), "--seconds", "20", *options]
        case.update(command=command, log=f"{name}.log", report=f"{name}/report.json",
                    trajectory=f"{name}/trajectory.json", status="running")
        save()
        try:
            with (out / case["log"]).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            case["returncode"] = result.returncode
            report = json.loads((out / case["report"]).read_text(encoding="utf-8"))
            if not isinstance(report, dict) or type(report.get("qualified")) is not bool:
                raise ValueError("Evaluation report is missing a boolean qualified result")
            case["metrics"] = {key: report.get(key) for key in METRICS}
            case["status"] = "qualified" if report["qualified"] else "unqualified"
            if result.returncode:
                case.update(status="error", error=f"Evaluator exited {result.returncode}")
        except (OSError, ValueError) as error:
            case.update(status="error", error=str(error))
        save()
    summary["qualified"] = all(c["status"] == "qualified" for c in summary["cases"])
    save()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Exported policy bundle")
    parser.add_argument("--out", required=True, help="Fresh suite output directory")
    args = parser.parse_args()
    try:
        summary = run_suite(args.run, args.out)
    except FileExistsError:
        parser.error("--out already exists; choose a fresh directory to preserve results")
    print(json.dumps(summary, indent=2))
    if any(case["status"] == "error" for case in summary["cases"]):
        raise SystemExit(2)
    raise SystemExit(0 if summary["qualified"] else 1)


if __name__ == "__main__":
    main()
