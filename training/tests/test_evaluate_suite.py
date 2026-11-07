import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from whalechan_training import evaluate_suite as suite


def fake_evaluator(monkeypatch, outcomes):
    calls = []

    def run(command, *, stdout, stderr):
        calls.append(command)
        out = Path(command[command.index("--out") + 1])
        # The wrapper must leave directory creation to the native evaluator.
        out.mkdir(exist_ok=False)
        status = outcomes[len(calls) - 1]
        stdout.write(f"case {len(calls)}: {status}\n")
        if status == "missing":
            return SimpleNamespace(returncode=2)
        if status == "malformed":
            (out / "report.json").write_text("{", encoding="utf-8")
        else:
            report = {"qualified": status is True or status == "nonzero", "sim_seconds": 20.,
                      "confirmed_steps_by_foot": [4, 5], "walking_slip_m": [.01, .02]}
            if status == "nonboolean":
                report["qualified"] = "false"
            (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (out / "trajectory.json").write_text("[]", encoding="utf-8")
        return SimpleNamespace(returncode=9 if status == "nonzero" else 0)

    monkeypatch.setattr(suite.subprocess, "run", run)
    return calls


def test_suite_runs_all_three_scenarios_and_keeps_unqualified_results(tmp_path, monkeypatch):
    calls = fake_evaluator(monkeypatch, [True, False, True])
    policy = tmp_path / "policy with spaces"
    out = tmp_path / "suite"
    result = suite.run_suite(policy, out)
    assert not result["qualified"]
    assert [c["status"] for c in result["cases"]] == ["qualified", "unqualified", "qualified"]
    assert len(calls) == 3
    for command, name in zip(calls, ("default", "faster", "immediate")):
        assert command[:4] == [suite.sys.executable, "-m", "whalechan_training.evaluate", str(policy)]
        assert command[4:8] == ["--out", str(out / name), "--seconds", "20"]
        assert (out / name / "trajectory.json").exists()
        assert (out / f"{name}.log").read_text().startswith("case")
    assert calls[0][8:] == []
    assert calls[1][8:] == ["--speed", "0.065"]
    assert calls[2][8:] == ["--start-immediately"]
    assert result["cases"][0]["metrics"]["confirmed_steps_by_foot"] == [4, 5]
    assert result == json.loads((out / "summary.json").read_text())


@pytest.mark.parametrize("failure", ["missing", "malformed", "nonzero", "nonboolean"])
def test_failed_case_does_not_skip_later_cases(tmp_path, monkeypatch, failure):
    calls = fake_evaluator(monkeypatch, [failure, True, True])
    result = suite.run_suite(tmp_path / "policy", tmp_path / "suite")
    assert len(calls) == 3 and not result["qualified"]
    assert result["cases"][0]["status"] == "error"
    assert result["cases"][0]["error"]
    assert result["cases"][1]["status"] == result["cases"][2]["status"] == "qualified"


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    calls = fake_evaluator(monkeypatch, [True] * 3)
    saved = tmp_path / "summary.json"
    saved.write_text("prior results", encoding="utf-8")
    with pytest.raises(FileExistsError):
        suite.run_suite(tmp_path / "policy", tmp_path)
    assert not calls and saved.read_text() == "prior results"


@pytest.mark.parametrize("outcomes,code", [([True] * 3, 0), ([False, True, True], 1),
                                           (["missing", True, True], 2)])
def test_cli_exit_distinguishes_motion_failure_from_execution_error(tmp_path, monkeypatch, outcomes, code):
    fake_evaluator(monkeypatch, outcomes)
    monkeypatch.setattr(suite.sys, "argv", ["whalechan-evaluate-suite", str(tmp_path / "policy"),
                                          "--out", str(tmp_path / "suite")])
    with pytest.raises(SystemExit) as result:
        suite.main()
    assert result.value.code == code
