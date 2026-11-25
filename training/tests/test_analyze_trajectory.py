import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from whalechan_training.analyze_trajectory import analyze_trajectory


def row(t=2.02, command=.05):
    return dict(t=t, command=[command, 0., 0.], qpos=[.1, .2, .3, 1., 0., 0., 0.],
                speed_m_s=.03, horizontal_speed_m_s=.03, angular_speed_rad_s=.1,
                normal_force=[8., 2.], obs=[0.] * 124)


def save(tmp_path, rows):
    path = tmp_path / "鲸鱼 trajectory.json"
    path.write_text(json.dumps(rows), encoding="utf-8-sig")
    return path


def test_saved_trajectory_records_provenance_and_keeps_original_bytes(tmp_path):
    path = save(tmp_path, [row()])
    before = path.read_bytes()
    report = analyze_trajectory(path)
    assert report["analysis"] == "offline_recorded_trajectory"
    assert report["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert report["recorded_samples"] == 1 and report["initial_stand_s"] == 2
    assert report["walking_speed_windows"][0]["mae_m_s"] == pytest.approx(.02)
    assert report["support_windows"][0]["left_normal_force_share"] == .8
    # Older records without angles still produce the other diagnostics.
    assert report["posture_windows"][0]["invalid_samples"] == 1
    assert report["posture_windows"][0]["sampled_max_tilt_deg"] is None
    assert "qualified" not in report and "neural_policy" not in report
    assert path.read_bytes() == before


def test_immediate_schedule_and_empty_recording(tmp_path):
    path = save(tmp_path, [row(.02)])
    assert analyze_trajectory(path)["walking_speed_windows"][0]["samples"] == 0
    report = analyze_trajectory(path, start_immediately=True)
    assert report["initial_stand_s"] == 0
    assert report["walking_speed_windows"][0]["samples"] == 1
    assert report["support_windows"][0]["samples"] == 1
    path.write_text("[]", encoding="utf-8")
    empty = analyze_trajectory(path)
    assert empty["recorded_samples"] == 0
    assert all(not w["complete"] and w["net_drift_m"] is None for w in empty["stop_windows"])


def test_posture_fields_survive_offline_loading_and_cli_without_runtime(tmp_path):
    sample = row(.02)
    sample.update(tilt_rad=0., heading_error_rad=0.)
    path = save(tmp_path, [sample]); out = tmp_path / "posture.json"
    result = invoke(path, out, "--start-immediately")
    assert result.returncode == 0, result.stderr
    posture = json.loads(out.read_text())["posture_windows"][0]
    assert posture["samples"] == 1 and posture["invalid_samples"] == 0
    assert posture["sampled_max_tilt_deg"] == 0.
    assert posture["final_heading_error_deg"] == 0.


def test_partial_stop_and_invalid_metric_samples_remain_visible(tmp_path):
    walking = row()
    walking["speed_m_s"] = float("nan")
    walking["normal_force"] = [-1., 2.]
    boundary = row(8.)
    stop = row(8.02, 0.)
    stop["qpos"][0] += .03
    report = analyze_trajectory(save(tmp_path, [walking, boundary, stop]))
    assert report["walking_speed_windows"][0]["invalid_samples"] == 1
    assert report["walking_speed_windows"][0]["mae_m_s"] is None
    assert report["support_windows"][0]["invalid_samples"] == 1
    assert report["support_windows"][0]["mean_normal_force_n"] is None
    assert report["stop_windows"][0]["net_drift_m"] == pytest.approx(.03)
    assert not report["stop_windows"][0]["complete"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field,value", [("t", float("nan")), ("command", [0.]),
                                        ("qpos", [0.]), ("horizontal_speed_m_s", None)])
def test_invalid_core_fields_report_row_number(tmp_path, field, value):
    bad = row(); bad[field] = value
    with pytest.raises(ValueError, match="trajectory row 1"):
        analyze_trajectory(save(tmp_path, [row(), bad]))


def test_wrong_top_level_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="JSON array"):
        analyze_trajectory(save(tmp_path, {"rows": []}))


def invoke(path, out, *extra):
    # This subprocess cannot import a physics engine, policy runtime or GPU stack.
    script = """
import sys
class NoRuntime:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'mujoco', 'onnxruntime', 'torch', 'mjlab', 'warp'}:
            raise AssertionError('Offline analysis imported ' + fullname)
sys.meta_path.insert(0, NoRuntime())
from whalechan_training.analyze_trajectory import main
main()
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONUTF8="1")
    return subprocess.run([sys.executable, "-c", script, str(path), "--out", str(out), *extra],
                          capture_output=True, text=True, encoding="utf-8", env=env)


def test_cli_writes_offline_report_and_preserves_existing_output(tmp_path):
    path = save(tmp_path, [row(.02)])
    out = tmp_path / "报告" / "analysis.json"
    result = invoke(path, out, "--start-immediately")
    assert result.returncode == 0, result.stderr
    before = out.read_bytes()
    assert json.loads(before)["walking_speed_windows"][0]["samples"] == 1
    assert invoke(path, out).returncode == 2
    assert out.read_bytes() == before
    original = path.read_bytes()
    assert invoke(path, path).returncode == 2
    assert path.read_bytes() == original


@pytest.mark.parametrize("content", ["{", '[{"t": 1}]'])
def test_cli_bad_input_keeps_source_and_leaves_no_report(tmp_path, content):
    path = tmp_path / "broken.json"; path.write_text(content, encoding="utf-8")
    out = tmp_path / "analysis.json"
    result = invoke(path, out)
    assert result.returncode == 2 and "Analysis failed:" in result.stderr
    assert not out.exists() and path.read_text(encoding="utf-8") == content
