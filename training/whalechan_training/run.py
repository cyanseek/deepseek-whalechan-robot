import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("WANDB_MODE", "disabled")
import argparse, json, time, hashlib, shutil, signal
from pathlib import Path
from dataclasses import asdict
import numpy as np, torch, mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner

if __package__:
    from .task import make_cfg, rl_cfg, C, NAMES
    from .data import RecordedEnv
else:
    from task import make_cfg, rl_cfg, C, NAMES
    from data import RecordedEnv


class SavedStop(Exception):
    pass


class RetainedRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        infos = {
            **(infos or {}),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
            },
        }
        super().save(path, infos)
        if (
            Path(self.logger.log_dir, "STOP").exists()
            and Path(path).name != "checkpoint-final.pt"
        ):
            raise SavedStop()

    def load(self, *args, **kwargs):
        infos = super().load(*args, **kwargs)
        if infos and "rng" in infos:
            rng = infos["rng"]
            torch.set_rng_state(rng["torch"].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in rng["cuda"]])
            np.random.set_state(rng["numpy"])
        self.current_learning_iteration += 1
        return infos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--resume")
    p.add_argument(
        "--recipe",
        choices=("baseline", "fullsole"),
        default="baseline",
    )
    p.add_argument("--consolidate", action="store_true")
    p.add_argument("--stand", action="store_true")
    a = p.parse_args()
    if a.envs < 1 or a.iterations < 1:
        p.error("--envs and --iterations must be positive")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    signal.signal(
        signal.SIGUSR1, lambda *_: (out / "STOP").write_text("save and stop requested")
    )
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = make_cfg(a.envs, play=a.stand, recipe=a.recipe)
    (out / "config.txt").write_text(repr(cfg))
    shutil.copytree(
        Path(__file__).parent,
        out / "source",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    started = time.perf_counter()
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    m = env.sim.mj_model
    (out / "model.xml").write_text(env.scene.spec.to_xml())
    mujoco.mj_saveModel(m, str(out / "model.mjb"))
    action = env.action_manager.get_term("joint_pos")
    print("ACTION_NAMES", action._target_names, flush=True)
    contract = {
        **C,
        "actuator_names": [m.actuator(i).name for i in range(m.nu)],
        "joint_names": [m.joint(int(j)).name for j in m.actuator_trnid[:, 0]],
        "qpos0": env.sim.data.qpos[0].cpu().tolist(),
        "qvel0": env.sim.data.qvel[0].cpu().tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "action_scale": action._scale[0].cpu().tolist(),
        "action_offset": action._offset[0].cpu().tolist(),
        "action_joint_names": action._target_names,
        "observation_order": list(cfg.observations["actor"].terms),
        "physics_dt": env.physics_dt,
        "policy_dt": env.step_dt,
        "model_sha256": hashlib.sha256((out / "model.xml").read_bytes()).hexdigest(),
        "policy_kind": "RSL-RL PPO; own body; no teacher or QP fallback",
        "mass_kg": float(m.body_mass.sum()),
        "seed": 42,
    }
    (out / "contract.json").write_text(json.dumps(contract, indent=2))
    report = {
        "envs": a.envs,
        "iterations": a.iterations,
        "recipe": a.recipe,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "init_wall_s": time.perf_counter() - started,
    }
    if a.stand:
        tilt = []
        resets = 0
        trajectory = []
        for step in range(250):
            obs, reward, terminated, truncated, info = env.step(
                torch.zeros(a.envs, len(NAMES), device="cuda")
            )
            resets += int(terminated.sum())
            tilt.append(
                float(
                    torch.acos(
                        (-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)
                    ).max()
                    * 180
                    / np.pi
                )
            )
            trajectory.append(env.sim.data.qpos[0].cpu().numpy().copy())
        report.update(terminated=resets, max_tilt=max(tilt), sim_seconds=5.0)
        np.savez_compressed(out / "stand.npz", qpos=trajectory)
    else:
        rcfg = asdict(rl_cfg())
        rcfg["save_interval"] = 50
        if a.consolidate:
            rcfg["algorithm"].update(
                entropy_coef=0.0, learning_rate=1e-5, schedule="fixed"
            )
        (out / "agent-config.json").write_text(json.dumps(rcfg, indent=2))
        wrapped = RecordedEnv(env, out)
        runner = RetainedRunner(wrapped, rcfg, str(out), device="cuda:0")
        if a.resume:
            runner.load(a.resume)
        if a.consolidate:
            with torch.no_grad():
                runner.alg.actor.distribution.std_param.clamp_(max=0.2)
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = 1e-5
            runner.alg.learning_rate = 1e-5
            report["consolidation"] = (
                "entropy 0; initial exploration std capped at 0.2; fixed learning rate 1e-5; unchanged reward recipe"
            )
        start_iter = runner.current_learning_iteration
        before = time.perf_counter()
        try:
            runner.learn(
                num_learning_iterations=a.iterations, init_at_random_ep_len=True
            )
        except SavedStop:
            report["stopped_after_save"] = True
        report["training_wall_s"] = time.perf_counter() - before
        runner.save(str(out / "checkpoint-final.pt"))
        runner.export_policy_to_onnx(str(out), "policy.onnx")
        wrapped.flush()
        report["completed_iterations"] = (
            runner.current_learning_iteration - start_iter + 1
        )
        report["transitions"] = a.envs * report["completed_iterations"] * 24
        report["resume_from"] = a.resume
        report["resume_scope"] = (
            "weights, normalizers, optimizer, iteration, curriculum counter and RNG; episodes restart"
        )
    report["total_wall_s"] = time.perf_counter() - started
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("REPORT", report, flush=True)
    env.close()


if __name__ == "__main__":
    main()
