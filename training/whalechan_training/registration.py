"""Optional mjlab task discovery; the character runtime has no training imports."""

from mjlab.tasks.registry import register_mjlab_task
from .task import make_cfg, rl_cfg
from .run import RetainedRunner

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-WhaleChan-VD39",
    env_cfg=make_cfg(64),
    play_env_cfg=make_cfg(1, play=True),
    rl_cfg=rl_cfg(),
    runner_cls=RetainedRunner,
)
