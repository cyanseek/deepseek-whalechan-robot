# Credits and licenses

Project-authored code is licensed under [MIT](LICENSE). The current release contains the VD24
dynamic character runtime, task control, recording tools and related tests.

## Character

Base model and artwork: **木子恶魔** — [original work](https://www.aplaybox.com/details/model/ayXkRzVflmnd).
The original file was obtained through [dsh-plugin-vrmpet3d](https://github.com/drliuhuan/dsh-plugin-vrmpet3d/tree/13cb2a2532ce34fa7dfd6ff2715da4f50565b161).
Meshes, textures, skeleton and original expression morphs retain their original authorship.
The runtime package preserves the source metadata and provenance notices. Its terms are separate
from code MIT; see [ASSETS.md](docs/ASSETS.md) and the package's credits/THIRD_PARTY_NOTICES.md.

Our work includes the dynamic body, motion and task control, tail/arm coordination, runtime clothing,
expression blending and visual adaptation. Character and generated-image references, including
GPT-assisted additions and detail refinement, are credited in the [README](README.md#来源与许可).

## Engineering references

[MicroDuck](https://github.com/pollen-robotics/microduck),
[MicroDuck RL](https://github.com/pollen-robotics/microduck_rl),
[Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) and
[Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) inform the engineering roadmap.
This release does not bundle their code, robot models or learned policies.

This is an independent community project with no official affiliation with DeepSeek.
