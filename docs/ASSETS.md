# Character assets / 角色资产

VD24 uses the version-matched **AR2 runtime package** (about 29.5 MB). It includes the adapted GLB, skins, textures, bone bindings and original source metadata. No Blender or manual rig editing is required.

## Automatic installation

```powershell
./.venv-sim/Scripts/python.exe tools/install_character_assets.py
./.venv-sim/Scripts/python.exe tools/character_preflight.py --delivery assets/local/BlueFatFish_Visual_Delivery
```

The installer downloads the pinned [Release asset](https://github.com/cyanseek/deepseek-whalechan-robot/releases/download/vd24-ar2/DeepSeek-Whalechan-Robot-Assets-AR2-VD24.zip), checks its size, SHA-256, manifest and individual files, then installs to `assets/local/BlueFatFish_Visual_Delivery/`. Existing files are preserved. The directory is ignored by Git.

安装器会自动下载匹配模型包并校验，无需额外转换或绑定。下载后即可按 README 启动；代码仓库与角色运行资产分别分发。

## Offline installation

Download the ZIP from [Releases](https://github.com/cyanseek/deepseek-whalechan-robot/releases/tag/vd24-ar2) and run:

```powershell
./.venv-sim/Scripts/python.exe tools/install_character_assets.py --archive DeepSeek-Whalechan-Robot-Assets-AR2-VD24.zip
```

Repeating installation verifies and reuses matching files. For an edited installation, choose another `--destination` and pass it to the launcher with `-Delivery`. `--url` accepts an HTTPS mirror of the same pinned ZIP; its hash must match [the descriptor](../assets/characters/bluefatfish-runtime.json).

## Package contents

```text
BlueFatFish_Visual_Delivery/
├── ASSET_BUNDLE.json
├── ASSET_PACKAGE.md
├── BlueFatFish_Visual.glb
├── DELIVERY_MANIFEST.json
├── mujoco_preview/
│   ├── manifest.json
│   ├── visual_mocap.xml
│   ├── skins/
│   └── textures/
├── source/model_document.json
└── credits/
    ├── THIRD_PARTY_NOTICES.md
    └── SOURCE_RECORD.json
```

The preview binding is an input to the dynamic-body builder. Run the project launcher for the current VD24 simulation. AR2 retains AR1's visual/runtime dependency bytes while excluding authoring files, duplicate VRM data and production records. Controllers and runtime clothing are in the code repository.

## Troubleshooting

| Result | Action |
| --- | --- |
| Download unavailable | Download the Release ZIP in your browser and use `--archive`. |
| Archive hash mismatch | Use the ZIP matching the versioned descriptor. |
| Existing installation differs | Preserve it and choose another destination. |
| Missing dependency | Install `requirements-character.txt`, including SciPy. |
| MuJoCo mismatch | Use MuJoCo 3.3.7 for VD24 reproduction. |
| OpenGL error | Check the graphics environment; preflight does not exercise rendering. |

`ready_for_native_check` confirms metadata and asset checks. Run the native stand scenario to check dynamics and rendering.

## Source and terms

Base model/art: 木子恶魔. The original file distribution channel is dsh-plugin-vrmpet3d. Full attribution is in the [README](../README.md#来源与许可) and [NOTICE](../NOTICE.md).
Original model metadata and public provenance are included in the asset ZIP. The source plugin's author-permission statement and embedded restrictions are distinct records; this project does not relicense the character as MIT. Further use follows the applicable author/source terms.
