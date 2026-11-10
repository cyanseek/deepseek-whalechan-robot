# Star Catcher · 星海收集站 🐋

[马上玩 / Play now](https://cyanseek.github.io/deepseek-whalechan-robot/play/) · [下载单文件 / Download HTML](https://github.com/cyanseek/deepseek-whalechan-robot/raw/refs/heads/main/docs/play/index.html)

陪大肥鱼接住 6 颗星星、躲开水母。三条泳道、3 颗爱心、35 秒；胜利、失败、暂停和重玩都可以直接体验。

Catch 6 stars, avoid jellyfish: three lanes, three lives, 35 seconds. Win, lose, pause and try again. The page has Chinese game text and bilingual controls/instructions.

## 运行 / Run

在线打开上方链接，或下载 `index.html` 后双击。克隆仓库后也可以直接打开 `docs/play/index.html`。现代浏览器即可；无需服务器、Python、模型文件、API 密钥或联网依赖。

Open the link above, or download `index.html` and open it in a modern browser. After cloning, open `docs/play/index.html` directly. The single file contains the level, engine and Canvas artwork; no server, Python, model, key or runtime network access is needed.

| 操作 / Input | 行为 / Action |
| --- | --- |
| ← / →, A / D | 左右移动 / Move left or right |
| 点击泳道或按钮 / Tap a lane or button | 直接选择泳道 / Select a lane |
| 空格 / Space | 暂停、继续 / Pause or resume |
| 开始 / Play | 开始或重新开始 / Start or restart |
| 切换浏览器标签 / Switch browser tab | 自动暂停，手动继续 / Pause automatically; resume manually |

## 与具身项目的关系 / How it fits

这是项目自行编写的屏幕互动原型，蓝鲸图案由 Canvas 几何代码绘制。它让屏幕玩法能在接入机器人前独立试用和修改。此公开示例只运行游戏状态和画面，没有调用 DeepSeek、仿真控制器或真实硬件。身体反馈需要通过任务与完成反馈接口另行接入；[现有动作接口](../BEHAVIORS.md)与[学习步态流程](../../training/README.md)分别维护。

This authored screen-game prototype uses a whale drawn with Canvas geometry. It makes the screen interaction independently playable and editable before body integration. This public example runs game state and visuals only, without DeepSeek calls, a simulation controller or hardware I/O. Body feedback needs a separate task/completion adapter; see the existing [behavior interface](../BEHAVIORS.md) and [learned-policy workflow](../../training/README.md).

## 修改与验证 / Edit and verify

Use Node.js 22 or newer; no packages to install. Run from the repository root:

```sh
node docs/play/build.cjs
node docs/play/build.cjs --check
node --test docs/play/game.test.cjs
```

| 文件 / File | 用途 / Purpose |
| --- | --- |
| `engine.js` | 固定 20ms 步长、碰撞、计分与结果 / Fixed-step game state and outcomes |
| `level.json` | 默认关卡：出现时间、泳道、星星与水母 / Default spawn schedule |
| `view.js` | Canvas 绘制与键盘/触摸事件 / Drawing and input events |
| `app.js` | 浏览器时钟与输入连接 / Browser clock and input wiring |
| `template.html` | 布局、说明和控件 / Layout, instructions and controls |
| `index.html` | 可复现构建的单文件成品 / Reproducible standalone build |

修改源码后重新构建并提交 `index.html`。测试覆盖完整胜负路线、超时、暂停与恢复、重玩、不同刷新步长一致性、页面输入连接及构建一致性。CI 不需要角色资产或 GPU。

Rebuild and commit `index.html` after source edits. Tests cover complete win/loss routes, timeout, pause/resume, restart, clock partitioning, bundled input wiring and build reproducibility. CI needs no character assets or GPU. This example uses the repository's MIT code license.
