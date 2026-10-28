# DeepSeek Whale-chan Robot · DeepSeek クジラ娘ロボット 🐋

**DeepSeek クジラ娘を、現実の世界へ。**

AI コンパニオンと人・ロボットのインタラクションを目指すオープンソースプロジェクトです。DeepSeek コミュニティのクジラ娘（大肥鱼 / 鲸鱼娘 / Whale-chan）を題材に、MuJoCo で小刻みな二足歩行、旋回、尾の連動、表情を実装しています。会話やタスクに応じ、新しい行動を追加できる具身パートナーを目指し、実機ロボットへ進めていきます。

継続的に更新中 · 現在は MuJoCo シミュレーション段階です。

現在のソースには [DeepSeek テキスト対話](docs/INTERACTION.md)、[接触を考慮した歩行と停止](docs/CONTACT_WALK.md)、[頭部動作・行動の組み合わせ・一時停止と再開](docs/BEHAVIORS.md)があります。標準起動と以下の39秒デモは VD24 のままで、追加機能はソースから実行します。

[简体中文](README.md) · [English](README.en.md) · **日本語**

**学習用ワークフローを公開：** [本体に合わせた mjlab / PPO 学習とデータ処理](training/README.md)。オンライン収集、チェックポイント、ONNX 出力、独立した CPU 歩行・停止評価を含みます。学習歩行は改善中で、以下のデモは既存のコントローラーを使用しています。

[デモを見る](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html) · [クイックスタート](#クイックスタート) · [ロードマップ](#ロードマップ) · [Agent 向け](#agent-向け) · [開発ガイド](docs/CHARACTER.md)

[![DeepSeek Whale-chan Robot：MuJoCo 内で歩き、尾と腕を連動させる DeepSeek Whale-chan。物理シミュレーションの等速抜粋](docs/media/vd24/walk.gif)](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html)

**MuJoCo 物理シミュレーション · 3 軸の尾 · 動作と表情の連動 · VD2.4**

GIF は録画の 6〜12 秒を等速で繰り返します。クリックすると **39 秒 / 60fps** の全編を見られます。挨拶、歩行、静止してからの小さな跳躍、旋回、手と尾による応答、休息を収録しています。身体はアクチュエータと地面接触によって動き、上位の動作順序はデモ用に設定されています。

[▶ 39 秒 / 60fps のフルデモを見る](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html) · [動画をダウンロード](https://github.com/cyanseek/deepseek-whalechan-robot/releases/download/vd24-ar2/DeepSeek-Whalechan-Robot-VD24-demo.mp4)

## 現在の機能

| 手を振って挨拶 | 笑顔と表情の変化 |
| :---: | :---: |
| ![実際の物理シミュレーションで手を振る様子、等速](docs/media/vd24/wave.gif) | ![瞬き、笑顔、表情の切り替え](docs/media/vd24/expressions.gif) |
| **3 軸のクジラの尾** | **小刻みな歩行と動作の連動** |
| ![後方から見た尾の動き、実際のシミュレーション](docs/media/vd24/tail.gif) | ![腕と尾を連動させて歩く様子、等速](docs/media/vd24/walk-detail.gif) |

すべて実際のシミュレーションを等速で抜粋しています。挨拶はフルデモ、表情と尾は個別のシナリオから収録しました。[映像の出典と再実行コマンド](docs/DEMO.md)

- **動作と身体表現**：小刻みな歩行、旋回、手振り、実験的な小跳躍。腕と 3 軸の尾を動作に合わせて連動させます。
- **見た目と表情**：目を開けた笑顔、自然な瞬き、滑らかな表情の遷移、衣装と表示の調整。
- **知覚とインタラクション**：ロボットの RGB カメラで色付きマーカーを検出して接近。キャンセル時は支持状態に応じて歩行を終了します。
- **シミュレーションと再現**：23 アクチュエータ、自由基底、地面接触、トルク制限。動画、軌跡、結果、ソースのスナップショットを保存します。

明示的な IK/PD 制御と理想的な自己状態フィードバックを使い、平地のシナリオで検証しています。跳躍はまだ低く、速い歩行と着地制御は改善中です。[デモと測定結果](docs/DEMO.md)

## クイックスタート

**まず見るだけならインストール不要：** [39 秒の全編動画](https://cyanseek.github.io/deepseek-whalechan-robot/watch.html)。

**ローカル実行：** 以下はプロジェクトのソースディレクトリで実行します。検証環境は Windows、Python **3.13.5**、MuJoCo **3.3.7**。物理と制御は CPU、描画は OpenGL を使用します。デモに API キーや言語モデルは不要です。現在はソースからの実行に対応しています。

### 1. 環境を準備する

プロジェクトのルートで PowerShell を開きます。

```powershell
git clone https://github.com/cyanseek/deepseek-whalechan-robot.git
cd deepseek-whalechan-robot
py -3.13 -m venv .venv-sim
./.venv-sim/Scripts/python.exe -m pip install -r requirements-character.txt
```

### 2. キャラクターアセットをインストール

対応する **AR2 ランタイムパッケージ**（約 29.5 MB）にはモデル、スキン、テクスチャ、表情のバインディングが含まれます。モデルの手動編集は不要です。

```powershell
./.venv-sim/Scripts/python.exe tools/install_character_assets.py
./.venv-sim/Scripts/python.exe tools/character_preflight.py --delivery assets/local/BlueFatFish_Visual_Delivery
```

インストーラーが [GitHub Release](https://github.com/cyanseek/deepseek-whalechan-robot/releases/tag/vd24-ar2) から取得・検証し、既存ファイルを保護しながら `assets/local/BlueFatFish_Visual_Delivery/` に展開します。[ZIP の手動取得](https://github.com/cyanseek/deepseek-whalechan-robot/releases/download/vd24-ar2/DeepSeek-Whalechan-Robot-Assets-AR2-VD24.zip)と `--archive` によるオフラインインストールも可能です。[アセットの説明](docs/ASSETS.md)

`ready_for_native_check` は依存関係のメタデータとアセット検査の通過を示します。続いて実際のシミュレーションを実行してください。

### 3. 動かしてみる

```powershell
# 操作できるビューアを開く
./Start-BlueFatFish.ps1 -Delivery assets/local/BlueFatFish_Visual_Delivery

# 全編デモを実行し、等速の 60fps 動画を保存する
./Start-BlueFatFish.ps1 -Delivery assets/local/BlueFatFish_Visual_Delivery -Scenario coordinated -Seconds 39 -Record -Presentation -Fps 60
```

| キー | 動作 | キー | 動作 |
| --- | --- | --- | --- |
| W / A | 前進 / 旋回 | J | 実験的な小跳躍 |
| T / G | 尾を振る / 手を振る | E | 表情を切り替える |
| R | 赤いマーカーに近づく | C / スペース | タスクをキャンセル |
| V / D | 視点 / 衝突形状の表示 | ウィンドウを閉じる | 終了 |

起動ごとに `runs/` 内へ独立したディレクトリを作ります。動画は `simulation.mp4`、結果は `run.json` に保存されます。軌跡などは[出力仕様](docs/AGENT_GUIDE.md#outputs)を参照してください。キャンセルは接触状態を考慮して動作を終了します。

### 割り込み可能な対話を試す

オフラインパネルで「过来」「停一下」「继续」「打个招呼」を入力できます。接近時はロボットカメラで赤いマーカーを検出し、動作の完了を確認して次へ進みます。

```powershell
./.venv-sim/Scripts/python.exe -m pip install -r requirements-contact.txt
./Start-WhaleChan-Behaviors.ps1 -Backend commands -Record

./.venv-sim/Scripts/python.exe tools/run_behavior_demo.py --out runs/behavior-demo-01 --video
```

自動テキストクライアントのデモは、歩行・一時停止・再開・手振り→尾振り→うなずきを約46秒のシミュレーションとして記録します。[DeepSeek API設定](docs/INTERACTION.md)と[行動インターフェース](docs/BEHAVIORS.md)を参照してください。

## ロードマップ

目指すのは、会話ができ、身体を動かし、個性を表現できるパートナーです。「こっちに来て」で近づいて挨拶し、途中で「待って」と言えば安定して止まり、応答する。目標を変えれば、新しいタスクに進む。そんな体験をつくります。

| 段階 | 内容 |
| --- | --- |
| **実装済み：身体と表現** | MuJoCo での小刻みな歩行、尾の連動、表情、色付きマーカーへの接近、キャンセル可能なタスク。対応アセットのインストーラーと実際の録画も用意しています。 |
| **ソースで利用可能：テキストと動作** | DeepSeek のテキスト意図、割り込み可能な接近と安定停止、頭部フィードバック、行動の組み合わせ、一時停止と再開。オフラインの明示的コマンドにも対応。 |
| **次の段階：運動と音声** | 速度追従、足滑りと外乱への対応、身体に適合する学習方策、音声入力。 |
| **さらに拡張：新しい行動を教える** | 挨拶、追従、日常の交流を再現可能な例にし、開発者や Agent が行動を追加・検証・共有できる安定したタスクインターフェースを整えます。 |
| **実機へ：シミュレーションから現実へ** | 機構、アクチュエータ、センサー、校正を進め、同じタスクロジックで実機を動かし、挨拶やデスクのそばでの交流、リマインダーを段階的に検証します。 |

DeepSeek インターフェースはテキストと上位の意図を、ローカル制御はバランス・動作・結果のフィードバックを扱います。ソースには再現可能な行動の組み合わせが含まれます。標準 VD24 は IK/PD、任意の接触歩行は制約付き逆動力学を使用します。学習歩行・音声入力・実機は今後の目標です。[開発段階と検証目標](docs/ROADMAP.md)

## Agent 向け

プロジェクトファイルにアクセスできるコーディング Agent に、次の指示を渡せます。

> README.ja.md、docs/AGENT_GUIDE.md、project-info.json を読んでください。アセットと依存関係を確認し、2 秒の stand 検査、続いて VD24 の 39 秒 coordinated シナリオを実行してください。毎回新しい出力ディレクトリを使い、run.json の結果と動画の場所を報告してください。アセットが不足している場合は不足ファイルを列挙してください。

[Agent ガイド](docs/AGENT_GUIDE.md)、[機械可読のプロジェクト情報](project-info.json)、[llms.txt 文書索引](llms.txt)を用意しています。既存の Python CLI を呼び出し、結果を解析できます。デモ自体に LLM は必要ありません。

## 開発とコントリビューション

実装は[身体と制御](docs/CHARACTER.md)、検証範囲は[測定結果](docs/DEMO.md)をご覧ください。

歩行・着地制御、DeepSeek 対話と音声、インタラクション設計、許諾の明確なモデルの適合、実機の機構・制御への貢献を歓迎します。具体的なシナリオに、入力、期待する動作、再実行コマンドを添えてください。[貢献ガイド](CONTRIBUTING.md)

開発の流れは **S1 設計・シミュレーション → S2 実機ロボット**。

別名：**DeepSeek 大肥鱼 / DeepSeek 鲸鱼娘 / Whale-chan / whale girl / DeepSeek-chan / OpenDafeiyu**。

## クレジットとライセンス

関連するクリエイターとオープンソースコミュニティに感謝します。

- **二次元のキャラクターと生成画像の参考**：[Neko3000 / DeepSeek Whale-chan](https://github.com/Neko3000/deepseek-whalechan) と [YunYueSama / codex-deepseek-pet](https://github.com/YunYueSama/codex-deepseek-pet)。Neko の[出典説明](https://github.com/Neko3000/deepseek-whalechan/blob/main/THIRD-PARTY-NOTICES.md)には、Bilibili のクリエイター [ZipZipPipe](https://space.bilibili.com/4168597) と [上善无形](https://space.bilibili.com/4456176) が記載されています。これらの参考をもとに、私たちも GPT を活用してキャラクター案の補足と細部の改善を行いました。
- **基礎モデルと原画**：**木子恶魔** の[大肥鱼作品](https://www.aplaybox.com/details/model/ayXkRzVflmnd)。原ファイルの取得元として [dsh-plugin-vrmpet3d](https://github.com/drliuhuan/dsh-plugin-vrmpet3d/tree/13cb2a2532ce34fa7dfd6ff2715da4f50565b161) が含まれます。
- **本プロジェクトの具身実装**：受領した姿勢・表示調整を基に、動力学的な身体、接触とトルク制限、運動制御、タスク処理、歩行と尾の連携、衣装・表情・描画時の改善を実装しています。元のモデルと表情モーフの著作者は保持します。
- **ロボット技術の参考**：[MicroDuck](https://github.com/pollen-robotics/microduck) / [MicroDuck RL](https://github.com/pollen-robotics/microduck_rl) を具身プロダクト、学習、デプロイの主要な参考にしています。[Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) / [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground)。

プロジェクト独自のコードは [MIT](LICENSE) です。第三者のモデル、美術、およびそれらを含むデモ素材には別の条件が適用され、コードのライセンスは再配布権を付与しません。[アセット説明](docs/ASSETS.md)と [NOTICE](NOTICE.md)を参照してください。DeepSeek と公式な関係を持たない独立コミュニティプロジェクトです。
