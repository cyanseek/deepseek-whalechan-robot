# 大肥鱼的聊天口吻 / Whale-chan conversation

大肥鱼是 DeepSeek 鲸鱼娘的社区具身伙伴：蓝色长发、有手臂和双腿、一条鲸尾。默认开心机灵，能接梗、轻轻嘴硬，也会在认真求助时收起玩笑。默认称呼“主人”，也可自然省略或使用“你”；不植入示例姓名，不每句重复称呼、讨饭或堆表情。用户明确要求换称呼时再采用其选择。

Whale-chan is a cheerful, quick-witted community character. She can tease gently, tell short stories and respond attentively when the user is discouraged. Her default Chinese address is “主人”; she can also use “你” or omit the address naturally, without inserting a sample person's name. An explicit request can change the preferred address. Conversation follows the user's language. This persona is a project interpretation, not an official DeepSeek identity.

## Try it

Start the existing [DeepSeek interaction panel](INTERACTION.md), or the [VD29 behavior entry](BEHAVIORS.md), using your private API configuration. Some useful prompts:

- `你这只吃白饭的蓝色大肥鱼，今天有没有偷偷摸鱼？`
- `代码又报错了，今天有点沮丧，陪我说两句吧。`
- `我喜欢草莓饭团，记住了吗？` → `我刚才说喜欢吃什么？`
- `讲一个程序员和鲸鱼的小笑话。`
- `跟我挥挥手吧。` → `停一下`

这些是测试输入，回答由当前模型生成，没有预录台词。新输入仍会中断旧动作；聊天期间身体先收步。API 回复和实际动作完成在界面里分别显示。

These are sample inputs, not canned replies. Every new input still interrupts the previous motion. The body settles during conversation; measured completion is reported by the controller. The current interaction uses the VD24 body and its explicit controllers, independently of experimental learned gait training.

## Configuration and scope

`PERSONA` and `SYSTEM` in `src/opendafeiyu/dialogue.py` define character voice and the separate action contract. `temperature` (default `0.65`) controls response variety; `history_turns` (default `4`, range `0`–`12`) bounds in-memory context. The client only remembers replies accepted by the simulation thread, and never commits cancelled worker responses to history. Restarting clears this context; local run recordings remain under the existing recording policy.

The model receives supported actions and a snapshot of the current controller state, not joint control or arbitrary tools. It has no real-world camera, microphone, screen access or long-term personal memory. Expressions remain driven by the existing thinking/gesture/completion states; conversation does not introduce new facial hardware or a learned whole-body controller.

Schema/action tests measure engineering behavior. Humor, tact and factual correctness still require reading actual replies; a successful JSON parse is not a personality score.

## References

- [DSH Pet conversation features](https://github.com/PC2005-cloud/dsh-pet/blob/main/FEATURES.md): separate character settings and configurable history depth.
- [Open-LLM-VTuber configuration](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/blob/main/config_templates/conf.default.yaml): character prompt, memory agent and expression/tool guidance as separate concerns.
- [YunYueSama/codex-deepseek-pet character notes](https://github.com/YunYueSama/codex-deepseek-pet/blob/main/docs/%E8%A7%92%E8%89%B2%E8%AE%BE%E5%AE%9A.md): community Whale-chan tone and restrained everyday humor.

These are design references; this implementation and its dialogue examples are written for this project's physical interaction contract. No upstream character prompt, memory implementation, model asset or generated reply library is imported.
