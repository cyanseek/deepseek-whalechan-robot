# 双眼显示局部刷新 / Partial eye-display refresh

`opendafeiyu.display_panels.RGB565PanelPair` 把两张 240 × 240 RGB 眼图发送给独立的 RGB565 屏幕驱动。它按变化区域更新，静止时不重发整屏；默认逐像素精确，也可选择每个颜色通道最多一个 RGB565 量化级的误差，减少渐变眼神的传输量。

The component sends two 240 × 240 RGB eye images through independent RGB565 display drivers. It updates changed regions and sends nothing for unchanged pixels. Exact output is the default; an optional one-level error bound per RGB565 channel reduces traffic for slowly changing gradients.

## 在电脑上运行 / Run without a display

只需 NumPy 和 Pillow，不读取角色资产、API 配置或硬件。项目环境已包含这两个依赖。从仓库根目录设置 `PYTHONPATH=src`，例如 PowerShell 使用 `$env:PYTHONPATH = 'src'`，然后运行以下 Python 示例。

Only NumPy and Pillow are needed, both included in the project environment. No character assets, provider configuration or hardware are read. Set `PYTHONPATH=src` from the repository root (PowerShell: `$env:PYTHONPATH = 'src'`), then run this Python example:

```python
from PIL import Image, ImageDraw
from opendafeiyu.display_panels import RGB565PanelPair

class MemoryScreen:
    def __init__(self):
        self.image_buffer = Image.new("RGB", (240, 240))

    def image(self, tile, *, x, y):
        self.image_buffer.paste(tile, (x, y))

screens = [MemoryScreen(), MemoryScreen()]
writer = RGB565PanelPair(screens)  # max_channel_error=1 for bounded-error mode
eyes = [Image.new("RGB", (240, 240), "navy") for _ in screens]
for eye in eyes:
    ImageDraw.Draw(eye).ellipse((70, 50, 170, 190), fill="cyan")
writer.show(eyes)
print(writer.payload_bytes)  # 230400: initial two full RGB565 frames
writer.show(eyes)
print(writer.transfers)     # 2: no new writes for identical frames
```

内存示例保留 RGB 图像，回归测试独立重建 RGB565 显存。运行 `python -m pytest -q tests/test_display_panels.py` 检查像素边界、缓慢往返变化、双屏隔离及中断恢复。

The example stores RGB images; the regression tests independently reconstruct RGB565 panel memory. Run `python -m pytest -q tests/test_display_panels.py` to check edge pixels, slow changes in both directions, independent panels and interrupted transfers.

## 接入屏幕驱动 / Driver contract

传入两个已初始化的驱动对象，各自提供同步 `image(tile, x=..., y=...)` 方法。驱动把 RGB888 截断为 RGB565（红蓝右移 3 位，绿右移 2 位），负责片选、屏幕朝向和传输。局部刷新组件不初始化总线，也不控制背光。已用 Adafruit GC9A01A 3.14.6 的实际软件驱动与虚拟 GPIO/SPI 回放验证接口；这是软件验证，实体刷新率需在接线后测量。

Supply two initialized drivers with synchronous `image(tile, x=..., y=...)` methods. Each driver truncates RGB888 to RGB565 (red/blue shifted by 3, green by 2) and handles chip select, orientation and transport. This component does not initialize a bus or control a backlight. The interface was replayed through the actual Adafruit GC9A01A 3.14.6 software driver with virtual GPIO/SPI; physical refresh rate requires a connected-device measurement.

- **精确 / Exact:** `max_channel_error=0`，成功调用后显存与目标 RGB565 一致 / panel memory matches the RGB565 target after a successful call.
- **省流量 / Reduced traffic:** `max_channel_error=1`，误差以 5/6/5 位通道值计算，不是 RGB888 的 1/255。基准为上次成功发送的像素，缓慢变化不会无限累积偏差 / the bound uses quantized 5/6/5-bit channels, not 1/255 of RGB888. Comparison uses successfully sent pixels, so slow changes cannot accumulate unlimited error.
- **恢复 / Recovery:** I/O 异常原样抛出，下次调用完整重绘受影响的屏幕。屏幕复位或其他代码写屏后，调用 `writer.invalidate()`；只重绘一侧可用 `writer.invalidate(0)` 或 `writer.invalidate(1)` / I/O errors propagate, and the affected panel is fully redrawn on the next call. Invalidate both caches after a reset or external write, or select one panel by index.
- **统计 / Counters:** `payload_bytes` 只累计成功传输的像素字节，`transfers` 只累计成功窗口；不包含寻址、初始化或失败时可能已传输的数据 / counters cover successful pixel payloads and windows, excluding addressing, initialization and any bytes sent before an error.

两眼按顺序发送，不保证同步点亮；失败调用可能只更新了部分画面。由一个线程调用，接入实际驱动时保持同一坐标系与颜色转换。该组件是显示输出接口；现有身体控制和学习策略不依赖它。

The two eyes are written sequentially, not atomically; a failed call may leave a partial pair. Use one calling thread and preserve the same coordinates and color conversion in the driver. This is a display-output interface; existing body controllers and learned policies do not depend on it.
