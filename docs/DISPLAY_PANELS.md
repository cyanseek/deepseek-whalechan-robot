# 双眼显示局部刷新 / Partial eye-display refresh

`opendafeiyu.display_panels.RGB565PanelPair` 把两张同尺寸 RGB 眼图发送给独立的 RGB565 屏幕驱动，默认尺寸为 240 × 240，也支持 320 × 240 等矩形尺寸。它按变化区域更新，静止时不重发整屏；默认逐像素精确，也可选择每个颜色通道最多一个 RGB565 量化级的误差，减少渐变眼神的传输量。

The component sends two equal-size RGB eye images through independent RGB565 display drivers. The default size is 240 × 240; rectangular sizes such as 320 × 240 are also supported. It updates changed regions and sends nothing for unchanged pixels. Exact output is the default; an optional one-level error bound per RGB565 channel reduces traffic for slowly changing gradients.

## 在电脑上运行 / Run without a display

只需 NumPy 和 Pillow，不读取角色资产、API 配置或硬件。项目环境已包含这两个依赖。从仓库根目录设置 `PYTHONPATH=src`，例如 PowerShell 使用 `$env:PYTHONPATH = 'src'`，然后运行以下 Python 示例。

Only NumPy and Pillow are needed, both included in the project environment. No character assets, provider configuration or hardware are read. Set `PYTHONPATH=src` from the repository root (PowerShell: `$env:PYTHONPATH = 'src'`), then run this Python example:

```python
from PIL import Image, ImageDraw
from opendafeiyu.display_panels import RGB565PanelPair

class MemoryScreen:
    def __init__(self, size=(240, 240)):
        self.image_buffer = Image.new("RGB", size)

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

## 横屏与相反安装方向 / Landscape images and opposite orientations

若两块屏幕的原生显存为 240 × 320，眼图为 320 × 240，可在刷新组件与驱动之间使用 `QuarterTurnPanel`。左右屏可以分别顺时针、逆时针旋转90度。传入的 `size` 始终是旋转前的逻辑图像尺寸；底层驱动设为原生朝向，避免重复旋转。以下代码接着上面的内存屏示例运行：

For 320 × 240 eye images on native 240 × 320 panels, place `QuarterTurnPanel` between the refresh component and each driver. The panels can rotate 90 degrees in opposite directions. `size` always describes the logical image before rotation; leave the underlying driver in its native orientation to avoid double rotation. Continue the memory-screen example with:

```python
from opendafeiyu.display_panels import QuarterTurnPanel

native_screens = [MemoryScreen((240, 320)), MemoryScreen((240, 320))]
landscape_screens = [
    QuarterTurnPanel(native_screens[0], clockwise=True, size=(320, 240)),
    QuarterTurnPanel(native_screens[1], clockwise=False, size=(320, 240)),
]
wide_writer = RGB565PanelPair(landscape_screens, size=(320, 240), split_regions=True)
wide_eyes = [Image.new("RGB", (320, 240), color) for color in ("navy", "cyan")]
wide_writer.show(wide_eyes)
print(wide_writer.payload_bytes)  # 307200: two full rectangular RGB565 frames
```

`split_regions=True` 在8行条带内拆分相隔至少7列未变化像素的区域，避免把虹膜、高光与眼睑之间的静止像素一起发送。默认仍采用12行条带中的单个包围框。拆分可能增加调用次数；比较总线寻址开销和主控耗时后再选。它可与 `max_channel_error=1` 组合，成功写入后的像素误差约定保持不变。

`split_regions=True` splits regions separated by at least seven unchanged columns within eight-row bands, avoiding quiet space between iris, glint and lid edges. The default remains one bounding rectangle per twelve-row band. Splitting can increase call count; include addressing overhead and host execution time when choosing a mode. It can be combined with `max_channel_error=1` without changing the successful-write pixel error bound.

## 接入屏幕驱动 / Driver contract

传入两个已初始化的驱动对象，各自提供同步 `image(tile, x=..., y=...)` 方法。驱动把 RGB888 截断为 RGB565（红蓝右移 3 位，绿右移 2 位），负责片选、屏幕朝向和传输。局部刷新组件不初始化总线，也不控制背光。已用 Adafruit GC9A01A 与 ST7789 3.14.6 的实际软件驱动与虚拟 GPIO/SPI 回放验证接口；这是软件验证，实体刷新率需在接线后测量。

Supply two initialized drivers with synchronous `image(tile, x=..., y=...)` methods. Each driver truncates RGB888 to RGB565 (red/blue shifted by 3, green by 2) and handles chip select, orientation and transport. This component does not initialize a bus or control a backlight. The interface was replayed through the actual Adafruit GC9A01A and ST7789 3.14.6 software drivers with virtual GPIO/SPI; physical refresh rate requires a connected-device measurement.

- **精确 / Exact:** `max_channel_error=0`，成功调用后显存与目标 RGB565 一致 / panel memory matches the RGB565 target after a successful call.
- **省流量 / Reduced traffic:** `max_channel_error=1`，误差以 5/6/5 位通道值计算，不是 RGB888 的 1/255。基准为上次成功发送的像素，缓慢变化不会无限累积偏差 / the bound uses quantized 5/6/5-bit channels, not 1/255 of RGB888. Comparison uses successfully sent pixels, so slow changes cannot accumulate unlimited error.
- **恢复 / Recovery:** I/O 异常原样抛出，下次调用完整重绘受影响的屏幕。屏幕复位或其他代码写屏后，调用 `writer.invalidate()`；只重绘一侧可用 `writer.invalidate(0)` 或 `writer.invalidate(1)` / I/O errors propagate, and the affected panel is fully redrawn on the next call. Invalidate both caches after a reset or external write, or select one panel by index.
- **统计 / Counters:** `payload_bytes` 只累计成功传输的像素字节，`transfers` 只累计成功窗口；不包含寻址、初始化或失败时可能已传输的数据 / counters cover successful pixel payloads and windows, excluding addressing, initialization and any bytes sent before an error.

两眼按顺序发送，不保证同步点亮；失败调用可能只更新了部分画面。由一个线程调用，接入实际驱动时保持同一坐标系与颜色转换。该组件是显示输出接口；现有身体控制和学习策略不依赖它。

The two eyes are written sequentially, not atomically; a failed call may leave a partial pair. Use one calling thread and preserve the same coordinates and color conversion in the driver. This is a display-output interface; existing body controllers and learned policies do not depend on it.
