"""Reconstruct panel memory independently from transmitted RGB565 tiles."""
import numpy as np
import pytest
from PIL import Image

from opendafeiyu.display_panels import RGB565PanelPair


def words(image):
    rgb = np.asarray(image).astype(np.uint16)
    return ((rgb[..., 0] & 248) << 8) | ((rgb[..., 1] & 252) << 3) | (rgb[..., 2] >> 3)


def channels(pixels):
    return np.stack([pixels >> 11, (pixels >> 5) & 63, pixels & 31], axis=-1).astype(int)


class Screen:
    def __init__(self):
        self.memory = np.zeros((240, 240), dtype=np.uint16)
        self.calls = []
        self.fail_on = None

    def image(self, tile, *, x, y):
        self.calls.append((x, y, tile.width, tile.height))
        self.memory[y:y + tile.height, x:x + tile.width] = words(tile)
        if len(self.calls) == self.fail_on:
            raise OSError("deliberate interrupted transfer")


def setup(error=0):
    screens = [Screen(), Screen()]
    return screens, RGB565PanelPair(screens, max_channel_error=error)


def blank():
    return [Image.new("RGB", (240, 240)) for _ in range(2)]


def test_exact_pixels_edges_idle_and_separate_screens():
    screens, writer = setup()
    frames = blank()
    frames[0].putpixel((0, 0), (255, 0, 0))
    frames[1].putpixel((239, 239), (0, 255, 255))
    writer.show(frames)
    assert screens[0].memory[0, 0] == 0xF800
    assert screens[1].memory[-1, -1] == 0x07FF
    assert writer.payload_bytes == 230400
    writer.show(frames)
    assert writer.transfers == 2
    # Reuse and mutate the input objects: cache must not alias them.
    for y in (11, 12, 239):
        frames[0].putpixel((239, y), (0, 0, 255))
    writer.show(frames)
    assert writer.payload_bytes == 230406
    assert len(screens[1].calls) == 1
    for frame, screen in zip(frames, screens):
        np.testing.assert_array_equal(screen.memory, words(frame))


def test_unrepresentable_rgb888_changes_need_no_transfer():
    screens, writer = setup()
    frames = blank()
    writer.show(frames)
    frames[0].paste((7, 3, 7), (0, 0, 240, 240))
    writer.show(frames)
    assert writer.transfers == 2
    assert not screens[0].memory.any()


@pytest.mark.parametrize("error", [0, 1])
def test_gradual_bidirectional_changes_compare_to_last_sent_pixels(error):
    screens, writer = setup(error)
    frames = blank()
    writer.show(frames)
    for value in [*range(0, 256, 4), *range(252, -1, -4)]:
        frames[1].putpixel((231, 239), (value, value, value))
        writer.show(frames)
        delta = channels(screens[1].memory) - channels(words(frames[1]))
        assert np.abs(delta).max() <= error
    assert len(screens[0].calls) == 1
    before = writer.transfers
    writer.show(frames)
    assert writer.transfers == before


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("error", [0, 1])
def test_failure_mid_frame_redraws_failed_eye_using_new_target(side, error):
    screens, writer = setup(error)
    frames = blank()
    writer.show(frames)
    frames[side].putpixel((2, 2), (255, 0, 0))
    frames[side].putpixel((7, 130), (0, 255, 0))
    screens[side].fail_on = 3  # Initial full frame, first tile, then failed tile.
    with pytest.raises(OSError, match="interrupted"):
        writer.show(frames)
    assert writer.transfers == 3  # Failed tile is not counted as successful.
    before = writer.payload_bytes
    frames = blank()  # Caller may supply a different expression after failure.
    writer.show(frames)
    assert writer.payload_bytes - before == 115200
    assert screens[side].calls[-1] == (0, 0, 240, 240)
    assert all(not screen.memory.any() for screen in screens)


def test_failed_initial_frame_is_retried_and_second_eye_waits():
    screens, writer = setup()
    screens[0].fail_on = 1
    with pytest.raises(OSError):
        writer.show(blank())
    assert not screens[1].calls
    assert writer.payload_bytes == writer.transfers == 0
    writer.show(blank())
    assert writer.payload_bytes == 230400


def test_invalidate_after_external_reset_repaints_only_requested_screens():
    screens, writer = setup()
    frames = [Image.new("RGB", (240, 240), "white") for _ in range(2)]
    writer.show(frames)
    screens[1].memory.fill(0)
    writer.invalidate(1)
    writer.show(frames)
    assert [len(s.calls) for s in screens] == [1, 2]
    assert (screens[1].memory == 65535).all()
    writer.invalidate()
    writer.show(frames)
    assert [len(s.calls) for s in screens] == [2, 3]


@pytest.mark.parametrize("invalid", [True, -1, 2, 0.5, "1"])
def test_invalid_error_is_rejected(invalid):
    with pytest.raises(ValueError):
        RGB565PanelPair([Screen(), Screen()], max_channel_error=invalid)


@pytest.mark.parametrize("invalid", [True, -1, 2, "0"])
def test_invalid_side_is_rejected_without_io(invalid):
    screens, writer = setup()
    with pytest.raises(ValueError):
        writer.invalidate(invalid)
    assert not any(s.calls for s in screens)


@pytest.mark.parametrize("bad", [None, Image.new("RGBA", (240, 240)), Image.new("RGB", (239, 240))])
def test_entire_pair_validated_before_any_write(bad):
    screens, writer = setup()
    with pytest.raises(ValueError):
        writer.show([blank()[0], bad])
    assert not any(s.calls for s in screens)


def test_wrong_pair_count_and_shared_driver_rejected():
    screen = Screen()
    with pytest.raises(ValueError):
        RGB565PanelPair([screen, screen])
    with pytest.raises(ValueError):
        RGB565PanelPair([screen])
    screens, writer = setup()
    with pytest.raises(ValueError):
        writer.show([blank()[0]])
    assert not any(s.calls for s in screens)
