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
    def __init__(self, size=(240, 240)):
        self.memory = np.zeros((size[1], size[0]), dtype=np.uint16)
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


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("error", [0, 1])
def test_rectangular_edge_and_partial_final_band(split, error):
    size = (37, 19)  # Neither height nor width is a multiple of a band.
    screens = [Screen(size), Screen(size)]
    writer = RGB565PanelPair(screens, size=size, split_regions=split, max_channel_error=error)
    frames = [Image.new("RGB", size) for _ in screens]
    writer.show(frames)
    assert writer.payload_bytes == 37 * 19 * 4
    for value in (4, 8, 12, 32, 255, 128, 0):
        for xy in ((0, 0), (36, 0), (0, 18), (36, 18), (19, 8)):
            frames[0].putpixel(xy, (value, value, value))
        writer.show(frames)
        assert np.abs(channels(screens[0].memory) - channels(words(frames[0]))).max() <= error
    assert len(screens[1].calls) == 1
    assert all(x + w <= 37 and y + h <= 19 for x, y, w, h in screens[0].calls)


def test_sparse_regions_skip_quiet_space_between_changes():
    size = (320, 240)
    frames = [Image.new("RGB", size) for _ in range(2)]
    results = []
    for split in (False, True):
        screens = [Screen(size), Screen(size)]
        writer = RGB565PanelPair(screens, size=size, split_regions=split)
        writer.show(frames)
        changed = frames[0].copy()
        changed.putpixel((0, 7), (255, 0, 0))
        changed.putpixel((319, 7), (0, 0, 255))
        before = writer.payload_bytes
        writer.show([changed, frames[1]])
        results.append(writer.payload_bytes - before)
        np.testing.assert_array_equal(screens[0].memory, words(changed))
    assert results == [640, 4]


@pytest.mark.parametrize("clockwise", [False, True])
@pytest.mark.parametrize("error", [0, 1])
def test_rotated_rectangular_partial_updates_and_failed_write(clockwise, error):
    from opendafeiyu.display_panels import QuarterTurnPanel
    size = (37, 19)
    screens = [Screen((19, 37)), Screen((19, 37))]
    adapters = [QuarterTurnPanel(screen, clockwise=direction, size=size)
                for screen, direction in zip(screens, (clockwise, not clockwise))]
    writer = RGB565PanelPair(adapters, size=size, split_regions=True, max_channel_error=error)
    pixels = np.arange(37 * 19 * 3, dtype=np.uint16).reshape(19, 37, 3).astype(np.uint8)
    frames = [Image.fromarray(pixels), Image.fromarray(255 - pixels)]
    writer.show(frames)
    for xy in ((0, 0), (36, 18), (0, 18), (36, 0), (21, 8)):
        frames[1].putpixel(xy, (255, 8, 0))
    writer.show(frames)
    for frame, screen, direction in zip(frames, screens, (clockwise, not clockwise)):
        expected = np.rot90(words(frame), -1 if direction else 1)
        assert np.abs(channels(screen.memory) - channels(expected)).max() <= error
    screens[1].fail_on = len(screens[1].calls) + 1
    frames[1].putpixel((10, 10), (0, 255, 255))
    with pytest.raises(OSError):
        writer.show(frames)
    before = writer.payload_bytes
    writer.show(frames)
    assert writer.payload_bytes - before == 37 * 19 * 2
    assert screens[1].calls[-1] == (0, 0, 19, 37)
    expected = np.rot90(words(frames[1]), 1 if clockwise else -1)
    np.testing.assert_array_equal(screens[1].memory, expected)


@pytest.mark.parametrize("size", [(0, 240), (320, -1), (True, 240), (320,), None])
def test_invalid_dimensions_rejected(size):
    with pytest.raises(ValueError):
        RGB565PanelPair([Screen(), Screen()], size=size)


def test_rotated_tiles_validate_bounds_before_driver_io():
    from opendafeiyu.display_panels import QuarterTurnPanel
    screen = Screen((19, 37))
    adapter = QuarterTurnPanel(screen, clockwise=True, size=(37, 19))
    tile = Image.new("RGB", (3, 2))
    for x, y in ((-1, 0), (35, 0), (0, 18)):
        with pytest.raises(ValueError):
            adapter.image(tile, x=x, y=y)
    assert not screen.calls
