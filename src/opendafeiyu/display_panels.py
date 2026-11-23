"""Synchronous partial refresh for a pair of equal-size RGB565 displays."""

import numpy as np
from PIL import Image


def _display_size(size):
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)):
        raise ValueError("Expected positive integer display dimensions")
    return tuple(size)


class QuarterTurnPanel:
    """Rotate logical tiles 90 degrees into an unrotated driver's framebuffer.

    ``size`` is the logical (width, height), so the underlying driver must
    expose (height, width). Set its own rotation to zero to avoid rotating
    twice. Coordinate conversion and pixel rotation use the same direction.
    """

    def __init__(self, screen, *, clockwise, size=(320, 240)):
        self._screen = screen
        self._size = _display_size(size)
        if type(clockwise) is not bool:
            raise ValueError("clockwise must be a boolean")
        self._clockwise = clockwise

    def image(self, tile, *, x, y):
        width, height = self._size
        if (not isinstance(tile, Image.Image) or tile.mode != "RGB"
                or type(x) is not int or type(y) is not int
                or tile.width <= 0 or tile.height <= 0
                or x < 0 or y < 0 or x + tile.width > width or y + tile.height > height):
            raise ValueError("RGB tile must fit inside the logical display")
        if self._clockwise:
            self._screen.image(tile.transpose(Image.Transpose.ROTATE_270),
                               x=height - y - tile.height, y=x)
        else:
            self._screen.image(tile.transpose(Image.Transpose.ROTATE_90),
                               x=y, y=width - x - tile.width)


class RGB565PanelPair:
    """Send changed tiles through two drivers exposing ``image(tile, x=, y=)``.

    ``max_channel_error=0`` reproduces the target RGB565 pixels exactly.
    Set it to 1 to allow one quantized level per channel between updates.
    The cache tracks successful writes, not unsent target frames. Call
    ``invalidate`` after a panel reset or any write made outside this instance.

    Drivers must perform synchronous RGB888-to-RGB565 truncation (R/B >> 3,
    G >> 2). Each driver owns its panel's orientation and chip select. Use one
    calling thread. I/O exceptions propagate; the affected panel is redrawn
    fully on its next successful call. No hardware is opened by this module.
    """

    def __init__(self, screens, *, max_channel_error=0, size=(240, 240), split_regions=False):
        if len(screens) != 2 or screens[0] is screens[1]:
            raise ValueError("Expected two independent display drivers")
        if type(max_channel_error) is not int or max_channel_error not in (0, 1):
            raise ValueError("max_channel_error must be 0 (exact) or 1")
        self._screens = tuple(screens)
        self._size = _display_size(size)
        self._split_regions = split_regions
        self._max_channel_error = max_channel_error
        self._sent = [None, None]
        # Successful pixel payload only: excludes commands and failed writes.
        self.payload_bytes = 0
        self.transfers = 0

    def invalidate(self, side=None):
        """Request a full repaint of both panels, or panel index 0 or 1."""
        if side is None:
            self._sent = [None, None]
        elif type(side) is int and side in (0, 1):
            self._sent[side] = None
        else:
            raise ValueError("side must be 0, 1 or None")

    def show(self, frames):
        """Update both eyes; validate the entire pair before performing I/O.

        After a successful call each displayed RGB565 channel differs from the
        requested value by at most ``max_channel_error`` quantization levels.
        A failed call can leave a partial pair; it does not claim this bound.
        """
        if len(frames) != 2 or any(
            not isinstance(frame, Image.Image)
            or frame.mode != "RGB" or frame.size != self._size
            for frame in frames
        ):
            raise ValueError(f"Expected two {self._size[0]} x {self._size[1]} RGB images")

        for side, image in enumerate(frames):
            target = np.asarray(image) >> np.array([3, 2, 3], dtype=np.uint8)
            sent = self._sent[side]
            if sent is None:
                boxes = [(0, 0, *self._size)]
                sent = target.copy()
            else:
                error = np.abs(target.astype(np.int16) - sent)
                changed = (error > self._max_channel_error).any(axis=2)
                boxes = []
                band = 8 if self._split_regions else 12
                for top in range(0, self._size[1], band):
                    stripe = changed[top:top + band]
                    cols = np.flatnonzero(stripe.any(axis=0))
                    if not len(cols):
                        continue
                    # Split separated iris/lid edges instead of sending quiet
                    # pixels between them. Opt-in: this can mean more calls.
                    gaps = np.flatnonzero(np.diff(cols) > 7) + 1 if self._split_regions else []
                    for segment in np.split(cols, gaps):
                        left, right = int(segment[0]), int(segment[-1]) + 1
                        rows = np.flatnonzero(stripe[:, left:right].any(axis=1))
                        boxes.append((left, top + int(rows[0]), right, top + int(rows[-1]) + 1))
            for box in boxes:
                x, y, right, bottom = box
                # Even a driver that wrote part of a tile before raising needs
                # a complete repaint, not a retry against the previous cache.
                self._sent[side] = None
                self._screens[side].image(image.crop(box), x=x, y=y)
                sent[y:bottom, x:right] = target[y:bottom, x:right]
                self.payload_bytes += 2 * (right - x) * (bottom - y)
                self.transfers += 1
            self._sent[side] = sent
