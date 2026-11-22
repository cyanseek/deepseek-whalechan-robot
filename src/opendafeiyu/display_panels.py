"""Synchronous partial refresh for two 240 x 240 RGB565 expression displays."""

import numpy as np
from PIL import Image


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

    def __init__(self, screens, *, max_channel_error=0):
        if len(screens) != 2 or screens[0] is screens[1]:
            raise ValueError("Expected two independent display drivers")
        if type(max_channel_error) is not int or max_channel_error not in (0, 1):
            raise ValueError("max_channel_error must be 0 (exact) or 1")
        self._screens = tuple(screens)
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
            or frame.mode != "RGB" or frame.size != (240, 240)
            for frame in frames
        ):
            raise ValueError("Expected two 240 x 240 RGB images")

        for side, image in enumerate(frames):
            target = np.asarray(image) >> np.array([3, 2, 3], dtype=np.uint8)
            sent = self._sent[side]
            if sent is None:
                boxes = [(0, 0, 240, 240)]
                sent = target.copy()
            else:
                error = np.abs(target.astype(np.int16) - sent)
                changed = (error > self._max_channel_error).any(axis=2)
                boxes = []
                # One rectangle per 12-row band bounds the transaction count.
                for top in range(0, 240, 12):
                    yy, xx = np.nonzero(changed[top:top + 12])
                    if len(xx):
                        boxes.append((int(xx.min()), top + int(yy.min()),
                                      int(xx.max()) + 1, top + int(yy.max()) + 1))
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
