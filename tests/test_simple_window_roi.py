import types
import unittest

import numpy as np

from yarppg import pixelate  # noqa: F401 - ensure yarppg.pixelate exists for method
from yarppg.containers import RegionOfInterest
from yarppg.ui.qt6.simple_window_old import SimpleQt6Window


class DummyWindow(SimpleQt6Window):
    """Utility to access _handle_roi without starting a QApplication."""

    def __init__(self):
        # bypass QMainWindow initialization
        pass


class HandleRoiMaskingTest(unittest.TestCase):
    def setUp(self):
        self.window = types.SimpleNamespace(
            blursize=None,
            roi_alpha=0.0,
            use_reliability=False,
            _face_detected=False,
        )

    def test_pixels_outside_roi_are_zeroed(self):
        h, w = 60, 80
        frame = np.full((h, w, 3), 100, dtype=np.uint8)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:40, 20:60] = 1

        roi = RegionOfInterest(
            mask=mask,
            baseimg=frame.copy(),
            face_rect=(20, 10, 40, 30),
        )

        processed = SimpleQt6Window._handle_roi(self.window, frame.copy(), roi)

        roi_mask = mask.astype(bool)
        self.assertTrue(np.all(processed[~roi_mask] == 0))
        self.assertTrue(np.all((processed[roi_mask, 0] >= 0) & (processed[roi_mask, 0] <= 255)))
        self.assertTrue(np.all(processed[roi_mask, 1:] == 0))


if __name__ == "__main__":
    unittest.main()
