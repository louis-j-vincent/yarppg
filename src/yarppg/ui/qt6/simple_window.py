"""Provides a PyQt window for displaying rPPG processing in real-time."""

import csv
import dataclasses
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import pyqtgraph
import scipy.signal
from PyQt6 import QtCore, QtWidgets

import yarppg
from yarppg.ui.qt6 import camera, utils
from yarppg.reliability import SignalReliabilityEstimator

@dataclasses.dataclass
class SimpleQt6WindowSettings(yarppg.UiSettings):
    blursize: int | None = None
    roi_alpha: float = 0.0
    video: int | str = 0
    frame_delay: float = float("nan")
    use_reliability: bool = False  # <— Nouveau flag
    metrics_csv_path: str | None = None



class SimpleQt6Window(QtWidgets.QMainWindow):
    """A simple window displaying the webcam feed and processed signals."""

    new_image = QtCore.pyqtSignal(np.ndarray)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        blursize: int | None = None,
        roi_alpha: float = 0,
        use_reliability: bool = False,   # <— Nouveau paramètre
        metrics_csv_path: str | None = None,
    ):
        super().__init__(parent=parent)

        pyqtgraph.setConfigOptions(
            imageAxisOrder="row-major", antialias=True, foreground="k", background="w"
        )

        self.blursize = blursize
        self.roi_alpha = roi_alpha
        self.use_reliability = use_reliability
        self.metrics_csv_path = Path(metrics_csv_path) if metrics_csv_path else None
        self.reliability_labels: dict[str, QtWidgets.QLabel] = {}

        self.history = deque(maxlen=150)
        self.hr_history = deque(maxlen=1500)
        self.latest_metrics: dict[str, float] | None = None
        self.last_hr_bpm: float = float("nan")
        self.csv_file: TextIO | None = None
        self.csv_writer: csv.writer | None = None
        self.setWindowTitle("yet another rPPG")

        self._init_ui()
        self.tracker = yarppg.FpsTracker()
        self.new_image.connect(self.update_image)

        # Performance optimization: frame counter for reducing reliability calculations
        self.frame_count = 0
        self.reliability_update_interval = 5  # Update reliability every 5 frames

        # Charger le module reliability seulement si nécessaire
        if self.use_reliability:
            self.reliability = SignalReliabilityEstimator(fs=30.0)
            self._init_csv_logging()

        

    def _init_ui(self) -> None:
        child = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        child.setLayout(layout)
        self.setCentralWidget(child)

        graph = pyqtgraph.GraphicsLayoutWidget()

        # --- Left side: camera + HR plot stacked vertically ---
        left_panel = pyqtgraph.GraphicsLayoutWidget()
        layout.addWidget(left_panel, 0, 0)

        # Camera view
        self.img_item = pyqtgraph.ImageItem(axisOrder="row-major")
        vb = left_panel.addViewBox(row=0, col=0, invertX=True, invertY=True, lockAspect=True)
        vb.addItem(self.img_item)

        # HR history plot removed - keeping only camera view

        self.hrv_label = QtWidgets.QLabel("HRV: --")
        font = self.hrv_label.font()
        font.setPointSize(10)
        self.hrv_label.setFont(font)

        layout = self.centralWidget().layout()
        if self.use_reliability:
            self._init_reliability_labels(layout)
            layout.addWidget(
                self.hrv_label, 6, 1, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
            )
        else:
            layout.addWidget(
                self.hrv_label, 2, 1, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter
            )

        # --- Right side: main + RGB plots ---
        grid = self._make_plots()
        layout.addWidget(grid, 0, 1)

        self.fps_label = QtWidgets.QLabel("FPS:")
        layout.addWidget(
            self.fps_label, 1, 0, alignment=QtCore.Qt.AlignmentFlag.AlignBottom
        )
        self.hr_label = QtWidgets.QLabel("HR:")
        font = self.hr_label.font()
        font.setPointSize(24)
        self.hr_label.setFont(font)
        layout.addWidget(
            self.hr_label, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )

        # --- Reliability plot (optional) ---
        if self.use_reliability:
            # Reliability plot removed - keeping only quality score display
            pass


    def _make_plots(self) -> pyqtgraph.GraphicsLayoutWidget:
        # We create a 2-row layout with linked x-axes.
        # The first plot shows the signal obtained through the processor.
        # The second plot shows average R, G and B channels in the ROI.
        grid = pyqtgraph.GraphicsLayoutWidget()
        main_plot: pyqtgraph.PlotItem = grid.addPlot(row=0, col=0)  # type: ignore
        self.rgb_plot: pyqtgraph.PlotItem = grid.addPlot(row=1, col=0)  # type: ignore
        self.rgb_plot.setXLink(main_plot.vb)  # type: ignore[attr-defined]
        main_plot.hideAxis("bottom")
        main_plot.hideAxis("left")
        self.rgb_plot.hideAxis("left")

        self.plots = [main_plot]

        self.lines = [main_plot.plot(pen=pyqtgraph.mkPen("k", width=3))]
        for c in "rgb":
            pen = pyqtgraph.mkPen(c, width=1.5)
            line, plot = utils.add_multiaxis_plot(self.rgb_plot, pen=pen)
            self.plots.append(plot)
            self.lines.append(line)

        for plot in self.plots:
            plot.disableAutoRange()  # type: ignore

        return grid

    def _init_reliability_labels(self, layout: QtWidgets.QGridLayout) -> None:
        """Initialize labels that display detailed reliability metrics."""
        def make_label(text: str, bold: bool = False) -> QtWidgets.QLabel:
            label = QtWidgets.QLabel(text)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            font = label.font()
            font.setPointSize(10)
            font.setBold(bold)
            label.setFont(font)
            return label

        rows = [
            ("reliability", "Signal Reliability", True),
            ("spectral_quality", "Spectral Quality", False),
            ("hr_stability", "HR Stability", False),
            ("morphology_quality", "Morphology Quality", False),
        ]
        for idx, (key, title, bold) in enumerate(rows, start=2):
            label = make_label(f"{title}: --", bold)
            self.reliability_labels[key] = label
            layout.addWidget(label, idx, 1, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

    def _init_csv_logging(self) -> None:
        """Prepare CSV logging of reliability metrics per frame."""
        if not self.use_reliability:
            return

        if self.metrics_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_dir = Path.cwd() / "logs"
            default_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_csv_path = default_dir / f"yarppg_metrics_{timestamp}.csv"

        if self.metrics_csv_path.suffix == "":
            # Ensure file has a suffix to avoid directory confusion
            self.metrics_csv_path = self.metrics_csv_path.with_suffix(".csv")

        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.metrics_csv_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "frame",
                "timestamp",
                "hr_bpm",
                "reliability",
                "spectral_quality",
                "hr_stability",
                "morphology_quality",
            ]
        )

    def update_image(self, frame: np.ndarray) -> None:
        """Update image plot item with new frame."""
        self.img_item.setImage(frame)

    def _handle_roi(
        self, frame: np.ndarray, roi: yarppg.RegionOfInterest
    ) -> np.ndarray:
        # Create a copy to avoid modifying the original frame
        processed_frame = frame.copy()
        
        # Apply blur if specified
        if self.blursize is not None and roi.face_rect is not None:
            yarppg.pixelate(processed_frame, roi.face_rect, size=self.blursize)

        # Create ROI-only display with modified color channels
        roi_frame = self._create_roi_display(processed_frame, roi)
        
        # Apply overlay mask if alpha > 0
        if self.roi_alpha > 0:
            roi_frame = yarppg.roi.overlay_mask(
                roi_frame, roi.mask == 1, color=(98, 3, 252), alpha=self.roi_alpha
            )

        return roi_frame

    def _create_roi_display(self, frame: np.ndarray, roi: yarppg.RegionOfInterest) -> np.ndarray:
        """Create ROI-only display with modified color channels.
        
        Shows only the bottom face ROI zone with:
        - G and B channels set to 0
        - R channel = R/B + R/G
        
        Optimized version for better performance.
        """
        # Performance optimization: reduce resolution for faster processing
        scale_factor = 0.5  # Reduce to half resolution
        if scale_factor != 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            roi_mask_small = cv2.resize(roi.mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            frame_small = frame
            roi_mask_small = roi.mask
        
        # Create a black frame of the same size
        roi_frame = np.zeros_like(frame_small)
        
        # Get the ROI mask (where mask == 1)
        roi_mask = roi_mask_small == 1
        
        if not np.any(roi_mask):
            # No ROI detected, return black frame
            return cv2.resize(roi_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Extract RGB channels from the original frame - use float32 for better performance
        r_channel = frame_small[:, :, 0].astype(np.float32)
        g_channel = frame_small[:, :, 1].astype(np.float32) 
        b_channel = frame_small[:, :, 2].astype(np.float32)
        
        # Avoid division by zero using numpy's where for vectorized operations
        g_channel_safe = np.where(g_channel == 0, 1.0, g_channel)
        b_channel_safe = np.where(b_channel == 0, 1.0, b_channel)
        
        # Calculate modified R channel: R/B + R/G (vectorized operation)
        modified_r = (r_channel / b_channel_safe) + (r_channel / g_channel_safe)
        
        # Normalize to 0-255 range and convert to uint8
        modified_r = np.clip(modified_r, 0, 255).astype(np.uint8)
        
        # Apply the modified channels only to ROI areas (vectorized assignment)
        roi_frame[roi_mask, 0] = modified_r[roi_mask]  # R channel
        # G and B channels are already 0 from np.zeros_like
        
        # Resize back to original resolution
        if scale_factor != 1.0:
            roi_frame = cv2.resize(roi_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return roi_frame

    def _handle_signals(self, result: yarppg.RppgResult) -> None:
        rgb = result.roi_mean
        self.history.append((result.value, rgb.r, rgb.g, rgb.b))
        data = np.asarray(self.history)

        self.plots[0].setXRange(0, len(data))
        for i in range(4):
            self.lines[i].setData(np.arange(len(data)), data[:, i])
            self.plots[i].setYRange(*utils.get_autorange(data[:, i]))

        # --- Optional reliability (compute each frame; update UI less often) ---
        if self.use_reliability:
            metrics = self.reliability.compute_metrics(
                data[:, 0], np.asarray(self.hr_history)
            )
            self.latest_metrics = metrics
            if self.frame_count % self.reliability_update_interval == 0:
                self._update_reliability_labels(metrics)

            # Reliability curve plotting removed - keeping only quality score display


    def _update_reliability_labels(self, metrics: dict[str, float]) -> None:
        """Update detailed reliability indicators."""
        entries = [
            ("reliability", "Signal Reliability"),
            ("spectral_quality", "Spectral Quality"),
            ("hr_stability", "HR Stability"),
            ("morphology_quality", "Morphology Quality"),
        ]

        for key, title in entries:
            label = self.reliability_labels.get(key)
            if label is None:
                continue

            value = metrics.get(key, float("nan"))
            if not np.isfinite(value):
                style = "color: gray;"
                if key == "reliability":
                    style += " font-weight: bold;"
                label.setText(f"{title}: --")
                label.setStyleSheet(style)
                continue

            color, quality = self._score_to_color_label(value)
            text = f"{title}: {value:.2f}"
            if key == "reliability":
                text += f" ({quality})"
                style = f"color: {color}; font-weight: bold;"
            else:
                text += f" [{quality}]"
                style = f"color: {color};"
            label.setText(text)
            label.setStyleSheet(style)

    @staticmethod
    def _score_to_color_label(value: float) -> tuple[str, str]:
        """Classify a reliability score into color-coded categories."""
        if value > 0.5:
            return "green", "Good"
        if value > 0.3:
            return "orange", "Medium"
        return "red", "Poor"

    def _log_frame(self) -> None:
        """Append current HR and reliability metrics to the CSV log."""
        if not self.use_reliability or self.csv_writer is None:
            return

        metrics = self.latest_metrics or {}

        def fmt(val: float | None) -> str:
            if val is None:
                return ""
            try:
                number = float(val)
            except (TypeError, ValueError):
                return ""
            if np.isfinite(number):
                return f"{number:.6f}"
            return ""

        timestamp = datetime.now().isoformat(timespec="milliseconds")
        row = [
            self.frame_count,
            timestamp,
            fmt(self.last_hr_bpm),
            fmt(metrics.get("reliability")),
            fmt(metrics.get("spectral_quality")),
            fmt(metrics.get("hr_stability")),
            fmt(metrics.get("morphology_quality")),
        ]
        self.csv_writer.writerow(row)
        if self.csv_file is not None:
            self.csv_file.flush()

    def _close_csv_logging(self) -> None:
        """Close CSV resources if they were opened."""
        if self.csv_file is not None:
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None

    def _compute_hrv(self, hr_values: np.ndarray) -> float:
        """Compute heart rate variability (SDNN in seconds) from HR history."""
        if len(hr_values) < 5:
            return np.nan
        rr_intervals = 60.0 / hr_values  # convert BPM → RR intervals
        return np.std(rr_intervals)      # SDNN

    def _handle_hrvalue(self, value: float) -> None:
        """Update user interface with the new HR value."""
        if np.isfinite(value) and value > 0:
            hr_bpm = self.tracker.fps * 60 / value
            self.hr_label.setText(f"HR: {hr_bpm:.1f}")
            self.last_hr_bpm = hr_bpm

            # --- push to history ---
            self.hr_history.append(hr_bpm)
            # HR history plotting removed - keeping only HR display and HRV calculation

            # --- compute HRV ---
            window_size = 200
            if len(self.hr_history) > window_size:
                y = np.asarray(self.hr_history, dtype=float)
                hrv_val = self._compute_hrv(y)
                if np.isfinite(hrv_val):
                    self.hrv_label.setText(f"HRV: {hrv_val*1000:.0f} ms")  # conversion en ms
                else:
                    self.hrv_label.setText("HRV: --")
        else:
            self.last_hr_bpm = float("nan")

    def _update_fps(self):
        self.tracker.tick()
        self.fps_label.setText(f"FPS: {self.tracker.fps:.1f}")

    def on_result(self, result: yarppg.RppgResult, frame: np.ndarray) -> None:
        """Update user interface with the new rPPG results."""
        self.frame_count += 1  # Increment frame counter for performance optimization
        self._update_fps()
        self.new_image.emit(self._handle_roi(frame, result.roi))
        self._handle_signals(result)
        self._handle_hrvalue(result.hr)
        self._log_frame()

    def closeEvent(self, event):  # noqa: N802
        """Ensure logging resources are released."""
        self._close_csv_logging()
        super().closeEvent(event)

    def keyPressEvent(self, e):  # noqa: N802
        """Handle key presses. Closes the window on Q."""
        if e.key() == ord("Q"):
            self.close()

    @staticmethod
    def _smooth(y: np.ndarray, window_size: int = 5) -> np.ndarray:
        if len(y) < window_size:
            return y
        kernel = np.ones(window_size) / window_size
        return np.convolve(y, kernel, mode="valid")


def launch_window(rppg: yarppg.Rppg, config: SimpleQt6WindowSettings) -> int:
    """Launch a simple Qt6-based GUI visualizing rPPG results in real-time."""
    app = QtWidgets.QApplication([])
    win = SimpleQt6Window(
        blursize=config.blursize, 
        roi_alpha=config.roi_alpha,
        use_reliability=config.use_reliability,
        metrics_csv_path=config.metrics_csv_path,
    )

    cam = camera.Camera(config.video, delay_frames=config.frame_delay)
    cam.frame_received.connect(
        lambda frame: win.on_result(rppg.process_frame(frame), frame)
    )
    cam.start()

    win.show()
    ret = app.exec()
    cam.stop()
    return ret


if __name__ == "__main__":
    b, a = scipy.signal.iirfilter(2, [0.7, 1.8], fs=30, btype="band")
    livefilter = yarppg.DigitalFilter(b, a)
    processor = yarppg.FilteredProcessor(yarppg.Processor(), livefilter)

    rppg = yarppg.Rppg(processor=processor)
    launch_window(rppg, yarppg.settings.get_config(["ui=qt6_simple"]).ui)
