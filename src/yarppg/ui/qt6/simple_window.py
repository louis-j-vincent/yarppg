"""Provides a PyQt window for displaying rPPG processing in real-time."""

import dataclasses
from collections import deque

import numpy as np
import pyqtgraph
import scipy.signal
from PyQt6 import QtCore, QtWidgets

import yarppg
from yarppg.ui.qt6 import camera, utils

@dataclasses.dataclass
class SimpleQt6WindowSettings(yarppg.UiSettings):
    """Settings for the simple Qt6 window."""

    blursize: int | None = None
    roi_alpha: float = 0.0
    video: int | str = 0
    frame_delay: float = float("nan")


class SimpleQt6Window(QtWidgets.QMainWindow):
    """A simple window displaying the webcam feed and processed signals."""

    new_image = QtCore.pyqtSignal(np.ndarray)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        blursize: int | None = None,
        roi_alpha: float = 0,
    ):
        super().__init__(parent=parent)

        pyqtgraph.setConfigOptions(
            imageAxisOrder="row-major", antialias=True, foreground="k", background="w"
        )

        self.blursize = blursize
        self.roi_alpha = roi_alpha

        self.history = deque(maxlen=150)
        self.hr_history = deque(maxlen=1500)  # ~50s at 30 FPS 
        self.setWindowTitle("yet another rPPG")
        self._init_ui()
        self.tracker = yarppg.FpsTracker()
        self.new_image.connect(self.update_image)
        

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

        # HR history plot (independent x-axis)
        self.hr_plot = left_panel.addPlot(row=1, col=0)
        self.hr_plot.setTitle("HR History (BPM)")
        self.hr_plot.setLabel("left", "BPM")
        self.hr_plot.setLabel("bottom", "Seconds")
        
        self.hr_line = self.hr_plot.plot(pen=pyqtgraph.mkPen("m", width=2))

        # --- Fond coloré statique (zones BPM) ---
        for (ymin, ymax, color) in [
            (0, 60, (150, 200, 255, 80)),   # bleu clair : repos
            (60, 90, (180, 255, 180, 80)),  # vert clair : normal
            (90, 150, (255, 160, 160, 80)), # rouge clair : élevé
        ]:
            region = pyqtgraph.LinearRegionItem(values=(ymin, ymax), orientation='horizontal')
            region.setBrush(pyqtgraph.mkBrush(color))
            region.setZValue(-10)  # derrière la courbe
            region.setMovable(False)
            self.hr_plot.addItem(region)

        # --- Indicateur de qualité du signal ---
        self.quality_label = QtWidgets.QLabel("Quality: --")
        self.quality_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        font = self.quality_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.quality_label.setFont(font)

        self.hrv_label = QtWidgets.QLabel("HRV: --")
        font = self.hrv_label.font()
        font.setPointSize(10)
        self.hrv_label.setFont(font)

        layout = self.centralWidget().layout()
        layout.addWidget(self.hrv_label, 3, 1, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        # Ajout au layout principal, en dessous du HR plot
        layout = self.centralWidget().layout()
        layout.addWidget(self.quality_label, 2, 1, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

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

    def update_image(self, frame: np.ndarray) -> None:
        """Update image plot item with new frame."""
        self.img_item.setImage(frame)

    def _handle_roi(
        self, frame: np.ndarray, roi: yarppg.RegionOfInterest
    ) -> np.ndarray:
        if self.blursize is not None and roi.face_rect is not None:
            yarppg.pixelate(frame, roi.face_rect, size=self.blursize)

        frame = yarppg.roi.overlay_mask(
            frame, roi.mask == 1, color=(98, 3, 252), alpha=self.roi_alpha
        )

        return frame

    def _handle_signals(self, result: yarppg.RppgResult) -> None:
        rgb = result.roi_mean
        self.history.append((result.value, rgb.r, rgb.g, rgb.b))
        data = np.asarray(self.history)

        self.plots[0].setXRange(0, len(data))  # type: ignore
        for i in range(4):
            self.lines[i].setData(np.arange(len(data)), data[:, i])
            self.plots[i].setYRange(*utils.get_autorange(data[:, i]))  # type: ignore

        # --- Évaluer la qualité du signal ---
        quality = self._estimate_quality(data[:, 0])  # colonne 0 = signal filtré
        self._update_quality_label(quality)

    def _estimate_quality(self, signal: np.ndarray, fs: float = 30.0) -> float:
        """Estimate signal quality based on spectral energy in the heart-rate band."""
        if len(signal) < 60:
            return np.nan
        f, pxx = scipy.signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
        band_mask = (f >= 0.7) & (f <= 1.8)
        band_power = np.trapz(pxx[band_mask], f[band_mask])
        total_power = np.trapz(pxx, f)
        return band_power / total_power if total_power > 0 else np.nan

    def _update_quality_label(self, q: float):
        """Update the quality indicator text and color."""
        if not np.isfinite(q):
            self.quality_label.setText("Quality of Signal: --")
            self.quality_label.setStyleSheet("color: gray;")
            return

        if q > 0.5:
            color, label = "green", "Good"
        elif q > 0.3:
            color, label = "orange", "Medium"
        else:
            color, label = "red", "Poor"

        self.quality_label.setText(f"Spectral Energy Ratio (SER): {q:.2f} - Quality of Signal: {label}")
        self.quality_label.setStyleSheet(f"color: {color}; font-weight: bold;")

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

            # --- push to history ---
            self.hr_history.append(hr_bpm)
            y = np.asarray(self.hr_history, dtype=float)

            # --- smoothing ---
            window_size = 200
            y_smooth = self._smooth(y, window_size=window_size)

            # --- x-axis in seconds ---
            dt = 1.0 / max(1e-6, self.tracker.fps)
            x = np.arange(len(y_smooth)) * dt

            if len(y_smooth) > window_size : # plot after a 100 points in order to avoid initial jitter

                # --- update plot ---
                self.hr_plot.setXRange(0, x[-1] if len(x) > 0 else 1)
                if len(y_smooth) > 1:
                    ymin, ymax = np.nanmin(y_smooth), np.nanmax(y_smooth)
                    margin = 0.1 * (ymax - ymin + 1e-6)
                    self.hr_plot.setYRange(ymin - margin, ymax + margin)
                self.hr_line.setData(x, y_smooth)

            # --- compute HRV ---
            if len(self.hr_history) > window_size:
                y = np.asarray(self.hr_history, dtype=float)
                hrv_val = self._compute_hrv(y)
                if np.isfinite(hrv_val):
                    self.hrv_label.setText(f"HRV: {hrv_val*1000:.0f} ms")  # conversion en ms
                else:
                    self.hrv_label.setText("HRV: --")


    def _update_fps(self):
        self.tracker.tick()
        self.fps_label.setText(f"FPS: {self.tracker.fps:.1f}")

    def on_result(self, result: yarppg.RppgResult, frame: np.ndarray) -> None:
        """Update user interface with the new rPPG results."""
        self._update_fps()
        self.new_image.emit(self._handle_roi(frame, result.roi))
        self._handle_signals(result)
        self._handle_hrvalue(result.hr)

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
    win = SimpleQt6Window(blursize=config.blursize, roi_alpha=config.roi_alpha)

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
