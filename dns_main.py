# dns_main.py
import time
import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFontDatabase, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QStatusBar,
)

from dns_wrapper import NumPyDnsSimulator

import numpy as np

# Simple helper: build a 256x3 uint8 LUT from color stops in 0..1
# stops: list of (pos, (r,g,b)) with pos in [0,1], r,g,b in [0,255]
def _make_lut_from_stops(stops, size: int = 256) -> np.ndarray:
    stops = sorted(stops, key=lambda s: s[0])
    lut = np.zeros((size, 3), dtype=np.uint8)

    positions = [int(round(p * (size - 1))) for p, _ in stops]
    colors = [np.array(c, dtype=np.float32) for _, c in stops]

    for i in range(len(stops) - 1):
        x0 = positions[i]
        x1 = positions[i + 1]
        c0 = colors[i]
        c1 = colors[i + 1]

        if x1 <= x0:
            lut[x0] = c0.astype(np.uint8)
            continue

        length = x1 - x0
        for j in range(length):
            t = j / float(length)
            c = (1.0 - t) * c0 + t * c1
            lut[x0 + j] = c.astype(np.uint8)

    # last entry
    lut[positions[-1]] = colors[-1].astype(np.uint8)
    return lut


def _make_gray_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = (i, i, i)
    return lut


def _make_fire_lut() -> np.ndarray:
    """Approximate 'fire' palette via HSL ramp: red → yellow, brightening."""
    import colorsys
    lut = np.zeros((256, 3), dtype=np.uint8)
    for x in range(256):
        # Hue 0..85 degrees
        h_deg = 85.0 * (x / 255.0)
        h = h_deg / 360.0
        s = 1.0
        # Lightness: 0..1 up to mid, then flat
        l = min(1.0, x / 128.0)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        lut[x] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


def _make_doom_fire_lut() -> np.ndarray:
    """Classic Doom fire palette approximated as 256 RGB colors."""
    key_colors = np.array([
        [  0,   0,   0],
        [  7,   7,   7],
        [ 31,   7,   7],
        [ 47,  15,   7],
        [ 71,  15,   7],
        [ 87,  23,   7],
        [103,  31,   7],
        [119,  31,   7],
        [143,  39,   7],
        [159,  47,   7],
        [175,  63,   7],
        [191,  71,   7],
        [199,  71,   7],
        [223,  79,   7],
        [223,  87,   7],
        [223,  87,   7],
        [215,  95,   7],
        [215,  95,   7],
        [215, 103,  15],
        [207, 111,  15],
        [207, 119,  15],
        [207, 127,  15],
        [207, 135,  23],
        [199, 135,  23],
        [199, 143,  23],
        [199, 151,  31],
        [191, 159,  31],
        [191, 159,  31],
        [191, 167,  39],
        [191, 167,  39],
        [191, 175,  47],
        [183, 175,  47],
        [183, 183,  47],
        [183, 183,  55],
        [207, 207, 111],
        [223, 223, 159],
        [239, 239, 199],
        [255, 255, 255],
    ], dtype=np.uint8)

    stops = []
    n_keys = key_colors.shape[0]
    for i in range(n_keys):
        pos = i / (n_keys - 1)
        stops.append((pos, key_colors[i].tolist()))
    return _make_lut_from_stops(stops)


def _make_viridis_lut() -> np.ndarray:
    # Approximate viridis with a few key colors from the official palette
    stops = [
        (0.0,  (68,  1, 84)),
        (0.25, (59, 82,139)),
        (0.50, (33,145,140)),
        (0.75, (94,201, 98)),
        (1.0,  (253,231, 37)),
    ]
    return _make_lut_from_stops(stops)


def _make_inferno_lut() -> np.ndarray:
    stops = [
        (0.0,  (  0,   0,   4)),
        (0.25, ( 87,  15, 109)),
        (0.50, (187,  55,  84)),
        (0.75, (249, 142,   8)),
        (1.0,  (252, 255, 164)),
    ]
    return _make_lut_from_stops(stops)


def _make_plasma_lut() -> np.ndarray:
    stops = [
        (0.0,  ( 13,   8, 135)),
        (0.25, (126,   3, 167)),
        (0.50, (203,  71, 119)),
        (0.75, (248, 149,  64)),
        (1.0,  (240, 249,  33)),
    ]
    return _make_lut_from_stops(stops)


def _make_magma_lut() -> np.ndarray:
    stops = [
        (0.0,  (  0,   0,   4)),
        (0.25, ( 73,  18,  99)),
        (0.50, (150,  50,  98)),
        (0.75, (226, 102,  73)),
        (1.0,  (252, 253, 191)),
    ]
    return _make_lut_from_stops(stops)


def _make_turbo_lut() -> np.ndarray:
    # Approximate Google's Turbo colormap with a few key stops.
    stops = [
        (0.0,  ( 48,  18,  59)),
        (0.25, ( 31, 120, 180)),
        (0.50, ( 78, 181,  75)),
        (0.75, (241, 208,  29)),
        (1.0,  (133,  32,  26)),
    ]
    return _make_lut_from_stops(stops)


GRAY_LUT      = _make_gray_lut()
INFERNO_LUT   = _make_inferno_lut()
VIRIDIS_LUT   = _make_viridis_lut()
PLASMA_LUT    = _make_plasma_lut()
MAGMA_LUT     = _make_magma_lut()
TURBO_LUT     = _make_turbo_lut()
FIRE_LUT      = _make_fire_lut()
DOOM_FIRE_LUT = _make_doom_fire_lut()

COLOR_MAPS = {
    "Gray": GRAY_LUT,
    "Inferno": INFERNO_LUT,
    "Viridis": VIRIDIS_LUT,
    "Plasma": PLASMA_LUT,
    "Magma": MAGMA_LUT,
    "Turbo": TURBO_LUT,
    "Fire": FIRE_LUT,
    "Doom": DOOM_FIRE_LUT,
}

DEFAULT_CMAP_NAME = "Doom Fire"


class MainWindow(QMainWindow):
    def __init__(self, sim: NumPyDnsSimulator) -> None:
        super().__init__()

        self.sim = sim
        self.current_cmap_name = DEFAULT_CMAP_NAME

        # --- central image label ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- small icon buttons ---
        self.start_button = QPushButton()
        self.start_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.start_button.setToolTip("Start simulation")
        self.start_button.setFixedSize(36, 36)
        self.start_button.setIconSize(QSize(24, 24))

        self.stop_button = QPushButton()
        self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_button.setToolTip("Stop simulation")
        self.stop_button.setFixedSize(36, 36)
        self.stop_button.setIconSize(QSize(24, 24))

        #self.step_button = QPushButton("Step")
        self.reset_button = QPushButton("Reset")
        self.save_button = QPushButton("Save")

        self._status_update_counter = 0

        # Variable selector
        self.variable_combo = QComboBox()
        self.variable_combo.addItems(["U", "V", "K", "Ω", "φ"])

        # Colormap selector
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(list(COLOR_MAPS.keys()))
        idx = self.cmap_combo.findText(DEFAULT_CMAP_NAME)
        if idx >= 0:
            self.cmap_combo.setCurrentIndex(idx)

        # --- layout ---
        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        #button_row.addWidget(self.step_button)
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.save_button)
        button_row.addStretch()
        button_row.addWidget(self.variable_combo)
        button_row.addWidget(self.cmap_combo)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.image_label, stretch=1)
        layout.addLayout(button_row)
        self.setCentralWidget(central)

        # --- status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.status.setFont(mono)

        self.threads_label = QLabel(self)
        #self._update_threads_label()
        self.status.addPermanentWidget(self.threads_label)

        # Timer-based simulation (no QThread)
        self.timer = QTimer(self)
        self.timer.setInterval(0)   # as fast as Qt allows
        self.timer.timeout.connect(self._on_timer)

        # signal connections
        self.start_button.clicked.connect(self.on_start_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        #self.step_button.clicked.connect(self.on_step_clicked)
        self.reset_button.clicked.connect(self.on_reset_clicked)
        self.save_button.clicked.connect(self.on_save_clicked)
        self.variable_combo.currentIndexChanged.connect(self.on_variable_changed)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)

        # window setup
        self.setWindowTitle("2D Turbulent DNS (NumPy)")
        self.resize(self.sim.px + 40, self.sim.py + 120)

        self._last_pixels_rgb: Optional[np.ndarray] = None

        # --- FPS from simulation start ---
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

        # initial draw (omega mode)
        self.variable_combo.setCurrentIndex(0)
        self.sim.set_variable(self.sim.VAR_U)

        # (optional) a nice colormap index
        self.cmap_combo.setCurrentIndex(5)

        self._update_image(self.sim.get_frame_pixels())
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), None)

        self.timer.start()  # auto-start simulation immediately

    # ------------------------------------------------------------------
    def on_start_clicked(self) -> None:
        #self._update_threads_label()
        if not self.timer.isActive():
            self.timer.start()

    def on_stop_clicked(self) -> None:
        if self.timer.isActive():
            self.timer.stop()

    def on_step_clicked(self) -> None:
        self.sim.step()
        pixels = self.sim.get_frame_pixels()
        self._update_image(pixels)
        t = self.sim.get_time()
        it = self.sim.get_iteration()
        # one manual step doesn't really change the global FPS much,
        # so we leave fps=None here
        self._update_status(t, it, fps=None)

    def on_reset_clicked(self) -> None:
        self.on_stop_clicked()
        self.sim.reset_field()

        # reset FPS baseline to "new simulation start"
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

        self._update_image(self.sim.get_frame_pixels())
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), None)

    def on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save frame",
            "frame.png",
            "PNG images (*.png);;All files (*)",
        )
        if path:
            pm = self.image_label.pixmap()
            if pm:
                pm.save(path, "PNG")

    def on_variable_changed(self, index: int) -> None:
        mapping = {
            0: self.sim.VAR_U,
            1: self.sim.VAR_V,
            2: self.sim.VAR_ENERGY,
            3: self.sim.VAR_OMEGA,
            4: self.sim.VAR_STREAM,
        }
        self.sim.set_variable(mapping.get(index, self.sim.VAR_U))
        self._update_image(self.sim.get_frame_pixels())

    def on_cmap_changed(self, name: str) -> None:
        if name in COLOR_MAPS:
            self.current_cmap_name = name
            self._update_image(self.sim.get_frame_pixels())

    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        # one DNS step per timer tick
        self.sim.step()

        # Count frames since last GUI update
        self._status_update_counter += 1

        # Update GUI only every 10 frames
        UPDATE_INTERVAL = 10
        if self._status_update_counter >= UPDATE_INTERVAL:
            pixels = self.sim.get_frame_pixels()
            self._update_image(pixels)

            # ---- FPS from simulation start ----
            now = time.time()
            elapsed = now - self._sim_start_time
            steps = self.sim.get_iteration() - self._sim_start_iter
            fps = None
            if elapsed > 0 and steps > 0:
                fps = steps / elapsed

            self._update_status(
                self.sim.get_time(),
                self.sim.get_iteration(),
                fps,
            )

            self._status_update_counter = 0

        # Optional auto-reset like before
        if self.sim.get_iteration() >= 3000:
            self.sim.reset_field()
            self._sim_start_time = time.time()
            self._sim_start_iter = self.sim.get_iteration()

    # ------------------------------------------------------------------
    def _update_image(self, pixels: np.ndarray) -> None:
        """
        Map H×W uint8 pixels through colormap and show in label.
        """
        pixels = np.asarray(pixels, dtype=np.uint8)
        if pixels.ndim != 2:
            return

        lut = COLOR_MAPS.get(self.current_cmap_name)
        if lut is None:
            # fallback: grayscale
            h, w = pixels.shape
            qimg = QImage(
                pixels.data,
                w,
                h,
                w,
                QImage.Format.Format_Grayscale8,
            )
        else:
            lut_arr = np.asarray(lut, dtype=np.uint8)
            rgb = lut_arr[pixels]  # H×W×3
            h, w, _ = rgb.shape
            self._last_pixels_rgb = rgb  # keep alive
            qimg = QImage(
                rgb.data,
                w,
                h,
                3 * w,
                QImage.Format.Format_RGB888,
            )

        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def _update_status(self, t: float, it: int, fps: Optional[float]) -> None:
        fps_str = f"{fps:4.0f}" if fps is not None else " n/a"
        txt = f"FPS: {fps_str} | Iter: {it:5d} | T: {t:6.3f}"
        self.status.showMessage(txt)


    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()

        # rotate variable (P)
        if key == Qt.Key.Key_P:
            idx = self.variable_combo.currentIndex()
            count = self.variable_combo.count()
            self.variable_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate colormap (C)
        if key == Qt.Key.Key_C:
            idx = self.cmap_combo.currentIndex()
            count = self.cmap_combo.count()
            self.cmap_combo.setCurrentIndex((idx + 1) % count)
            return

        super().keyPressEvent(event)

# ----------------------------------------------------------------------
def main() -> None:
    app = QApplication(sys.argv)
    sim = NumPyDnsSimulator()
    window = MainWindow(sim)
    window.adjustSize()
    window.setFixedSize(window.size())
    screen = app.primaryScreen().availableGeometry()
    g = window.geometry()
    g.moveCenter(screen.center())
    window.setGeometry(g)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()