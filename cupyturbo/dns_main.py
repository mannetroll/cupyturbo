# dns_main.py
import time
import sys
import os
from typing import Optional

from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtCore import QStandardPaths
from PyQt6.QtGui import QImage, QPixmap, QFontDatabase
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
    QCheckBox,
    QStyle,
    QLineEdit,
)

from cupyturbo.dns_wrapper import NumPyDnsSimulator
from cupyturbo import dns_simulator as dns_all

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
        # allow shrinking when grid size becomes smaller
        self.image_label.setSizePolicy(
            self.image_label.sizePolicy().horizontalPolicy(),
            self.image_label.sizePolicy().verticalPolicy()
        )
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- small icon buttons ---
        style = QApplication.style()

        # Start button
        self.start_button = QPushButton()
        self.start_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.setToolTip("Start simulation")
        self.start_button.setFixedSize(28, 28)
        self.start_button.setIconSize(QSize(14, 14))

        # Stop button
        self.stop_button = QPushButton()
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.setToolTip("Stop simulation")
        self.stop_button.setFixedSize(28, 28)
        self.stop_button.setIconSize(QSize(14, 14))

        # Reset button
        self.reset_button = QPushButton()
        self.reset_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reset_button.setToolTip("Reset simulation")
        self.reset_button.setFixedSize(28, 28)
        self.reset_button.setIconSize(QSize(14, 14))

        # Save button
        self.save_button = QPushButton()
        self.save_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_button.setToolTip("Save current frame")
        self.save_button.setFixedSize(28, 28)
        self.save_button.setIconSize(QSize(14, 14))

        # Folder button
        self.folder_button = QPushButton()
        self.folder_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self.folder_button.setToolTip("Save files")
        self.folder_button.setFixedSize(28, 28)
        self.folder_button.setIconSize(QSize(14, 14))


        self._status_update_counter = 0

        # Variable selector
        self.variable_combo = QComboBox()
        self.variable_combo.setToolTip("V: Variable")
        self.variable_combo.addItems(["U", "V", "K", "Ω", "φ"])

        # Grid-size selector (N)
        self.n_combo = QComboBox()
        self.n_combo.setToolTip("N: Grid Size (N)")
        self.n_combo.addItems(["128", "192", "256", "384", "512", "768", "1024", "2048", "3072", "4096"])
        self.n_combo.setCurrentText(str(self.sim.N))

        # Reynolds selector (Re)
        self.re_combo = QComboBox()
        self.re_combo.setToolTip("R: Reynolds Number (Re)")
        self.re_combo.addItems(["1", "1000", "10000", "100000", "1E6", "1E9", "1E12"])
        self.re_combo.setCurrentText(str(int(self.sim.re)))

        # K0 selector
        self.k0_combo = QComboBox()
        self.k0_combo.setToolTip("K: Initial energy peak wavenumber (K0)")
        self.k0_combo.addItems(["1", "5", "10", "25", "50"])
        self.k0_combo.setCurrentText(str(int(self.sim.k0)))

        # Colormap selector
        self.cmap_combo = QComboBox()
        self.cmap_combo.setToolTip("C: Colormaps")
        self.cmap_combo.addItems(list(COLOR_MAPS.keys()))
        idx = self.cmap_combo.findText(DEFAULT_CMAP_NAME)
        if idx >= 0:
            self.cmap_combo.setCurrentIndex(idx)

        # CFL selector
        self.cfl_combo = QComboBox()
        self.cfl_combo.setToolTip("L: Controlling Δt (CFL)")
        self.cfl_combo.addItems(["0.15", "0.25", "0.50", "0.75", "0.95"])
        self.cfl_combo.setCurrentText(str(self.sim.cfl))

        # Steps selector
        self.steps_combo = QComboBox()
        self.steps_combo.setToolTip("S: Max steps before reset/stop")
        self.steps_combo.addItems(["2000", "5000", "10000", "25000", "50000", "1E5", "1E6"])
        self.steps_combo.setCurrentText("5000")

        self.auto_reset_checkbox = QCheckBox()
        self.auto_reset_checkbox.setToolTip("If checked, simulation auto-resets")
        self.auto_reset_checkbox.setChecked(True)

        self._build_layout()

        # --- status bar ---
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.status.setFont(mono)

        self.threads_label = QLabel(self)
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
        self.folder_button.clicked.connect(self.on_folder_clicked)
        self.variable_combo.currentIndexChanged.connect(self.on_variable_changed)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        self.n_combo.currentTextChanged.connect(self.on_n_changed)
        self.re_combo.currentTextChanged.connect(self.on_re_changed)
        self.k0_combo.currentTextChanged.connect(self.on_k0_changed)
        self.cfl_combo.currentTextChanged.connect(self.on_cfl_changed)
        self.steps_combo.currentTextChanged.connect(self.on_steps_changed)

        # window setup
        # GPU/CPU title selection (no extra logic, just based on CuPy backend)
        title_backend = "CuPy" if self.sim.state.backend == "gpu" else "NumPy"
        self.setWindowTitle(f"2D Turbulence ({title_backend}) © Mannetroll")
        self.resize(self.sim.px + 40, self.sim.py + 120)

        self._last_pixels_rgb: Optional[np.ndarray] = None

        # --- FPS from simulation start ---
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

        # initial draw (omega mode)
        self.variable_combo.setCurrentIndex(3)
        self.sim.set_variable(self.sim.VAR_OMEGA)

        # (optional) a nice colormap index
        self.cmap_combo.setCurrentIndex(5)

        self._update_image(self.sim.get_frame_pixels())
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), None)

        self.timer.start()  # auto-start simulation immediately

    # ------------------------------------------------------------------

    def move_widgets(self, src_layout, dst_layout):
        """Move only widgets from src_layout into dst_layout (ignore spacers)."""
        while src_layout.count() > 0:
            item = src_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                dst_layout.addWidget(w)

    def _build_layout(self):
        """Rebuild the control layout based on current N."""
        old = self.centralWidget()
        if old is not None:
            old.setParent(None)

        central = QWidget()
        main = QVBoxLayout(central)
        main.addWidget(self.image_label)

        # First row
        row1 = QHBoxLayout()
        row1.addWidget(self.start_button)
        row1.addWidget(self.stop_button)
        row1.addWidget(self.reset_button)
        row1.addWidget(self.save_button)
        row1.addWidget(self.folder_button)
        row1.addWidget(self.cmap_combo)
        row1.addWidget(self.auto_reset_checkbox)
        row1.addWidget(self.steps_combo)
        row1.addStretch()

        # Second row
        row2 = QHBoxLayout()
        row2.addWidget(self.variable_combo)
        row2.addWidget(self.n_combo)
        row2.addWidget(self.re_combo)
        row2.addWidget(self.k0_combo)
        row2.addWidget(self.cfl_combo)
        row2.addStretch()

        # Combine into single row if large N
        if self.sim.N >= 1024:
            single = QHBoxLayout()
            self.move_widgets(row1, single)
            self.move_widgets(row2, single)
            main.addLayout(single)
        else:
            main.addLayout(row1)
            main.addLayout(row2)

        self.setCentralWidget(central)

    def _maybe_downscale(self, rgb: np.ndarray) -> np.ndarray:
        """
        Downscale full-resolution RGB (H×W×3) by factor:
            N < 768       → 1×  (100%)
            768–1024      → 2×  (50%)
            1025–3072     → 4×  (25%)
            ≥ 3072        → 6×  (17%)
        """
        N = self.sim.N

        # Determine scale factor
        if N < 768:
            scale = 1
        elif N <= 1024:
            scale = 2
        elif N <= 3072:
            scale = 4
        else:
            scale = 6

        if scale == 1:
            return rgb

        # --- Downscale by averaging blocks ---
        h, w, _ = rgb.shape
        h2 = h // scale
        w2 = w // scale

        # Ensure reshape fits
        rgb = rgb[:h2 * scale, :w2 * scale]

        # Reshape to (h2, scale, w2, scale, 3) and average
        small = rgb.reshape(h2, scale, w2, scale, 3).mean(axis=(1, 3))
        return small.astype(np.uint8)

    def _get_full_field(self, variable: str) -> np.ndarray:
        """
        Return a 2D float32 array (NZ_full × NX_full) for variable:
            'u', 'v', 'kinetic', 'omega'
        """
        S = self.sim.state

        # --------------------------
        # Direct velocity components
        # --------------------------
        if variable == "u":
            field = S.ur_full[0]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        if variable == "v":
            field = S.ur_full[1]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        # --------------------------
        # Kinetic energy |u|
        # --------------------------
        if variable == "kinetic":
            dns_all.dns_kinetic(S)  # fills ur_full[2,:,:]
            field = S.ur_full[2]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        # --------------------------
        # Physical vorticity ω
        # --------------------------
        if variable == "omega":
            dns_all.dns_om2_phys(S)  # fills ur_full[2,:,:]
            field = S.ur_full[2]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        # --------------------------
        # Unknown variable
        # --------------------------
        raise ValueError(f"Unknown variable: {variable}")

    def on_start_clicked(self) -> None:
        # reset FPS baseline to "new simulation start"
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
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
        self._update_image(self.sim.get_frame_pixels())
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), None)
        self.on_start_clicked()

    @staticmethod
    def sci_no_plus(x, decimals=0):
        x = float(x)
        s = f"{x:.{decimals}E}"
        return s.replace("E+", "E").replace("e+", "e")

    def on_folder_clicked(self) -> None:
        from PyQt6.QtCore import QStandardPaths

        # --- Build the default folder name ---
        N = self.sim.N
        Re = self.sim.re
        K0 = self.sim.k0
        CFL = self.sim.cfl
        STEPS = self.sim.get_iteration()

        folder = f"cupyturbo_{N}_{self.sci_no_plus(Re)}_{K0}_{CFL}_{STEPS}"

        # Default root = Desktop
        desktop = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )

        # --- Create non-native dialog (macOS compatible) ---
        dlg = QFileDialog(self)
        dlg.setWindowTitle(f"Case: {folder}")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setDirectory(desktop)

        # Prefill the directory edit field
        for lineedit in dlg.findChildren(QLineEdit):
            lineedit.setText(".")

        # Execute dialog
        if dlg.exec():
            base_dir = dlg.selectedFiles()[0]
        else:
            return

        # Build final path
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[SAVE] Dumping fields to folder: {folder_path}")
        self._dump_pgm_full(self._get_full_field("u"), os.path.join(folder_path, "u_velocity.pgm"))
        self._dump_pgm_full(self._get_full_field("v"), os.path.join(folder_path, "v_velocity.pgm"))
        self._dump_pgm_full(self._get_full_field("kinetic"), os.path.join(folder_path, "kinetic.pgm"))
        self._dump_pgm_full(self._get_full_field("omega"), os.path.join(folder_path, "omega.pgm"))
        print("[SAVE] Completed.")

    def on_save_clicked(self) -> None:
        # determine variable name for filename
        var_name = self.variable_combo.currentText()

        default_name = f"cupyturbo_{var_name}.png"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save frame",
            default_name,
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

    def on_n_changed(self, value: str) -> None:
        N = int(value)
        self.sim.set_N(N)

        # 1) Update the image first
        self._update_image(self.sim.get_frame_pixels())

        # 2) Compute new geometry
        # compute new window size based on downscaled image
        new_w = self.image_label.pixmap().width() + 40
        new_h = self.image_label.pixmap().height() + 120
        print("Resize to:", new_w, new_h)

        # 3) Allow window to shrink (RESET constraints)
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)

        # 4) Now resize → Qt WILL shrink
        self.resize(new_w, new_h)

        # 5) Recenter
        screen = QApplication.primaryScreen().availableGeometry()
        g = self.geometry()
        g.moveCenter(screen.center())
        self.setGeometry(g)
        self._build_layout()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

    def _recenter_window(self):
        screen = QApplication.primaryScreen().availableGeometry()
        g = self.geometry()
        g.moveCenter(screen.center())
        self.move(g.topLeft())

    def on_re_changed(self, value: str) -> None:
        self.sim.re = float(value)
        self.sim.reset_field()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        self._update_image(self.sim.get_frame_pixels())

    def on_k0_changed(self, value: str) -> None:
        self.sim.k0 = float(value)
        self.sim.reset_field()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        self._update_image(self.sim.get_frame_pixels())

    def on_cfl_changed(self, value: str) -> None:
        # update simulator CFL
        self.sim.cfl = float(value)
        self.sim.reset_field()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        self._update_image(self.sim.get_frame_pixels())


    def on_steps_changed(self, value: str) -> None:
        self.sim.max_steps = int(float(value))

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

        # Optional auto-reset using STEPS combo
        if self.sim.get_iteration() >= self.sim.max_steps:
            if self.auto_reset_checkbox.isChecked():
                # Auto-reset ON → behave as before
                self.sim.reset_field()
                self._sim_start_time = time.time()
                self._sim_start_iter = self.sim.get_iteration()
            else:
                # Auto-reset OFF → stop simulation
                self.timer.stop()
                print("Max steps reached — simulation stopped (Auto-Reset OFF).")

    # ------------------------------------------------------------------
    import os

    def _dump_pgm_full(self, arr: np.ndarray, filename: str):
        """
        Write a single component field as PGM (like dnsCudaDumpFieldAsPGMFull).

        arr: 2D float32 array (NZ_full × NX_full)
        """
        h, w = arr.shape
        minv = float(arr.min())
        maxv = float(arr.max())

        rng = maxv - minv

        with open(filename, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())

            if rng <= 1e-12:  # constant field
                f.write(bytes([128]) * (w * h))
                return

            # scale to 1..255
            norm = (arr - minv) / rng
            pix = (1.0 + norm * 254.0).round().clip(1, 255).astype(np.uint8)
            f.write(pix.tobytes())

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

            # Downscale if N = 768 or 1024
            rgb = self._maybe_downscale(rgb)

            h, w, _ = rgb.shape
            self._last_pixels_rgb = rgb  # keep alive
            qimg = QImage(
                rgb.data,
                w,
                h,
                3 * w,
                QImage.Format.Format_RGB888,
            )

        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)
        self.image_label.setPixmap(pix)
        self.image_label.adjustSize()

    def _update_status(self, t: float, it: int, fps: Optional[float]) -> None:
        fps_str = f"{fps:4.1f}" if fps is not None else " N/a"

        # DPP = Display Pixel Percentage
        N = self.sim.N
        if N < 768:
            dpp = 100  # 1× scale
        elif N <= 1024:
            dpp = 50  # 2× downscale
        elif N <= 3072:
            dpp = 25  # 4× downscale
        else:
            dpp = 17  # 6× downscale (≈16.7%)

        # Viscosity from DNS state
        visc = float(self.sim.state.visc)

        txt = (
            f"FPS: {fps_str} | Iter: {it:5d} | T: {t:6.3f} "
            f"| DPP: {dpp}% | Visc: {visc:12.10f}"
        )

        self.status.showMessage(txt)

    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()

        # rotate variable (V)
        if key == Qt.Key.Key_V:
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

        # rotate n_combo (N)
        if key == Qt.Key.Key_N:
            idx = self.n_combo.currentIndex()
            count = self.n_combo.count()
            self.n_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate Reynolds (R)
        if key == Qt.Key.Key_R:
            idx = self.re_combo.currentIndex()
            count = self.re_combo.count()
            self.re_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate K0 (K)
        if key == Qt.Key.Key_K:
            idx = self.k0_combo.currentIndex()
            count = self.k0_combo.count()
            self.k0_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate CFL (L)
        if key == Qt.Key.Key_L:
            idx = self.cfl_combo.currentIndex()
            count = self.cfl_combo.count()
            self.cfl_combo.setCurrentIndex((idx + 1) % count)
            return

        # rotate steps (S)
        if key == Qt.Key.Key_S:
            idx = self.steps_combo.currentIndex()
            count = self.steps_combo.count()
            self.steps_combo.setCurrentIndex((idx + 1) % count)
            return

        super().keyPressEvent(event)

# ----------------------------------------------------------------------
def main() -> None:
    app = QApplication(sys.argv)
    sim = NumPyDnsSimulator()
    window = MainWindow(sim)
    screen = app.primaryScreen().availableGeometry()
    g = window.geometry()
    g.moveCenter(screen.center())
    window.setGeometry(g)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()