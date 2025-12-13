# dns_main_oglw.py
# (Option C: QOpenGLWidget + textures + LUT shader) — Win11Pro version
import colorsys
import ctypes
import os
import sys
import time
from typing import Any, Optional, Tuple

import numpy as np

from PyQt6.QtCore import QSize, QTimer, Qt, QStandardPaths
from PyQt6.QtGui import (
    QFontDatabase,
    QKeySequence,
    QShortcut,
    QSurfaceFormat,
)
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

from PyQt6.QtOpenGLWidgets import QOpenGLWidget

# IMPORTANT (Win11 / your PyQt6 build):
# These are NOT in PyQt6.QtGui for you. Import from QtOpenGL.
from PyQt6.QtOpenGL import (
    QOpenGLShaderProgram,
    QOpenGLShader,
    QOpenGLBuffer,
    QOpenGLVertexArrayObject,
)

from cupyturbo import dns_simulator as dns_all
from cupyturbo.dns_wrapper import NumPyDnsSimulator


# -----------------------------------------------------------------------------
# OpenGL numeric constants (avoid gl.GL_* attributes: not exposed in your build)
# -----------------------------------------------------------------------------
GL_DEPTH_TEST = 0x0B71
GL_COLOR_BUFFER_BIT = 0x00004000

GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0 = 0x84C0
GL_TEXTURE1 = 0x84C1

GL_UNPACK_ALIGNMENT = 0x0CF5
GL_NEAREST = 0x2600
GL_CLAMP_TO_EDGE = 0x812F

GL_TRIANGLES = 0x0004
GL_FLOAT = 0x1406

GL_R8 = 0x8229
GL_RED = 0x1903

GL_RGB = 0x1907
GL_RGB8 = 0x8051

GL_UNSIGNED_BYTE = 0x1401


# -----------------------------------------------------------------------------
# LUT helpers
# -----------------------------------------------------------------------------
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

    lut[positions[-1]] = colors[-1].astype(np.uint8)
    return lut


def _make_gray_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = (i, i, i)
    return lut


def _make_fire_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for x in range(256):
        h_deg = 85.0 * (x / 255.0)
        h = h_deg / 360.0
        s = 1.0
        l = min(1.0, x / 128.0)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        lut[x] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


def _make_doom_fire_lut() -> np.ndarray:
    key_colors = np.array([
        [0, 0, 0],
        [7, 7, 7],
        [31, 7, 7],
        [47, 15, 7],
        [71, 15, 7],
        [87, 23, 7],
        [103, 31, 7],
        [119, 31, 7],
        [143, 39, 7],
        [159, 47, 7],
        [175, 63, 7],
        [191, 71, 7],
        [199, 71, 7],
        [223, 79, 7],
        [223, 87, 7],
        [223, 87, 7],
        [215, 95, 7],
        [215, 95, 7],
        [215, 103, 15],
        [207, 111, 15],
        [207, 119, 15],
        [207, 127, 15],
        [207, 135, 23],
        [199, 135, 23],
        [199, 143, 23],
        [199, 151, 31],
        [191, 159, 31],
        [191, 159, 31],
        [191, 167, 39],
        [191, 167, 39],
        [191, 175, 47],
        [183, 175, 47],
        [183, 183, 47],
        [183, 183, 55],
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
    stops = [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ]
    return _make_lut_from_stops(stops)


def _make_inferno_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 0, 4)),
        (0.25, (87, 15, 109)),
        (0.50, (187, 55, 84)),
        (0.75, (249, 142, 8)),
        (1.0, (252, 255, 164)),
    ]
    return _make_lut_from_stops(stops)


def _make_ocean_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 5, 30)),
        (0.25, (0, 60, 125)),
        (0.50, (0, 140, 190)),
        (0.75, (0, 200, 175)),
        (1.0, (180, 245, 240)),
    ]
    return _make_lut_from_stops(stops)


def _make_cividis_lut() -> np.ndarray:
    stops = [
        (0.00, (0, 34, 77)),
        (0.25, (0, 68, 117)),
        (0.50, (60, 111, 130)),
        (0.75, (147, 147, 95)),
        (1.00, (250, 231, 33)),
    ]
    return _make_lut_from_stops(stops)


def _make_jet_lut() -> np.ndarray:
    stops = [
        (0.00, (0, 0, 131)),
        (0.35, (0, 255, 255)),
        (0.66, (255, 255, 0)),
        (1.00, (128, 0, 0)),
    ]
    return _make_lut_from_stops(stops)


def _make_coolwarm_lut() -> np.ndarray:
    stops = [
        (0.00, (59, 76, 192)),
        (0.25, (127, 150, 203)),
        (0.50, (217, 217, 217)),
        (0.75, (203, 132, 123)),
        (1.00, (180, 4, 38)),
    ]
    return _make_lut_from_stops(stops)


def _make_rdbu_lut() -> np.ndarray:
    stops = [
        (0.00, (103, 0, 31)),
        (0.25, (178, 24, 43)),
        (0.50, (247, 247, 247)),
        (0.75, (33, 102, 172)),
        (1.00, (5, 48, 97)),
    ]
    return _make_lut_from_stops(stops)


def _make_plasma_lut() -> np.ndarray:
    stops = [
        (0.0, (13, 8, 135)),
        (0.25, (126, 3, 167)),
        (0.50, (203, 71, 119)),
        (0.75, (248, 149, 64)),
        (1.0, (240, 249, 33)),
    ]
    return _make_lut_from_stops(stops)


def _make_magma_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 0, 4)),
        (0.25, (73, 18, 99)),
        (0.50, (150, 50, 98)),
        (0.75, (226, 102, 73)),
        (1.0, (252, 253, 191)),
    ]
    return _make_lut_from_stops(stops)


def _make_turbo_lut() -> np.ndarray:
    stops = [
        (0.0, (48, 18, 59)),
        (0.25, (31, 120, 180)),
        (0.50, (78, 181, 75)),
        (0.75, (241, 208, 29)),
        (1.0, (133, 32, 26)),
    ]
    return _make_lut_from_stops(stops)


GRAY_LUT = _make_gray_lut()
INFERNO_LUT = _make_inferno_lut()
OCEAN_LUT = _make_ocean_lut()
VIRIDIS_LUT = _make_viridis_lut()
PLASMA_LUT = _make_plasma_lut()
MAGMA_LUT = _make_magma_lut()
TURBO_LUT = _make_turbo_lut()
FIRE_LUT = _make_fire_lut()
DOOM_FIRE_LUT = _make_doom_fire_lut()
CIVIDIS_LUT = _make_cividis_lut()
JET_LUT = _make_jet_lut()
COOLWARM_LUT = _make_coolwarm_lut()
RDBU_LUT = _make_rdbu_lut()

COLOR_MAPS = {
    "Gray": GRAY_LUT,
    "Inferno": INFERNO_LUT,
    "Ocean": OCEAN_LUT,
    "Viridis": VIRIDIS_LUT,
    "Plasma": PLASMA_LUT,
    "Magma": MAGMA_LUT,
    "Turbo": TURBO_LUT,
    "Fire": FIRE_LUT,
    "Doom": DOOM_FIRE_LUT,
    "Cividis": CIVIDIS_LUT,
    "Jet": JET_LUT,
    "Coolwarm": COOLWARM_LUT,
    "RdBu": RDBU_LUT,
}

DEFAULT_CMAP_NAME = "Magma"


# -----------------------------------------------------------------------------
# OpenGL colormap widget
# -----------------------------------------------------------------------------
class GLColormapWidget(QOpenGLWidget):
    """
    Displays a single-channel uint8 frame texture with a 256x1 RGB LUT texture.
    Colormapping happens in the fragment shader.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # NOTE: don't type this as QOpenGLFunctions; some PyQt6 builds don't expose it.
        self._gl: Optional[Any] = None
        self._prog: Optional[QOpenGLShaderProgram] = None
        self._vao: Optional[QOpenGLVertexArrayObject] = None
        self._vbo: Optional[QOpenGLBuffer] = None

        self._tex_frame: int = 0
        self._tex_lut: int = 0

        self._frame_w: int = 0
        self._frame_h: int = 0

        self._pending_frame: Optional[np.ndarray] = None  # HxW uint8
        self._pending_lut: Optional[np.ndarray] = None    # 256x3 uint8

        self._have_textures: bool = False

    def set_frame(self, pixels_u8: np.ndarray) -> None:
        pix = np.asarray(pixels_u8, dtype=np.uint8)
        if pix.ndim != 2:
            return
        self._pending_frame = np.ascontiguousarray(pix)

        if self._have_textures:
            self.makeCurrent()
            self._upload_pending()
            self.doneCurrent()
        self.update()

    def set_lut(self, lut_rgb: np.ndarray) -> None:
        lut = np.asarray(lut_rgb, dtype=np.uint8)
        if lut.shape != (256, 3):
            return
        self._pending_lut = np.ascontiguousarray(lut)

        if self._have_textures:
            self.makeCurrent()
            self._upload_pending()
            self.doneCurrent()
        self.update()

    def initializeGL(self) -> None:
        # Grab the function table from the actual context.
        self._gl = self.context().functions()  # type: ignore[assignment]
        gl = self._gl

        gl.glDisable(GL_DEPTH_TEST)

        vs = """
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aUV;
        out vec2 vUV;
        void main() {
            vUV = aUV;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
        """

        fs = """
        #version 330 core
        in vec2 vUV;
        out vec4 fragColor;

        uniform sampler2D uFrame; // R8 normalized
        uniform sampler2D uLUT;   // 256x1 RGB

        void main() {
            float v = texture(uFrame, vUV).r; // 0..1
            float x = (v * 255.0 + 0.5) / 256.0; // sample center in LUT
            vec3 rgb = texture(uLUT, vec2(x, 0.5)).rgb;
            fragColor = vec4(rgb, 1.0);
        }
        """

        prog = QOpenGLShaderProgram()
        prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs)
        prog.link()
        self._prog = prog

        verts = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,

            -1.0, -1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ], dtype=np.float32)

        vao = QOpenGLVertexArrayObject()
        vao.create()
        vao.bind()
        self._vao = vao

        vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        vbo.create()
        vbo.bind()
        vbo.allocate(verts.tobytes(), verts.nbytes)
        self._vbo = vbo

        prog.bind()
        stride = 4 * 4  # bytes per vertex (4 floats)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, GL_FLOAT, False, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, GL_FLOAT, False, stride, ctypes.c_void_p(2 * 4))
        prog.release()

        vbo.release()
        vao.release()

        self._tex_frame = gl.glGenTextures(1)
        self._tex_lut = gl.glGenTextures(1)

        gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        gl.glBindTexture(GL_TEXTURE_2D, self._tex_frame)
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2801, GL_NEAREST)       # GL_TEXTURE_MIN_FILTER
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2800, GL_NEAREST)       # GL_TEXTURE_MAG_FILTER
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2802, GL_CLAMP_TO_EDGE) # GL_TEXTURE_WRAP_S
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2803, GL_CLAMP_TO_EDGE) # GL_TEXTURE_WRAP_T
        gl.glBindTexture(GL_TEXTURE_2D, 0)

        gl.glBindTexture(GL_TEXTURE_2D, self._tex_lut)
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2801, GL_NEAREST)       # GL_TEXTURE_MIN_FILTER
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2800, GL_NEAREST)       # GL_TEXTURE_MAG_FILTER
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2802, GL_CLAMP_TO_EDGE) # GL_TEXTURE_WRAP_S
        gl.glTexParameteri(GL_TEXTURE_2D, 0x2803, GL_CLAMP_TO_EDGE) # GL_TEXTURE_WRAP_T

        default_lut = COLOR_MAPS.get(DEFAULT_CMAP_NAME, GRAY_LUT)
        default_lut = np.ascontiguousarray(default_lut, dtype=np.uint8)
        gl.glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB8,
            256,
            1,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            default_lut.tobytes(),
        )
        gl.glBindTexture(GL_TEXTURE_2D, 0)

        self._have_textures = True
        self._upload_pending()

    def resizeGL(self, w: int, h: int) -> None:
        if self._gl is None:
            return
        self._gl.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        if self._gl is None or self._prog is None or self._vao is None:
            return

        gl = self._gl
        self._upload_pending()

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT)

        self._prog.bind()

        gl.glActiveTexture(GL_TEXTURE0)
        gl.glBindTexture(GL_TEXTURE_2D, self._tex_frame)
        self._prog.setUniformValue("uFrame", 0)

        gl.glActiveTexture(GL_TEXTURE1)
        gl.glBindTexture(GL_TEXTURE_2D, self._tex_lut)
        self._prog.setUniformValue("uLUT", 1)

        self._vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, 6)
        self._vao.release()

        gl.glBindTexture(GL_TEXTURE_2D, 0)
        gl.glActiveTexture(GL_TEXTURE0)
        gl.glBindTexture(GL_TEXTURE_2D, 0)

        self._prog.release()

    def _upload_pending(self) -> None:
        if self._gl is None:
            return
        gl = self._gl

        gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        if self._pending_lut is not None:
            lut = self._pending_lut
            self._pending_lut = None
            gl.glBindTexture(GL_TEXTURE_2D, self._tex_lut)
            gl.glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                256,
                1,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                lut.tobytes(),
            )
            gl.glBindTexture(GL_TEXTURE_2D, 0)

        if self._pending_frame is not None:
            pix = self._pending_frame
            self._pending_frame = None

            h, w = pix.shape
            gl.glBindTexture(GL_TEXTURE_2D, self._tex_frame)

            if (w != self._frame_w) or (h != self._frame_h):
                self._frame_w = w
                self._frame_h = h
                gl.glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_R8,
                    w,
                    h,
                    0,
                    GL_RED,
                    GL_UNSIGNED_BYTE,
                    pix.tobytes(),
                )
            else:
                gl.glTexSubImage2D(
                    GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    w,
                    h,
                    GL_RED,
                    GL_UNSIGNED_BYTE,
                    pix.tobytes(),
                )

            gl.glBindTexture(GL_TEXTURE_2D, 0)


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, sim: NumPyDnsSimulator) -> None:
        super().__init__()

        self.sim = sim
        self.current_cmap_name = DEFAULT_CMAP_NAME

        self.gl_view = GLColormapWidget()
        self.gl_view.setMinimumSize(1, 1)

        style = QApplication.style()

        self.start_button = QPushButton()
        self.start_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.start_button.setToolTip("Start simulation")
        self.start_button.setFixedSize(28, 28)
        self.start_button.setIconSize(QSize(14, 14))

        self.stop_button = QPushButton()
        self.stop_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.setToolTip("Stop simulation")
        self.stop_button.setFixedSize(28, 28)
        self.stop_button.setIconSize(QSize(14, 14))

        self.reset_button = QPushButton()
        self.reset_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.reset_button.setToolTip("Reset simulation")
        self.reset_button.setFixedSize(28, 28)
        self.reset_button.setIconSize(QSize(14, 14))

        self.save_button = QPushButton()
        self.save_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_button.setToolTip("Save current frame")
        self.save_button.setFixedSize(28, 28)
        self.save_button.setIconSize(QSize(14, 14))

        self.folder_button = QPushButton()
        self.folder_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self.folder_button.setToolTip("Save files")
        self.folder_button.setFixedSize(28, 28)
        self.folder_button.setIconSize(QSize(14, 14))

        self._status_update_counter = 0
        self._update_intervall = 2

        self.variable_combo = QComboBox()
        self.variable_combo.setToolTip("V: Variable")
        self.variable_combo.addItems(["U", "V", "K", "Ω", "φ"])

        self.n_combo = QComboBox()
        self.n_combo.setToolTip("N: Grid Size (N)")
        self.n_combo.addItems(["128", "192", "256", "384", "512", "768", "1024", "2048", "3072", "4096"])
        self.n_combo.setCurrentText(str(self.sim.N))

        self.re_combo = QComboBox()
        self.re_combo.setToolTip("R: Reynolds Number (Re)")
        self.re_combo.addItems(["1", "1000", "10000", "100000", "1E6", "1E9", "1E12", "1E15"])
        self.re_combo.setCurrentText(str(int(self.sim.re)))

        self.k0_combo = QComboBox()
        self.k0_combo.setToolTip("K: Initial energy peak wavenumber (K0)")
        self.k0_combo.addItems(["1", "5", "10", "15", "25", "35", "50"])
        self.k0_combo.setCurrentText(str(int(self.sim.k0)))

        self.cmap_combo = QComboBox()
        self.cmap_combo.setToolTip("C: Colormaps")
        self.cmap_combo.addItems(list(COLOR_MAPS.keys()))
        idx = self.cmap_combo.findText(DEFAULT_CMAP_NAME)
        if idx >= 0:
            self.cmap_combo.setCurrentIndex(idx)

        self.cfl_combo = QComboBox()
        self.cfl_combo.setToolTip("L: Controlling Δt (CFL)")
        self.cfl_combo.addItems(["0.05", "0.15", "0.25", "0.50", "0.75", "0.95"])
        self.cfl_combo.setCurrentText(str(self.sim.cfl))

        self.steps_combo = QComboBox()
        self.steps_combo.setToolTip("S: Max steps before reset/stop")
        self.steps_combo.addItems(["2000", "5000", "10000", "25000", "50000", "1E5", "1E6"])
        self.steps_combo.setCurrentText("5000")

        self.update_combo = QComboBox()
        self.update_combo.setToolTip("U: Update intervall")
        self.update_combo.addItems(["2", "5", "10", "50", "100", "1000"])
        self.update_combo.setCurrentText("2")

        self.auto_reset_checkbox = QCheckBox()
        self.auto_reset_checkbox.setToolTip("If checked, simulation auto-resets")
        self.auto_reset_checkbox.setChecked(True)

        self._build_layout()

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.status.setFont(mono)

        self.threads_label = QLabel(self)
        self.status.addPermanentWidget(self.threads_label)

        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._on_timer)  # type: ignore[attr-defined]

        self.start_button.clicked.connect(self.on_start_clicked)  # type: ignore[attr-defined]
        self.stop_button.clicked.connect(self.on_stop_clicked)  # type: ignore[attr-defined]
        self.reset_button.clicked.connect(self.on_reset_clicked)  # type: ignore[attr-defined]
        self.save_button.clicked.connect(self.on_save_clicked)  # type: ignore[attr-defined]
        self.folder_button.clicked.connect(self.on_folder_clicked)  # type: ignore[attr-defined]
        self.variable_combo.currentIndexChanged.connect(self.on_variable_changed)  # type: ignore[attr-defined]
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)  # type: ignore[attr-defined]
        self.n_combo.currentTextChanged.connect(self.on_n_changed)  # type: ignore[attr-defined]
        self.re_combo.currentTextChanged.connect(self.on_re_changed)  # type: ignore[attr-defined]
        self.k0_combo.currentTextChanged.connect(self.on_k0_changed)  # type: ignore[attr-defined]
        self.cfl_combo.currentTextChanged.connect(self.on_cfl_changed)  # type: ignore[attr-defined]
        self.steps_combo.currentTextChanged.connect(self.on_steps_changed)  # type: ignore[attr-defined]
        self.update_combo.currentTextChanged.connect(self.on_update_changed)  # type: ignore[attr-defined]

        import importlib.util

        title_backend = "(NumPy)"
        if importlib.util.find_spec("cupy") is not None:
            import cupy as cp
            try:
                props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_name = props["name"].decode(errors="replace")
                title_backend = f"(CuPy) {gpu_name}"
            except (RuntimeError, OSError, ValueError, IndexError):
                pass

        self.setWindowTitle(f"2D Turbulence {title_backend} © Mannetroll")

        dw, dh = self._display_size_for_N(self.sim.N)
        self.gl_view.setFixedSize(dw, dh)
        self.resize(dw + 40, dh + 120)

        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

        self.sim.set_variable(self.sim.VAR_OMEGA)
        self.variable_combo.setCurrentIndex(3)

        self._apply_current_lut()
        self._update_image(self.sim.get_frame_pixels())
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), None)

        self.on_start_clicked()

    @staticmethod
    def move_widgets(src_layout, dst_layout):
        while src_layout.count() > 0:
            item = src_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                dst_layout.addWidget(w)

    def _build_layout(self):
        old = self.centralWidget()
        if old is not None:
            old.setParent(None)

        central = QWidget()
        main = QVBoxLayout(central)
        main.addWidget(self.gl_view)

        row1 = QHBoxLayout()
        row1.addWidget(self.start_button)
        row1.addWidget(self.stop_button)
        row1.addWidget(self.reset_button)
        row1.addWidget(self.save_button)
        row1.addWidget(self.folder_button)
        row1.addWidget(self.cmap_combo)
        row1.addWidget(self.steps_combo)
        row1.addWidget(self.update_combo)
        row1.addWidget(self.auto_reset_checkbox)

        row2 = QHBoxLayout()
        row2.addWidget(self.variable_combo)
        row2.addWidget(self.n_combo)
        row2.addWidget(self.re_combo)
        row2.addWidget(self.k0_combo)
        row2.addWidget(self.cfl_combo)

        if self.sim.N >= 1024:
            single = QHBoxLayout()
            self.move_widgets(row1, single)
            self.move_widgets(row2, single)
            main.addLayout(single)
        else:
            main.addLayout(row1)
            main.addLayout(row2)

        self.setCentralWidget(central)

    def _display_size_for_N(self, N: int) -> Tuple[int, int]:
        if N < 768:
            scale = 1
        elif N <= 1024:
            scale = 2
        elif N <= 3072:
            scale = 4
        else:
            scale = 6
        d = max(1, int(N // scale))
        return (d, d)

    def _apply_current_lut(self) -> None:
        lut = COLOR_MAPS.get(self.current_cmap_name, GRAY_LUT)
        self.gl_view.set_lut(lut)

    def _update_run_buttons(self) -> None:
        running = self.timer.isActive()
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def on_start_clicked(self) -> None:
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        if not self.timer.isActive():
            self.timer.start()
        self._update_run_buttons()

    def on_stop_clicked(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self._update_run_buttons()

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

    def _get_full_field(self, variable: str) -> np.ndarray:
        S = self.sim.state

        if variable == "u":
            field = S.ur_full[0]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        if variable == "v":
            field = S.ur_full[1]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        if variable == "kinetic":
            dns_all.dns_kinetic(S)
            field = S.ur_full[2]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        if variable == "omega":
            dns_all.dns_om2_phys(S)
            field = S.ur_full[2]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        raise ValueError(f"Unknown variable: {variable}")

    def on_folder_clicked(self) -> None:
        N = self.sim.N
        Re = self.sim.re
        K0 = self.sim.k0
        CFL = self.sim.cfl
        STEPS = self.sim.get_iteration()

        folder = f"cupyturbo_{N}_{self.sci_no_plus(Re)}_{K0}_{CFL}_{STEPS}"

        desktop = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )

        dlg = QFileDialog(self)
        dlg.setWindowTitle(f"Case: {folder}")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setDirectory(desktop)

        for lineedit in dlg.findChildren(QLineEdit):
            lineedit.setText(".")  # type: ignore[attr-defined]

        if dlg.exec():
            base_dir = dlg.selectedFiles()[0]
        else:
            return

        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[SAVE] Dumping fields to folder: {folder_path}")
        self._dump_pgm_full(self._get_full_field("u"), os.path.join(folder_path, "u_velocity.pgm"))
        self._dump_pgm_full(self._get_full_field("v"), os.path.join(folder_path, "v_velocity.pgm"))
        self._dump_pgm_full(self._get_full_field("kinetic"), os.path.join(folder_path, "kinetic.pgm"))
        self._dump_pgm_full(self._get_full_field("omega"), os.path.join(folder_path, "omega.pgm"))
        print("[SAVE] Completed.")

    def on_save_clicked(self) -> None:
        var_name = self.variable_combo.currentText()
        cmap_name = self.cmap_combo.currentText()
        default_name = f"cupyturbo_{var_name}_{cmap_name}.png"

        desktop = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )
        initial_path = desktop + "/" + default_name

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save frame",
            initial_path,
            "PNG images (*.png);;All files (*)",
        )
        if path:
            img = self.gl_view.grabFramebuffer()
            img.save(path, "PNG")

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
            self._apply_current_lut()
            self._update_image(self.sim.get_frame_pixels())

    def on_n_changed(self, value: str) -> None:
        N = int(value)
        self.sim.set_N(N)

        dw, dh = self._display_size_for_N(N)
        self.gl_view.setFixedSize(dw, dh)

        self._update_image(self.sim.get_frame_pixels())

        new_w = dw + 40
        new_h = dh + 120
        print("Resize to:", new_w, new_h)

        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self.resize(new_w, new_h)

        screen = QApplication.primaryScreen().availableGeometry()
        g = self.geometry()
        g.moveCenter(screen.center())
        self.setGeometry(g)

        self._build_layout()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

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
        self.sim.cfl = float(value)
        self.sim.reset_field()
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        self._update_image(self.sim.get_frame_pixels())

    def on_steps_changed(self, value: str) -> None:
        self.sim.max_steps = int(float(value))

    def on_update_changed(self, value: str) -> None:
        self._update_intervall = int(float(value))

    def _on_timer(self) -> None:
        self.sim.step()
        self._status_update_counter += 1

        if self._status_update_counter >= self._update_intervall:
            pixels = self.sim.get_frame_pixels()
            self._update_image(pixels)

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

        if self.sim.get_iteration() >= self.sim.max_steps:
            if self.auto_reset_checkbox.isChecked():
                self.sim.reset_field()
                self._sim_start_time = time.time()
                self._sim_start_iter = self.sim.get_iteration()
            else:
                self.timer.stop()
                print("Max steps reached — simulation stopped (Auto-Reset OFF).")

    @staticmethod
    def _dump_pgm_full(arr: np.ndarray, filename: str):
        h, w = arr.shape
        minv = float(arr.min())
        maxv = float(arr.max())
        rng = maxv - minv

        with open(filename, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            if rng <= 1e-12:
                f.write(bytes([128]) * (w * h))
                return
            norm = (arr - minv) / rng
            pix = (1.0 + norm * 254.0).round().clip(1, 255).astype(np.uint8)
            f.write(pix.tobytes())

    def _update_image(self, pixels: np.ndarray) -> None:
        pixels = np.asarray(pixels, dtype=np.uint8)
        if pixels.ndim != 2:
            return
        self.gl_view.set_frame(pixels)

    def _update_status(self, t: float, it: int, fps: Optional[float]) -> None:
        fps_str = f"{fps:4.1f}" if fps is not None else " N/a"

        N = self.sim.N
        if N < 768:
            dpp = 100
        elif N <= 1024:
            dpp = 50
        elif N <= 3072:
            dpp = 25
        else:
            dpp = 17

        visc = float(self.sim.state.visc)
        txt = (
            f"FPS: {fps_str} | Iter: {it:5d} | T: {t:6.3f} "
            f"| DPP: {dpp}% | Visc: {visc:12.10f}"
        )
        self.status.showMessage(txt)

    def keyPressEvent(self, event) -> None:
        key = event.key()

        if key == Qt.Key.Key_V:
            idx = self.variable_combo.currentIndex()
            count = self.variable_combo.count()
            self.variable_combo.setCurrentIndex((idx + 1) % count)
            return

        if key == Qt.Key.Key_C:
            idx = self.cmap_combo.currentIndex()
            count = self.cmap_combo.count()
            self.cmap_combo.setCurrentIndex((idx + 1) % count)
            return

        if key == Qt.Key.Key_N:
            idx = self.n_combo.currentIndex()
            count = self.n_combo.count()
            self.n_combo.setCurrentIndex((idx + 1) % count)
            return

        if key == Qt.Key.Key_R:
            idx = self.re_combo.currentIndex()
            count = self.re_combo.count()
            self.re_combo.setCurrentIndex((idx + 1) % count)
            return

        if key == Qt.Key.Key_K:
            idx = self.k0_combo.currentIndex()
            count = self.k0_combo.count()
            self.k0_combo.setCurrentIndex((idx + 1) % count)
            return

        if key == Qt.Key.Key_L:
            idx = self.cfl_combo.currentIndex()
            count = self.cfl_combo.count()
            self.cfl_combo.setCurrentIndex((idx + 1) % count)
            return

        if key == Qt.Key.Key_S:
            idx = self.steps_combo.currentIndex()
            count = self.steps_combo.count()
            self.steps_combo.setCurrentIndex((idx + 1) % count)
            return

        if key == Qt.Key.Key_U:
            idx = self.update_combo.currentIndex()
            count = self.update_combo.count()
            self.update_combo.setCurrentIndex((idx + 1) % count)
            return

        super().keyPressEvent(event)


# -----------------------------------------------------------------------------
def main() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    sim = NumPyDnsSimulator(n=512)
    window = MainWindow(sim)

    screen = app.primaryScreen().availableGeometry()
    g = window.geometry()
    g.moveCenter(screen.center())
    window.setGeometry(g)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
