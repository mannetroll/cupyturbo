# dns_main_oglw.py
# (Option C: QOpenGLWidget + textures + LUT shader) — Win11Pro version
# Uses same OpenGL approach as the working dns_main_oglm.py:
#   - QOpenGLFunctions_4_1_Core
#   - explicit GL_* constants
#   - uploads via .tobytes() (no raw pointers / sip.voidptr)
#   - allocate textures immediately (never incomplete)
#   - keep last frame/LUT so context recreation doesn't go black

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import (
    QOpenGLShaderProgram,
    QOpenGLShader,
    QOpenGLBuffer,
    QOpenGLVertexArrayObject,
    QOpenGLFunctions_4_1_Core,
)

from cupyturbo.dns_wrapper import NumPyDnsSimulator
from cupyturbo.dns_simulator import check_cupy  # if you have it; otherwise remove

from dns_main_base import (
    COLOR_MAPS,
    DEFAULT_CMAP_NAME,
    GRAY_LUT,
    MainWindowBase,
)

# -----------------------------------------------------------------------------
# OpenGL constants (Qt function wrappers don't provide GL_* constants)
# -----------------------------------------------------------------------------
GL_DEPTH_TEST = 0x0B71
GL_COLOR_BUFFER_BIT = 0x00004000
GL_TRIANGLES = 0x0004
GL_FLOAT = 0x1406

GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE0 = 0x84C0
GL_TEXTURE1 = 0x84C1
GL_UNPACK_ALIGNMENT = 0x0CF5

GL_NEAREST = 0x2600
GL_CLAMP_TO_EDGE = 0x812F
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803

GL_TEXTURE_BASE_LEVEL = 0x813C
GL_TEXTURE_MAX_LEVEL = 0x813D

GL_R8 = 0x8229
GL_RGB8 = 0x8051
GL_RED = 0x1903
GL_RGB = 0x1907
GL_UNSIGNED_BYTE = 0x1401


class GLColormapWidget(QOpenGLWidget):
    """
    Displays a single-channel uint8 frame texture with a 256x1 RGB LUT texture.
    Colormapping happens in the fragment shader.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._gl: Optional[QOpenGLFunctions_4_1_Core] = None
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

        # Keep last valid data so context recreates don't go black
        self._last_frame: Optional[np.ndarray] = None      # HxW uint8
        self._last_lut: np.ndarray = np.ascontiguousarray(
            COLOR_MAPS.get(DEFAULT_CMAP_NAME, GRAY_LUT), dtype=np.uint8
        )

    def set_frame(self, pixels_u8: np.ndarray) -> None:
        pix = np.asarray(pixels_u8, dtype=np.uint8)
        if pix.ndim != 2:
            return
        pix_c = np.ascontiguousarray(pix)
        self._last_frame = pix_c
        self._pending_frame = pix_c
        if self._have_textures:
            self.makeCurrent()
            self._upload_pending()
            self.doneCurrent()
        self.update()

    def set_lut(self, lut_rgb: np.ndarray) -> None:
        lut = np.asarray(lut_rgb, dtype=np.uint8)
        if lut.shape != (256, 3):
            return
        lut_c = np.ascontiguousarray(lut)
        self._last_lut = lut_c
        self._pending_lut = lut_c
        if self._have_textures:
            self.makeCurrent()
            self._upload_pending()
            self.doneCurrent()
        self.update()

    def initializeGL(self) -> None:
        self._gl = QOpenGLFunctions_4_1_Core()
        self._gl.initializeOpenGLFunctions()
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
            float x = (v * 255.0 + 0.5) / 256.0; // sample center
            vec3 rgb = texture(uLUT, vec2(x, 0.5)).rgb;
            fragColor = vec4(rgb, 1.0);
        }
        """

        prog = QOpenGLShaderProgram()
        prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vs)
        prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fs)
        prog.link()
        self._prog = prog

        verts = np.array(
            [
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                 1.0,  1.0, 1.0, 1.0,

                -1.0, -1.0, 0.0, 0.0,
                 1.0,  1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 1.0,
            ],
            dtype=np.float32,
        )

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
        stride = 4 * 4  # 4 floats per vertex * 4 bytes

        # IMPORTANT: offsets are byte offsets. This matches the working oglm version.
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, GL_FLOAT, False, stride, 0)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, GL_FLOAT, False, stride, 8)

        prog.release()
        vbo.release()
        vao.release()

        self._tex_frame = gl.glGenTextures(1)
        self._tex_lut = gl.glGenTextures(1)

        gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        # ---- frame texture params ----
        gl.glBindTexture(GL_TEXTURE_2D, self._tex_frame)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)

        # Allocate something NOW so texture is never incomplete
        if self._last_frame is not None:
            h, w = self._last_frame.shape
            self._frame_w, self._frame_h = w, h
            gl.glTexImage2D(
                GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, self._last_frame.tobytes()
            )
        else:
            self._frame_w, self._frame_h = 1, 1
            gl.glTexImage2D(
                GL_TEXTURE_2D, 0, GL_R8, 1, 1, 0, GL_RED, GL_UNSIGNED_BYTE, bytes([0])
            )
        gl.glBindTexture(GL_TEXTURE_2D, 0)

        # ---- LUT texture params ----
        gl.glBindTexture(GL_TEXTURE_2D, self._tex_lut)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)

        gl.glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB8, 256, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, self._last_lut.tobytes()
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
                GL_TEXTURE_2D, 0, 0, 0, 256, 1, GL_RGB, GL_UNSIGNED_BYTE, lut.tobytes()
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
                    GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, pix.tobytes()
                )
            else:
                gl.glTexSubImage2D(
                    GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, pix.tobytes()
                )

            gl.glBindTexture(GL_TEXTURE_2D, 0)


# -----------------------------------------------------------------------------
# Window: reuse base and only provide the view widget
# -----------------------------------------------------------------------------
class MainWindow(MainWindowBase):
    def _create_view_widget(self) -> QOpenGLWidget:
        return GLColormapWidget()


def main() -> None:
    # Request a core profile context (4.1 core works on macOS, and you said this path works on Win11 too)
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)

    # Optional: your diagnostics
    try:
        check_cupy()
    except Exception:
        pass

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
