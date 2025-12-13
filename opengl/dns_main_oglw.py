# dns_main_oglw.py
# (Option C: QOpenGLWidget + textures + LUT shader) — Win11Pro version
import sys
from typing import Any, Optional

import numpy as np
from PyQt6 import sip

from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from PyQt6.QtOpenGL import (
    QOpenGLShaderProgram,
    QOpenGLShader,
    QOpenGLBuffer,
    QOpenGLVertexArrayObject,
)

from cupyturbo.dns_wrapper import NumPyDnsSimulator
from cupyturbo.dns_simulator import check_cupy  # if you have this helper; otherwise remove

from dns_main_base import (
    COLOR_MAPS,
    DEFAULT_CMAP_NAME,
    GRAY_LUT,
    MainWindowBase,
)


def _dbg(msg: str) -> None:
    print(msg, flush=True)


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
        _dbg("[GL] initializeGL: enter")

        ctx = self.context()
        if ctx is None:
            _dbg("[GL] initializeGL: NO CONTEXT")
            return

        self._gl = ctx.functions()  # type: ignore[assignment]
        gl = self._gl
        if gl is None:
            _dbg("[GL] initializeGL: NO FUNCTIONS")
            return

        gl.glDisable(gl.GL_DEPTH_TEST)

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
        stride = 4 * 4
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, stride, sip.voidptr(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, stride, sip.voidptr(8))
        prog.release()

        vbo.release()
        vao.release()

        self._tex_frame = gl.glGenTextures(1)
        self._tex_lut = gl.glGenTextures(1)

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_frame)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_lut)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        default_lut = np.ascontiguousarray(COLOR_MAPS.get(DEFAULT_CMAP_NAME, GRAY_LUT), dtype=np.uint8)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB8,
            256,
            1,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            sip.voidptr(int(default_lut.ctypes.data)),
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self._have_textures = True
        self._upload_pending()
        _dbg("[GL] initializeGL: done")

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
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self._prog.bind()

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_frame)
        self._prog.setUniformValue("uFrame", 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_lut)
        self._prog.setUniformValue("uLUT", 1)

        self._vao.bind()
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        self._vao.release()

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self._prog.release()

    def _upload_pending(self) -> None:
        if self._gl is None:
            return
        gl = self._gl

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        if self._pending_lut is not None:
            lut = self._pending_lut
            self._pending_lut = None
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_lut)
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D,
                0,
                0,
                0,
                256,
                1,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                sip.voidptr(int(lut.ctypes.data)),
            )
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        if self._pending_frame is not None:
            pix = self._pending_frame
            self._pending_frame = None

            h, w = pix.shape
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex_frame)

            if (w != self._frame_w) or (h != self._frame_h):
                self._frame_w = w
                self._frame_h = h
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,
                    gl.GL_R8,
                    w,
                    h,
                    0,
                    gl.GL_RED,
                    gl.GL_UNSIGNED_BYTE,
                    sip.voidptr(int(pix.ctypes.data)),
                )
            else:
                gl.glTexSubImage2D(
                    gl.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    w,
                    h,
                    gl.GL_RED,
                    gl.GL_UNSIGNED_BYTE,
                    sip.voidptr(int(pix.ctypes.data)),
                )

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


# -----------------------------------------------------------------------------
# Window: reuse base and only provide the view widget
# -----------------------------------------------------------------------------
class MainWindow(MainWindowBase):
    def _create_view_widget(self) -> QOpenGLWidget:
        return GLColormapWidget()


# -----------------------------------------------------------------------------
def main() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(0)
    fmt.setStencilBufferSize(0)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    _dbg("[APP] Creating QApplication...")
    app = QApplication(sys.argv)

    # Optional: keep your existing CuPy diagnostics
    try:
        check_cupy()  # prints what you showed earlier; if you don't have it, remove this call
    except Exception:
        pass

    sim = NumPyDnsSimulator()

    _dbg("[APP] Constructing MainWindow...")
    window = MainWindow(sim)
    _dbg("[APP] MainWindow constructed")

    screen = app.primaryScreen().availableGeometry()
    g = window.geometry()
    g.moveCenter(screen.center())
    window.setGeometry(g)

    _dbg("[APP] window.show() about to run...")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
