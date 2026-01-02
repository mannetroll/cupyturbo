# win32.spec
# Build (PowerShell):
#   Remove-Item -Recurse -Force build,dist -ErrorAction SilentlyContinue
#   uv sync --extra cuda
#   uv run pyinstaller win32.spec --noconfirm --clean
#
# Run:
#   .\dist\cupyturbo\cupyturbo.exe

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
)

# ---- CuPy (GPU extra) payload ----
cupy_hiddenimports = []
cupy_binaries = []
cupy_datas = []

# If CuPy is installed in this build environment, include its dynamic imports and DLL payload.
cupy_hiddenimports += collect_submodules("cupy")
cupy_hiddenimports += collect_submodules("cupyx")
cupy_hiddenimports += collect_submodules("cupy_backends")
cupy_hiddenimports += ["fastrlock"]

# Missing-at-runtime module you hit earlier:
cupy_hiddenimports += ["cupy_backends.cuda._softlink"]

# DLL/PYD payloads from wheels:
cupy_binaries += collect_dynamic_libs("cupy")
cupy_binaries += collect_dynamic_libs("cupy_backends")
cupy_binaries += collect_dynamic_libs("fastrlock")

# Package data that CuPy wheels sometimes use for runtime CUDA libs, etc.
# IMPORTANT: pass these into Analysis (don't append to a.datas later).
cupy_datas += collect_data_files("cupy", include_py_files=False)
cupy_datas += collect_data_files("cupy_backends", include_py_files=False)

a = Analysis(
    ["cupyturbo/dns_main.py"],
    pathex=["."],
    binaries=cupy_binaries,
    datas=cupy_datas,
    hiddenimports=cupy_hiddenimports,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cupyturbo",
    console=True,   # keep True until runtime is clean; then set False
    icon="cupyturbo/cupyturbo.ico",  # optional
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="cupyturbo",
)