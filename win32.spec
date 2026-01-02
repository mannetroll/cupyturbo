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

a = Analysis(
    ["cupyturbo/dns_main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
)

# ---- CuPy (GPU extra) payload ----
# These are commonly imported dynamically by CuPy at runtime.
a.hiddenimports += collect_submodules("cupy")
a.hiddenimports += collect_submodules("cupyx")
a.hiddenimports += collect_submodules("cupy_backends")
a.hiddenimports += [
    "cupy_backends.cuda._softlink",  # <- your missing module
]

# Pull in packaged DLL/PYD files that ship with the CuPy wheels.
a.binaries += collect_dynamic_libs("cupy")
a.binaries += collect_dynamic_libs("cupy_backends")

# Pull in CuPy's data payloads (wheels often include CUDA libs under package data dirs).
a.datas += collect_data_files("cupy", include_py_files=False)
a.datas += collect_data_files("cupy_backends", include_py_files=False)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cupyturbo",
    console=True,   # keep True while debugging; set False once it works
    icon="cupyturbo/cupyturbo.ico",  # optional
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="cupyturbo",
)