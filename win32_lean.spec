# win32_lean.spec
# Build (PowerShell):
#   Remove-Item -Recurse -Force build,dist -ErrorAction SilentlyContinue
#   uv sync --extra cuda
#   uv run pyinstaller win32_lean.spec --noconfirm --clean
#
# Run:
#   .\dist\cupyturbo\cupyturbo.exe
#
# NOTE (runtime requirement):
#   The target machine must have a CUDA runtime/toolkit installed and discoverable via PATH
#   (typically %CUDA_PATH%\bin). Otherwise CuPy will fail to import.
# win32_lean.spec (lean CUDA: rely on system CUDA DLLs)

import os
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
)

EXCLUDE_DLL_PREFIXES = (
    "cublas", "cufft", "curand", "cusolver", "cusparse",
    "nvrtc", "cudart", "cudnn", "cutensor", "nccl", "nvjitlink",
)

def _is_excluded_dll_name(name: str) -> bool:
    base = os.path.basename(name).lower()
    return base.endswith(".dll") and base.startswith(EXCLUDE_DLL_PREFIXES)

def _filter_toc_3(toc3):
    # Entries: (dest_name, src_name, typecode)
    out = []
    for dest, src, typ in toc3:
        if _is_excluded_dll_name(src) or _is_excluded_dll_name(dest):
            continue
        out.append((dest, src, typ))
    return out

def _filter_datas_2(datas2):
    # Entries: (src, dest)
    out = []
    for src, dest in datas2:
        if _is_excluded_dll_name(src) or _is_excluded_dll_name(dest):
            continue
        out.append((src, dest))
    return out


# ---- CuPy (GPU extra) payload ----
cupy_hiddenimports = []
cupy_binaries = []
cupy_datas = []

cupy_hiddenimports += collect_submodules("cupy")
cupy_hiddenimports += collect_submodules("cupyx")
cupy_hiddenimports += collect_submodules("cupy_backends")
cupy_hiddenimports += ["cupy_backends.cuda._softlink"]

# fastrlock
cupy_hiddenimports += collect_submodules("fastrlock")
cupy_hiddenimports += ["fastrlock.rlock"]
cupy_datas += collect_data_files("fastrlock", include_py_files=False)
cupy_binaries += collect_dynamic_libs("fastrlock")

# CuPy binaries from wheels (we'll still filter later, but keep these for .pyd etc.)
cupy_binaries += collect_dynamic_libs("cupy")
cupy_binaries += collect_dynamic_libs("cupy_backends")

# CuPy data payloads (filter out any DLLs that sneak in as "data")
cupy_datas += collect_data_files("cupy", include_py_files=False)
cupy_datas += collect_data_files("cupy_backends", include_py_files=False)
cupy_datas = _filter_datas_2(cupy_datas)

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
    console=True,
    icon="cupyturbo/cupyturbo.ico",
)

# CRITICAL: filter the *final* TOCs (these include auto-collected DLL deps)
filtered_binaries = _filter_toc_3(a.binaries)
filtered_datas = _filter_toc_3(a.datas)

coll = COLLECT(
    exe,
    filtered_binaries,
    a.zipfiles,
    filtered_datas,
    name="cupyturbo",
)