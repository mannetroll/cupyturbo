# win32.spec
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

import os
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
    collect_data_files,
)

def _filter_out_cuda_runtime_dlls(binaries):
    """
    Remove large CUDA runtime DLLs that CuPy wheels often bundle, so we rely on
    system-installed CUDA instead. This dramatically reduces dist size.
    """
    # Case-insensitive prefix match against DLL basename.
    exclude_prefixes = (
        "cublas",      # cublas64_*.dll, cublasLt64_*.dll
        "cufft",       # cufft64_*.dll
        "curand",      # curand64_*.dll
        "cusolver",    # cusolver64_*.dll
        "cusparse",    # cusparse64_*.dll
        "nvrtc",       # nvrtc64_*.dll
        "cudart",      # cudart64_*.dll (small but still part of CUDA runtime)
        "cudnn",       # cudnn*.dll (if present)
        "cutensor",    # cutensor*.dll (if present)
        "nccl",        # nccl*.dll (if present)
        "nvjitlink",   # nvJitLink*.dll (if present)
    )

    out = []
    for entry in binaries:
        # PyInstaller binaries entries are typically: (dest_name, src_name, typecode)
        dest_name, src_name, typecode = entry
        base = os.path.basename(src_name).lower()
        if base.endswith(".dll") and base.startswith(exclude_prefixes):
            continue
        out.append(entry)
    return out


# ---- CuPy (GPU extra) payload ----
cupy_hiddenimports = []
cupy_binaries = []
cupy_datas = []

# CuPy dynamic imports
cupy_hiddenimports += collect_submodules("cupy")
cupy_hiddenimports += collect_submodules("cupyx")
cupy_hiddenimports += collect_submodules("cupy_backends")
cupy_hiddenimports += ["cupy_backends.cuda._softlink"]

# fastrlock (CuPy dependency)
cupy_hiddenimports += collect_submodules("fastrlock")
cupy_hiddenimports += ["fastrlock.rlock"]
cupy_datas += collect_data_files("fastrlock", include_py_files=False)
cupy_binaries += collect_dynamic_libs("fastrlock")

# CuPy binaries (wheel payload)
cupy_binaries += collect_dynamic_libs("cupy")
cupy_binaries += collect_dynamic_libs("cupy_backends")

# Drop bundled CUDA runtime DLLs to shrink distribution
cupy_binaries = _filter_out_cuda_runtime_dlls(cupy_binaries)

# CuPy data payloads
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
    console=True,   # set False when you’re done debugging
    icon="cupyturbo/cupyturbo.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="cupyturbo",
)