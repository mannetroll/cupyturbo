# win32.spec
# Build (PowerShell):
#   Remove-Item -Recurse -Force build,dist -ErrorAction SilentlyContinue
#   uv run pyinstaller win32.spec --noconfirm --clean
#
# Run:
#   .\dist\cupyturbo\cupyturbo.exe

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

a = Analysis(
    ["cupyturbo/dns_main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
)

# If you installed GPU extras (cupy-cuda13x), pull in CuPy modules + DLLs.
# This assumes cupy is present in the environment when building.
a.hiddenimports += collect_submodules("cupy")
a.hiddenimports += collect_submodules("cupyx")
a.binaries += collect_dynamic_libs("cupy")

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cupyturbo",
    console=False,                 # GUI app; set True if you want a console window
    icon="cupyturbo/cupyturbo.ico" # <-- make an .ico for Windows
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="cupyturbo",
)