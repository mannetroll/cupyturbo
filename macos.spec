# macos.spec
# Build:
#   rm -rf build dist
#   uv run pyinstaller macos.spec
#   ./dist/cupyturbo.app/Contents/MacOS/cupyturbo
#   open -n ./dist/cupyturbo.app
#

a = Analysis(
    ["cupyturbo/dns_main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="cupyturbo",
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="cupyturbo",
)

app = BUNDLE(
    coll,
    name="cupyturbo.app",
    icon="cupyturbo/cupyturbo.icns",
    bundle_identifier="se.mannetroll.cupyturbo",
)