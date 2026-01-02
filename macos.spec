# macos.spec
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
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="cupyturbo",
)

app = BUNDLE(
    exe,
    name="cupyturbo.app",
    icon="cupyturbo/cupyturbo.icns",
    bundle_identifier="se.mannetroll.cupyturbo",  # adjust to your preferred ID
)