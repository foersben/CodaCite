# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

block_cipher = None

# Base path of the repository
base_path = Path('.').absolute()

# Data files to bundle
datas = [
    ('app/static', 'app/static'),
    ('app/templates', 'app/templates'),
]

# Hidden imports that PyInstaller might miss
hidden_imports = [
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'pydantic_settings',
    'pydantic.deprecated.json',
    'fastapi.middleware.cors',
]

a = Analysis(
    ['app/main.py'],
    pathex=[str(base_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'openvino',
        'openvino_tokenizers',
        'openvino_telemetry',
        'optimum.intel.openvino',
        'tensorflow',
        'tensorboard',
        'torch.utils.tensorboard',
        'torch.distributed',
        'torch.testing',
        'unittest',
        'pytest',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide6',
        'matplotlib',
        'IPython',
        'notebook',
        'ipykernel',
    ],
    win_no_prefer_redirects=True,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out any OpenVINO shared libraries that Analysis may still have
# collected transitively. Their pre-existing code signatures cannot be
# replaced by PyInstaller's ad-hoc signing on macOS (codesign subsystem
# error), and the embedder already falls back gracefully without OpenVINO.
def _is_openvino_binary(dest: str, src: str) -> bool:
    """Return True if the binary belongs to the openvino package."""
    import posixpath
    parts_dest = dest.replace('\\', '/').split('/')
    parts_src = src.replace('\\', '/').split('/')
    openvino_names = {'openvino', 'openvino_tokenizers', 'openvino_telemetry'}
    return (
        parts_dest[0].lower() in openvino_names
        or any(p.lower() in openvino_names for p in parts_src)
    )

a.binaries = TOC([
    entry for entry in a.binaries
    if not _is_openvino_binary(entry[0], entry[1])
])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='codacite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
