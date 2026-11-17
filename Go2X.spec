# -*- mode: python ; coding: utf-8 -*-
"""Go2X PyInstaller 打包配置（onedir 绿色免安装文件夹）。"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# ── 带原生库 / 数据文件的包：整包收集 ──────────────────────────
for pkg in (
    "onnxruntime",
    "cv2",
    "av",
    "insightface",
    "comtypes",
    "cyclonedds",
    "wasmtime",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# 某些包的动态导入子模块
for pkg in ("aiortc", "aioice", "unitree_webrtc_connect", "pydantic"):
    hiddenimports += collect_submodules(pkg)

# 语音对话用的本地 ASR（运行时才导入，需要把原生库一起收进来）
for pkg in ("faster_whisper", "ctranslate2", "tokenizers", "huggingface_hub"):
    try:
        d, b, h = collect_all(pkg)
    except Exception:
        continue
    datas += d
    binaries += b
    hiddenimports += h

# ── 项目资源文件 ──────────────────────────────────────────────
# web 页面（server.py 通过 resource_dir()/go2x_stream/web/index.html 读取）
datas += [("go2x_stream/web/index.html", "go2x_stream/web")]

# InsightFace 模型（运行时通过 resource_dir()/models 加载）
datas += [("models", "models")]

# insightface 在 frozen 模式下从 sys._MEIPASS/objects 读取 meanshape_68.pkl
import insightface
from pathlib import Path as _Path

_insightface_dir = _Path(insightface.__file__).parent
datas += [
    (str(_insightface_dir / "data" / "objects" / "meanshape_68.pkl"), "objects"),
]

# unitree lidar 解码用的 wasm 资源（lidar_decoder_libvoxel.py 按 __file__ 相对读取）
import unitree_webrtc_connect

_unitree_dir = _Path(unitree_webrtc_connect.__file__).parent
datas += [
    (str(_unitree_dir / "lidar" / "libvoxel.wasm"), "unitree_webrtc_connect/lidar"),
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Go2X",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Go2X",
)
