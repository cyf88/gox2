"""路径工具：兼容源码运行与 PyInstaller 打包后的运行环境。

- app_dir：可写数据目录（配置文件、模型、人脸库）。打包后为可执行文件同级目录。
- resource_dir：只读打包资源目录（web 页面等）。打包后为 PyInstaller 解包目录。
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包产物中。"""
    return getattr(sys, "frozen", False)


def app_dir() -> Path:
    """返回可写数据目录。

    源码运行：项目根目录（go2x_stream 的上一级）。
    打包运行：可执行文件所在目录。
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_dir() -> Path:
    """返回只读资源目录。

    源码运行：项目根目录。
    打包运行：PyInstaller 的 _MEIPASS 解包目录。
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", str(app_dir())))
    return app_dir()
