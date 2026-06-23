#!/usr/bin/env python3
"""轻量级环境切换 — 零外部依赖，仅用标准库。

在导入 numpy/isaacgym 之前调用，确保执行环境正确。
"""

import os
import sys
from pathlib import Path

_REEXEC_ENV_VAR = "BATCH_REEXEC"


def maybe_reexec(target_python: str, target_ld_path: str) -> None:
    """若当前 Python 不是 target_python，则自动 execve 切换到目标环境。

    Args:
        target_python: 目标 Python 解释器的绝对路径。
        target_ld_path: 目标环境的 LD_LIBRARY_PATH 目录。
    """
    if os.environ.get(_REEXEC_ENV_VAR) == "1":
        return
    if sys.executable == target_python:
        return
    if not Path(target_python).exists():
        print(f"[reexec] WARNING: {target_python} not found, continuing in current env")
        return

    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{target_ld_path}:{ld}" if ld else target_ld_path
    env[_REEXEC_ENV_VAR] = "1"
    print(f"[reexec] Switching to {target_python}")
    os.execve(target_python, [target_python] + sys.argv, env)
