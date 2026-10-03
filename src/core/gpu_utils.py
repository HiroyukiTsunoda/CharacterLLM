"""
GPU VRAM 情報取得ユーティリティ。
NVML (NVIDIA Management Library) を ctypes 経由で呼び出す。
追加 Python パッケージ不要 — NVIDIA ドライバがインストールされていれば動作する。
"""

from __future__ import annotations

import ctypes
import logging
import platform
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    """1 枚の GPU の情報。"""
    index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    used_vram_bytes: int

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_bytes / (1024 ** 3)

    @property
    def free_vram_gb(self) -> float:
        return self.free_vram_bytes / (1024 ** 3)

    @property
    def used_vram_gb(self) -> float:
        return self.used_vram_bytes / (1024 ** 3)


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _load_nvml():
    """OS に応じて NVML 共有ライブラリをロードする。"""
    system = platform.system()
    if system == "Windows":
        candidates = ["nvml.dll", "nvml"]
    else:
        candidates = ["libnvidia-ml.so.1", "libnvidia-ml.so"]

    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def get_gpu_info_list() -> list[GpuInfo]:
    """検出された全 GPU の VRAM 情報を返す。取得失敗時は空リスト。"""
    nvml = _load_nvml()
    if nvml is None:
        logger.debug("NVML library not found — GPU info unavailable")
        return []

    try:
        if nvml.nvmlInit_v2() != 0:
            return []
    except Exception:
        return []

    gpus: list[GpuInfo] = []
    try:
        count = ctypes.c_uint(0)
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return gpus

        for i in range(count.value):
            handle = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle)) != 0:
                continue

            buf = ctypes.create_string_buffer(256)
            name = ""
            if nvml.nvmlDeviceGetName(handle, buf, 256) == 0:
                name = buf.value.decode("utf-8", errors="replace")

            mem = _NvmlMemory()
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
                continue

            gpus.append(GpuInfo(
                index=i,
                name=name,
                total_vram_bytes=mem.total,
                free_vram_bytes=mem.free,
                used_vram_bytes=mem.used,
            ))
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

    return gpus


def get_primary_gpu_info(gpu_index: int = 0) -> GpuInfo | None:
    """指定インデックスの GPU 情報を返す。取得不可なら None。"""
    gpus = get_gpu_info_list()
    for g in gpus:
        if g.index == gpu_index:
            return g
    return gpus[0] if gpus else None


# VRAM に載るかどうかの余裕度を判定
VRAM_OVERHEAD_GB = 1.0  # コンテキストバッファ等のオーバーヘッド見積り


def check_vram_fit(required_gb: float, free_gb: float) -> str:
    """
    VRAM 余裕度を判定する。

    Returns:
        "ok"      — 余裕あり (free > required + overhead)
        "tight"   — ギリギリ (free > required だが overhead 込みで不足)
        "over"    — 不足 (free < required)
        "unknown" — GPU 情報取得不可
    """
    if free_gb < 0:
        return "unknown"
    if free_gb >= required_gb + VRAM_OVERHEAD_GB:
        return "ok"
    if free_gb >= required_gb:
        return "tight"
    return "over"
