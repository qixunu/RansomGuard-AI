"""
PE 文件特征提取

6 个维度:
- CFG: 控制流图 (通过反汇编模拟得到控制流复杂度评分)
- EH : 字节熵直方图 (按固定窗口扫一遍)
- GI : 灰度图 (按 8505 字节重构为 135x63，算"纹理复杂度"作为概率)
- OP : 操作码频率 (通过 capstone / 字节级启发式估算)
- PF : PE 结构特征 (节区、导入表、资源等异常度)
- RB : 原始字节直方图 (256 维归一化，与 EH 类似但看值分布)

说明: 为保证产品"开箱即用"且不依赖重型反汇编器，这里采用的是 PE 启发式估计。
在有 pefile + 可选 capstone 的情况下，CFG/OP 可以做到更精确。
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pefile  # type: ignore
    HAVE_PEFILE = True
except Exception:
    HAVE_PEFILE = False

# 深度学习模型（可选）：
#  - 如果成功加载，GI/RB 维度会用 CNN/MalConv 做推理，准确率更高
#  - 如果没装 torch 或模型文件缺失，会自动回落到下面的启发式方法
try:
    from models_inference import gi_confidence, malconv_confidence, status as ml_status
    _HAVE_DL = True
except Exception as _e:
    _HAVE_DL = False
    _DL_IMPORT_ERROR = str(_e)

    def gi_confidence(data: bytes):
        return None

    def malconv_confidence(data: bytes):
        return None

    def ml_status():
        return {"torch_available": False, "error": _DL_IMPORT_ERROR}


# ===================== 通用工具 =====================

def _read_bytes(path: str) -> bytes:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    return p.read_bytes()


def _sigmoid(x: float) -> float:
    if x >= 35: return 1.0
    if x <= -35: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ===================== 1. RB: 字节值直方图 (MalConv) =====================

def extract_rb(data: bytes) -> float:
    """
    优先用 MalConv (在原始字节序列上的门控卷积模型) 给出置信度；
    如果没装 torch / 模型文件缺失，则回落到字节直方图启发式。
    输出: 0~1 之间的恶意倾向评分
    """
    # 1. 尝试深度学习模型
    conf = malconv_confidence(data) if _HAVE_DL else None
    if conf is not None:
        return float(conf)

    # 2. 回落：字节值分布启发式（原来的实现）
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(float)
    counts /= counts.sum() + 1e-9
    nonzero = counts[counts > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    norm = entropy / 8.0
    zero_ratio = float(counts[0])
    score = 0.7 * norm + 0.3 * (1.0 - min(1.0, zero_ratio * 3.0))
    return float(np.clip(score, 0.0, 1.0))


# ===================== 2. EH: 字节熵直方图（按滑窗） =====================

def extract_eh(data: bytes, window: int = 256) -> float:
    """
    滑动窗口的熵值分布。熵在 ~5-8 区间的比例越大 → 越可能加壳/加密。
    输出: 0~1 恶意倾向评分。
    """
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    if len(arr) < window:
        return extract_rb(data)
    n = (len(arr) // window) * window
    arr = arr[:n].reshape(-1, window)
    probs = arr.astype(np.float32) / 255.0
    # 用简单的"字节值方差"来近似每个窗口的复杂度
    std_per_window = np.std(arr.astype(np.float32), axis=1)
    mean_std = float(np.mean(std_per_window)) / 64.0  # 归一化
    # 熵
    flat = arr
    counts = np.bincount(flat.flatten(), minlength=256).astype(float)
    counts /= counts.sum() + 1e-9
    nonzero = counts[counts > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero))) / 8.0
    score = 0.6 * entropy + 0.4 * mean_std
    return float(np.clip(score, 0.0, 1.0))


# ===================== 3. GI: 灰度图 (CNN) =====================

def extract_gi(data: bytes) -> float:
    """
    优先用 GrayImage CNN (将二进制可视化为 135x63 灰度图后分类)；
    如果没装 torch / 模型文件缺失，则回落到像素梯度启发式。
    输出: 0~1
    """
    # 1. 尝试深度学习模型
    conf = gi_confidence(data) if _HAVE_DL else None
    if conf is not None:
        return float(conf)

    # 2. 回落：像素梯度启发式（原来的实现）
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    if len(arr) < 100:
        return 0.0
    target = 8505
    idx = np.linspace(0, len(arr) - 1, target, dtype=int)
    sampled = arr[idx].astype(np.float32)
    img = sampled.reshape(135, 63)
    gx = np.abs(np.diff(img, axis=1)).mean()
    gy = np.abs(np.diff(img, axis=0)).mean()
    complexity = (gx + gy) / (255.0 * 2.0)
    return float(np.clip(complexity * 1.8, 0.0, 1.0))


# ===================== 4. PF: PE 结构特征 =====================

_PE_SUSPICIOUS_SECTION_NAMES = {
    "upx", ".upx", "themida", "tengine", "theida",
    "enigma", "vmprotect", "vmp", "mpress",
    "aspack", "petite", "pecompact", "winlicense",
    "execryptor", "obsidium", "safengine", "shc",
}

_PE_SUSPICIOUS_IMPORTS = [
    "createprocess", "createfile", "writeprocessmemory",
    "virtualallocex", "openprocess", "cryptencrypt",
    "cryptdecrypt", "getprocaddress", "loadlibrary",
    "connectnamedpipe", "winexec", "createprocessasuser",
    "createprocesswithlogonw", "regsetvalueex",
    "internetopenurl", "internetconnect", "wsasocket",
    "send", "recv", "connect",
]


def extract_pf(data: bytes) -> float:
    """基于 PE 文件头/节区/导入表的启发式评分。"""
    score = 0.0
    reasons = []

    # 1. 是否为有效的 PE (MZ + PE\0\0 sig)
    is_pe = False
    try:
        if data[:2] == b"MZ":
            pe_off = struct.unpack_from("<I", data, 0x3C)[0]
            if 0 < pe_off < len(data) - 4 and data[pe_off:pe_off + 4] == b"PE\x00\x00":
                is_pe = True
    except Exception:
        pass

    if not is_pe:
        # 非 PE → 看字节特征做降级
        return float(np.clip((extract_rb(data) + extract_eh(data)) / 2.0 * 0.8, 0.0, 1.0))

    # 2. 使用 pefile（如果安装了）
    if HAVE_PEFILE:
        try:
            pe = pefile.PE(data=data, fast_load=True)
            pe.parse_data_directories(directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])

            # 节区名/属性扫描
            suspicious_sec = 0
            for sec in pe.sections:
                name = (sec.Name.rstrip(b"\x00").decode("ascii", "ignore")
                        .strip().lower())
                if any(s in name for s in _PE_SUSPICIOUS_SECTION_NAMES):
                    suspicious_sec += 1
                # 异常: 可写且可执行的节区 —— 典型加壳特征
                if (sec.Characteristics & 0x40000000) and (sec.Characteristics & 0x80000000):
                    suspicious_sec += 1
            if suspicious_sec:
                reasons.append(f"{suspicious_sec}个可疑/加壳节区")
                score += min(0.6, 0.12 * suspicious_sec)

            # 导入表扫描
            dangerous_imports = 0
            if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    for imp in entry.imports:
                        if imp.name:
                            n = imp.name.decode("ascii", "ignore").lower()
                            if any(s in n for s in _PE_SUSPICIOUS_IMPORTS):
                                dangerous_imports += 1
            reasons.append(f"{dangerous_imports}个可疑导入函数")
            score += min(0.4, 0.02 * dangerous_imports)

            pe.close()
        except Exception:
            # PE 解析失败本身就是"非标准 PE"的信号 → 加一点分数
            score += 0.15
            reasons.append("PE解析异常(非标准结构)")

    return float(np.clip(score, 0.0, 1.0))


# ===================== 5. CFG: 控制流图（启发式近似） =====================

def extract_cfg(data: bytes) -> float:
    """
    不做完整反汇编的 CFG 近似:
    - 统计常见的跳转字节 (0xE8 call, 0xE9 jmp, 0x7* jcc, 0xFF 间接call/jmp) 密度
    - 统计"ret" (0xC3) 数量 → 间接衡量函数数量
    - 与文件大小归一化
    这与"真实 IDA CFG"有差距，但在没有反汇编器的情况下给出合理估计。
    """
    if not data or len(data) < 512:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    total = float(len(arr))

    # 0xE8 call / 0xE9 jmp
    call_count = int(np.sum(arr == 0xE8))
    jmp_count = int(np.sum(arr == 0xE9))
    ret_count = int(np.sum(arr == 0xC3))
    # 0x70-0x7F 条件跳转
    jcc_count = int(np.sum((arr >= 0x70) & (arr <= 0x7F)))

    density_call = call_count / total * 1000.0
    density_jcc = jcc_count / total * 1000.0
    density_ret = ret_count / total * 1000.0

    # 启发式: 高密度 call/条件跳转 + 合理数量 ret → 复杂控制流
    score = 0.4 * _sigmoid((density_call - 1.5) * 2.0) + \
            0.35 * _sigmoid((density_jcc - 3.0) * 1.5) + \
            0.25 * _sigmoid((density_ret - 0.8) * 3.0)
    return float(np.clip(score, 0.0, 1.0))


# ===================== 6. OP: 操作码频率（启发式） =====================

_PATTERN = [
    # (签名字节, 是否为"可疑常见")
    # 这里用一个简化的"常见加密/循环相关指令字节"集合
    0x0F,  # 两字节指令前缀 (如 SIMD/加密)
    0xA4, 0xA5,  # movs (大量内存操作)
    0xAA, 0xAB,  # stos / lods
    0x81,  # sub/add with imm32 (循环计数常见)
    0xF3,  # rep prefix → rep movsb / rep stosb 大量内存操作
]


def extract_op(data: bytes) -> float:
    if not data or len(data) < 512:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    total = float(len(arr))
    counts = [int(np.sum(arr == b)) for b in _PATTERN]
    density = sum(counts) / total * 1000.0
    # 再加入熵的一小部分辅助
    entropy_boost = extract_eh(data) * 0.2
    s = _sigmoid((density - 5.0) * 0.8)
    return float(np.clip(s * 0.8 + entropy_boost, 0.0, 1.0))


# ===================== 主接口 =====================

def extract_all(filepath: str) -> Dict[str, float]:
    """提取一个 PE 文件的 6 维置信度 (0~1)"""
    data = _read_bytes(filepath)
    return {
        "CFG": extract_cfg(data),
        "EH":  extract_eh(data),
        "GI":  extract_gi(data),
        "OP":  extract_op(data),
        "PF":  extract_pf(data),
        "RB":  extract_rb(data),
    }


def extract_info(filepath: str) -> Dict[str, object]:
    """返回文件的基本元信息 + 6 维评分"""
    p = Path(filepath)
    data = _read_bytes(filepath)
    size_mb = len(data) / (1024.0 * 1024.0)
    scores = extract_all(filepath)
    is_pe = (data[:2] == b"MZ")
    return {
        "filename": p.name,
        "size_bytes": len(data),
        "size_mb": round(size_mb, 4),
        "is_pe": bool(is_pe),
        "scores": scores,
    }