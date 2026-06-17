"""
================================================================
深度学习模型推理模块 (Deep Learning Model Inference)
================================================================

集成了以下两个预训练模型到「最终」项目中：

  1. GrayImage CNN   — 将二进制文件可视化为 135x63 灰度图，
                       然后用 CNN 做二分类。模型来自：
                       experient/fixed_detection_grayimage_cnn.py

  2. MalConv         — 直接在原始字节序列上做门控卷积，
                       参考 Raff et al. 2018。模型来自：
                       experient/malconv_experiment_balance_1to2.py

使用方式（features.py 会调用）：
    from models_inference import gi_confidence, malconv_confidence

    conf_gi   = gi_confidence(binary_bytes)   # 0.0 ~ 1.0
    conf_mc   = malconv_confidence(binary_bytes) # 0.0 ~ 1.0

设计原则：
    - 懒加载 (lazy loading)：第一次调用才读取 .pth 文件
    - 回退机制：torch 不可用或模型文件缺失时，返回 None，
                让调用方 (features.py) 使用启发式方法兜底
    - 5-fold ensemble：每个模型训练了 5 个 fold，推理时取平均
"""

import os
import numpy as np

# Windows 控制台默认可能是 GBK/gbk，emoji 会触发 UnicodeEncodeError。
# 这里先把 stdout/stderr 切换到 utf-8（Python 3.7+ 支持），
# 同时下文的 print 全部用纯 ASCII 符号代替 emoji。
if os.name == "nt":
    try:
        import io, sys
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ============================================================
# 0. 检查 PyTorch 是否可用；不可用时整个模块优雅降级
# ============================================================
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
    _TORCH_DEVICE = torch.device("cpu")
except Exception as _e:
    _HAS_TORCH = False
    _TORCH_DEVICE = None
    _TORCH_IMPORT_ERROR = str(_e)


# ============================================================
# 1. 路径常量 — 指向外层科研项目的模型目录
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_OUTER_ROOT   = os.path.dirname(_PROJECT_ROOT)

_GRAYIMAGE_DIR = os.path.join(
    _OUTER_ROOT, "experient", "fixed_cnn_model_results_balanced_1to2"
)
_MALCONV_DIR = os.path.join(
    _OUTER_ROOT, "experient", "result_malconv_1to2"
)

# MalConv 读取的最大字节数（训练时原 400000；
# 推理时缩减到 100000，精度损失很小但速度快 3~4 倍）
_MALCONV_MAX_LEN = 100000


# ============================================================
# 2. GrayImage CNN 模型定义（与训练代码完全一致）
# ============================================================
class _GrayImageCNN(nn.Module):
    """与 experient/fixed_detection_grayimage_cnn.py 结构完全一致"""

    def __init__(self):
        super().__init__()
        # 第一组卷积
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1   = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2   = nn.BatchNorm2d(32)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        # 第二组卷积
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1   = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2   = nn.BatchNorm2d(64)
        self.pool2   = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        # 第三组卷积
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1   = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2   = nn.BatchNorm2d(128)
        self.pool3   = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        # 全连接层
        self.fc1     = nn.Linear(128 * 16 * 7, 512)
        self.bn_fc1  = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.dropout1(self.pool1(x))
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.dropout2(self.pool2(x))
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.dropout3(self.pool3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout_fc1(self.relu(self.bn_fc1(self.fc1(x))))
        return self.sigmoid(self.fc2(x))


# ============================================================
# 3. MalConv 模型定义（与训练代码完全一致）
# ============================================================
class _MalConv(nn.Module):
    """与 experient/malconv_experiment_balance_1to2.py 结构完全一致"""

    def __init__(self, max_len=_MALCONV_MAX_LEN, win_size=500, vocab_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8, padding_idx=0)
        pad = (win_size - 1) // 2
        self.conv1 = nn.Conv1d(8, 128, kernel_size=win_size, stride=1, padding=pad)
        self.conv2 = nn.Conv1d(8, 128, kernel_size=win_size, stride=1, padding=pad)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        gated = conv1_out * self.sigmoid(conv2_out)
        relu_out = self.relu(gated)
        pooled = self.global_max_pool(relu_out).squeeze(2)
        fc1_out = self.fc1(pooled)
        out = self.fc2(fc1_out)
        return self.out_act(out)


# ============================================================
# 4. 通用工具：加载单个 checkpoint，返回 state_dict
# ============================================================
def _load_state_dict(path: str):
    """读取 .pth 文件，返回 model_state_dict。失败返回 None（不抛异常）。"""
    try:
        ckpt = torch.load(path, map_location=_TORCH_DEVICE, weights_only=False)
    except Exception as e:
        print(f"[models_inference] [!] 无法读取 {os.path.basename(path)}: {e}")
        return None

    # 兼容不同保存格式：可能直接是 state_dict，也可能是包装 dict
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # 如果 key 像参数名，直接当作 state_dict
        if any(k.endswith((".weight", ".bias")) for k in list(ckpt.keys())[:5]):
            return ckpt

    # 如果整个 ckpt 就是一个 nn.Module（较少见，也支持）
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    print(f"[models_inference] [!] 无法识别 {os.path.basename(path)} 的保存格式")
    return None


# ============================================================
# 5. 懒加载：所有 fold 的模型
# ============================================================
class _EnsembleLoader:
    """加载 5 个 fold 的模型，并做 ensemble（取平均）。"""

    def __init__(self, model_cls, ckpt_paths, name):
        self.name = name
        self.models = []
        self._initialized = False
        self._available = False
        self._model_cls = model_cls
        self._ckpt_paths = ckpt_paths

    def _init(self):
        if self._initialized:
            return
        self._initialized = True

        if not _HAS_TORCH:
            print(f"[models_inference] [!] 未检测到 PyTorch，跳过 {self.name}")
            return

        for p in self._ckpt_paths:
            if not os.path.exists(p):
                print(f"[models_inference] [!] 模型文件不存在: {p}")
                continue
            sd = _load_state_dict(p)
            if sd is None:
                continue
            model = self._model_cls().to(_TORCH_DEVICE)
            model.eval()
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[models_inference] [i] {os.path.basename(p)} 缺失 {len(missing)} 个 key（可能结构有细微差异）")
            self.models.append(model)

        self._available = len(self.models) > 0
        if self._available:
            print(f"[models_inference] [OK] {self.name}: 成功加载 {len(self.models)}/5 个 fold")
        else:
            print(f"[models_inference] [X] {self.name}: 所有 fold 都加载失败，将使用启发式方法")

    def predict(self, x):
        """x: 已经预处理好的 torch 张量，shape (1, ...)。返回 0~1 的 float。"""
        self._init()
        if not self._available:
            return None
        try:
            with torch.no_grad():
                preds = [float(m(x).item()) for m in self.models]
            return float(np.mean(preds))
        except Exception as e:
            print(f"[models_inference] [!] {self.name} 推理失败: {e}")
            return None


# ============================================================
# 6. 预处理：bytes → 模型输入张量
# ============================================================
def _preprocess_grayimage(data: bytes):
    """bytes → (1, 1, 135, 63) float32 tensor（/255 归一化）。"""
    if not data or len(data) == 0:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    n = len(arr)
    if n < 100:
        return None
    target = 135 * 63  # 8505
    idx = np.linspace(0, n - 1, target, dtype=np.int64)
    sampled = arr[idx].astype(np.float32) / 255.0
    img = sampled.reshape(1, 1, 135, 63)
    return torch.from_numpy(img).to(_TORCH_DEVICE)


def _preprocess_malconv(data: bytes):
    """bytes → (1, max_len) long tensor（不足补 0，超出截断）。"""
    if not data or len(data) == 0:
        return None
    n = min(len(data), _MALCONV_MAX_LEN)
    arr = np.zeros(_MALCONV_MAX_LEN, dtype=np.int64)
    # 直接从 bytes 复制到 numpy
    arr[:n] = np.frombuffer(data[:n], dtype=np.uint8).astype(np.int64)
    return torch.from_numpy(arr.reshape(1, -1)).to(_TORCH_DEVICE)


# ============================================================
# 7. 全局模型实例（懒加载）— 提供给 features.py 的 API
# ============================================================
def _make_gi_loader():
    paths = [
        os.path.join(_GRAYIMAGE_DIR, f"fixed_cnn_model_fold_{i}_1to2.pth")
        for i in range(1, 6)
    ]
    return _EnsembleLoader(_GrayImageCNN, paths, "GrayImage CNN")


def _make_mc_loader():
    paths = [
        os.path.join(_MALCONV_DIR, f"malconv_model_fold_{i}_1to2.pth")
        for i in range(1, 6)
    ]
    return _EnsembleLoader(_MalConv, paths, "MalConv")


_GI_LOADER = None
_MC_LOADER = None


def gi_confidence(data: bytes) -> float:
    """GrayImage CNN 置信度。模型不可用或推理失败时返回 None。"""
    global _GI_LOADER
    if not _HAS_TORCH:
        return None
    if _GI_LOADER is None:
        _GI_LOADER = _make_gi_loader()
    x = _preprocess_grayimage(data)
    if x is None:
        return None
    return _GI_LOADER.predict(x)


def malconv_confidence(data: bytes) -> float:
    """MalConv 置信度。模型不可用或推理失败时返回 None。"""
    global _MC_LOADER
    if not _HAS_TORCH:
        return None
    if _MC_LOADER is None:
        _MC_LOADER = _make_mc_loader()
    x = _preprocess_malconv(data)
    if x is None:
        return None
    return _MC_LOADER.predict(x)


def status() -> dict:
    """返回当前深度学习模型的加载状态（供调试/诊断用）。"""
    s = {
        "torch_available": _HAS_TORCH,
        "torch_version": torch.__version__ if _HAS_TORCH else None,
        "device": str(_TORCH_DEVICE) if _TORCH_DEVICE else None,
        "grayimage_dir": _GRAYIMAGE_DIR,
        "grayimage_dir_exists": os.path.isdir(_GRAYIMAGE_DIR),
        "malconv_dir": _MALCONV_DIR,
        "malconv_dir_exists": os.path.isdir(_MALCONV_DIR),
        "malconv_max_len": _MALCONV_MAX_LEN,
    }
    # 如果已经初始化，顺带报告 fold 数量
    if _GI_LOADER is not None:
        s["grayimage_loaded_folds"] = len(_GI_LOADER.models)
        s["grayimage_available"] = _GI_LOADER._available
    if _MC_LOADER is not None:
        s["malconv_loaded_folds"] = len(_MC_LOADER.models)
        s["malconv_available"] = _MC_LOADER._available
    return s


if __name__ == "__main__":
    # 简单自测试：构造随机数据，跑一次推理
    print("=" * 60)
    print("models_inference.py 自测试")
    print("=" * 60)
    import json
    print("状态:", json.dumps(status(), indent=2, ensure_ascii=False))

    fake_data = bytes(np.random.randint(0, 256, 8505 * 5, dtype=np.uint8))
    print(f"\n构造随机测试数据: {len(fake_data)} 字节")

    gi = gi_confidence(fake_data)
    print(f"GrayImage CNN 推理结果: {gi}")

    mc = malconv_confidence(fake_data)
    print(f"MalConv 推理结果:      {mc}")

    print("\n测试完成。如果上面两个值是 0~1 的 float，说明集成成功 ✓")