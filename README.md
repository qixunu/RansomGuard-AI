# RansomGuard AI · 基于多模型融合与 AI 大模型的勒索软件检测系统

> 提供 Web UI / CLI / REST API 三种使用方式，支持离线规则引擎与在线大模型推理，快速识别 PE 文件中的勒索软件威胁。

- **仓库地址**：https://github.com/qixunu/RansomGuard-AI
- **作者**：qixunu
- **邮箱**：qiwang.xun@nuaa.edu.cn

## 项目简介

RansomGuard AI 是一款面向科研与教学用途的勒索软件检测产品。它从 **6 个独立维度** 对 PE 文件进行特征提取，包括控制流图（CFG）、字节熵直方图（EH）、灰度图纹理（GI）、操作码频率（OP）、PE 结构特征（PF）、原始字节分布（RB）。六个维度产出 0~1 的恶意倾向评分后，系统通过 **规则快速通道 + 加权融合 + LLM 专家推理** 的三级决策流水线给出最终判决，同时提供风险等级、推理过程、误报风险评估和处置建议，实现检测结果的高度可解释性。

项目的核心特点：

- **多模型并行架构**：六个相互独立的特征提取器并行工作，任何单一维度被攻击者绕过都不影响整体判断；
- **双重运行模式**：离线模式（本地 Mock 专家系统，无需网络和 API Key）与在线模式（接入真实大模型）并存；
- **优先控制误报**：LLM 推理明确要求"只有当恶意解释显著比良性解释更可信时才判定为恶意"；
- **优雅降级机制**：任何 LLM 调用失败自动回落到本地规则引擎 + 加权融合，检测永不中断；
- **完整产品化封装**：提供 Web UI、命令行工具、REST API 三种使用方式。

## 环境与依赖

### 运行环境

| 项目     | 版本                                               | 说明                                                         |
| -------- | -------------------------------------------------- | ------------------------------------------------------------ |
| 操作系统 | Windows 10 / Windows 11 / macOS 12+ / Ubuntu 20.04 | 开发与测试主要在 Windows 平台完成                            |
| Python   | 3.10.x 及以上                                      | 推荐 3.10 或 3.11，3.8 以下未测试                            |
| GPU      | 无 / NVIDIA RTX 3060+                              | 无 GPU 可使用启发式模式；有 GPU 可启用 CNN/MalConv 深度学习模型以获得更高精度 |
| 浏览器   | Chrome / Edge / Firefox 最新版本                   | Web UI 推荐使用 Chromium 内核浏览器                          |

### 开源程序与第三方依赖

> 本项目无需安装 MySQL、Redis 等外部服务，以下列出需单独安装的基础工具：

| 依赖名称              | 使用版本 | 下载链接                          | 安装方式           | 说明                          |
| --------------------- | -------- | --------------------------------- | ------------------ | ----------------------------- |
| Python                | 3.10.x   | https://www.python.org/downloads/ | 官方安装包         | 运行时环境                    |
| Git                   | 2.40+    | https://git-scm.com/downloads     | 官方安装包         | 代码版本管理（可选）          |
| OpenSSL（如需 HTTPS） | 3.0+     | https://www.openssl.org/          | 系统自带或源码编译 | 启用 HTTPS 部署时需要（可选） |

> **注意**：本项目核心逻辑完全使用 Python 实现，无外部数据库/缓存/消息队列依赖，开箱即用。

### Python 依赖

依赖清单文件：`requirements.txt`（已包含在本仓库中）

| 包名         | 最低版本 | 用途                                  |
| ------------ | -------- | ------------------------------------- |
| Flask        | 3.0.0    | Web UI 后端框架与 REST API 服务       |
| pandas       | 2.0.0    | CSV 数据读取、批量检测结果处理        |
| numpy        | 1.24.0   | 数值计算、矩阵运算、统计分析          |
| scikit-learn | 1.3.0    | 提供部分归一化与评估指标工具函数      |
| pefile       | 2023.2.7 | PE 文件结构解析，用于 PF 维度特征提取 |
| requests     | 2.31.0   | LLM API 调用（OpenAI/DeepSeek 等）    |
| tqdm         | 4.66.0   | 批量检测时的进度条显示                |

额外可选依赖（手动安装即可启用对应功能）：

| 包名   | 推荐版本 | 用途                                                      |
| ------ | -------- | --------------------------------------------------------- |
| torch  | 2.0+     | 启用 CNN / MalConv 深度学习推理（GI / RB 维度高精度模式） |
| openai | 1.10.0+  | 使用 OpenAI 官方 SDK 调用大模型（已内置兼容实现，可省略） |

安装命令：

```bash
# 基础依赖（必需）
pip install -r requirements.txt

# 完整依赖（可选，启用深度学习推理）
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 如需 GPU 版本请根据 CUDA 版本选择对应命令
```

> 国内用户建议使用镜像源加速：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

## 配置说明

### LLM 配置（可选）

修改 `config.json` 或在运行时通过命令行参数 / Web UI 配置 LLM 提供方：

```json
{
  "product_name": "RansomGuard AI",
  "product_version": "1.0.0",
  "description": "基于多模型融合 + AI大语言模型的勒索软件检测产品",
  "detection_threshold": 0.5,
  "models": {
    "CFG": {"description": "控制流图 (Control Flow Graph)", "weight": 1.0},
    "EH":  {"description": "字节熵直方图 (Entropy Histogram)", "weight": 1.0},
    "GI":  {"description": "灰度图特征 (Gray Image)", "weight": 1.0},
    "OP":  {"description": "操作码频率 (Opcode)", "weight": 1.0},
    "PF":  {"description": "PE 结构特征 (PE Feature)", "weight": 1.0},
    "RB":  {"description": "原始字节分布 (Raw Bytes)", "weight": 1.0}
  },
  "llm": {
    "enabled": true,
    "provider": "mock",
    "use_fallback_when_api_unavailable": true,
    "max_retries": 1,
    "retry_delay_seconds": 1
  },
  "risk_levels": {
    "low":    [0.0, 0.33],
    "medium": [0.33, 0.67],
    "high":   [0.67, 1.0]
  },
  "data": {
    "sample_confidence_csv": "data/sample_confidence.csv",
    "wild_label_csv": "data/sample_wild_label.csv"
  }
}
```

| provider 可选值 | 说明                                                   |
| --------------- | ------------------------------------------------------ |
| `mock`          | 默认值，使用本地多层专家系统推理，无需网络和 API Key   |
| `openai`        | 使用 OpenAI 官方 API（需配置 API Key）                 |
| `deepseek`      | 使用 DeepSeek API（需配置 API Key）                    |
| `anthropic`     | 使用 Anthropic Claude API（需配置 API Key）            |
| `dashscope`     | 使用阿里云百炼/通义千问 API（需配置 API Key）          |
| `local_http`    | 使用本地兼容 OpenAI 协议的推理服务（如 Ollama / vLLM） |

> **安全提示**：敏感配置（API Key、密钥）请通过环境变量 `RANSOMGUARD_API_KEY` 配置，或在 Web UI 中临时填写，**请勿将真实 Key 硬编码到代码或提交到 Git 仓库**。`config.json` 已做好敏感字段分离设计，默认不包含任何 Key。

### 其他关键配置

| 配置项                       | 默认值                       | 说明                                                  | 配置文件路径         |
| ---------------------------- | ---------------------------- | ----------------------------------------------------- | -------------------- |
| `product_name`               | "RansomGuard AI"             | 产品名称，用于标题显示                                | `config.json`        |
| `product_version`            | "1.0.0"                      | 产品版本号                                            | `config.json`        |
| `detection_threshold`        | 0.5                          | 检测阈值，ensemble_probability 超过此值判定为勒索软件 | `config.json`        |
| `models.*.weight`            | 1.0                          | 各模型权重（加权融合时使用，默认等权）                | `config.json`        |
| `risk_levels.low`            | [0.0, 0.33]                  | 低风险概率区间                                        | `config.json`        |
| `risk_levels.medium`         | [0.33, 0.67]                 | 中等风险概率区间                                      | `config.json`        |
| `risk_levels.high`           | [0.67, 1.0]                  | 高风险概率区间                                        | `config.json`        |
| `data.sample_confidence_csv` | "data/sample_confidence.csv" | 示例置信度数据文件路径                                | `config.json`        |
| `data.wild_label_csv`        | "data/sample_wild_label.csv" | 示例数据标签文件路径                                  | `config.json`        |
| `SERVER_HOST`                | 127.0.0.1                    | Web UI 监听地址（通过 `--host` 参数调整）             | `app.py` / `main.py` |
| `SERVER_PORT`                | 5000                         | Web UI 服务端口（通过 `--port` 参数调整）             | `app.py` / `main.py` |
| `MAX_FILE_SIZE`              | 512MB                        | 上传文件大小上限                                      | `app.py`             |
| `DEBUG_MODE`                 | false                        | 是否开启 Flask debug 模式（仅开发环境使用）           | `app.py`             |

## 数据集

### 数据集说明

本项目使用两类数据：**6 维模型置信度数据**（用于批量检测和演示）与 **可选原始 PE 样本**（用于单文件实时检测）。

| 数据集名称                             | 来源                                      | 大小                | 格式 | 说明                                                 |
| -------------------------------------- | ----------------------------------------- | ------------------- | ---- | ---------------------------------------------------- |
| sample_confidence.csv                  | 项目内置（从原始科研数据派生）            | 约 200KB（1000 条） | CSV  | 示例置信度数据，用于批量检测和 Web UI 演示           |
| sample_wild_label.csv                  | 项目内置（与 sample_confidence.csv 对应） | 约 30KB             | CSV  | 示例数据的真实标签（0=良性，1=勒索软件），仅用于评估 |
| crypto_confidence/wild/merged_wild.csv | 原始科研数据集（不纳入版本控制）          | 约 1-5MB            | CSV  | 完整科研数据集，包含更多样本及多维度置信度           |

示例数据字段：

```
hash, CFG, EH, GI, OP, PF, RB, label
```

- `hash`: 文件 SHA256 哈希，作为样本唯一标识
- `CFG`~`RB`: 6 个维度的恶意评分（0.0 = 完全良性，1.0 = 极度恶意）
- `label`: 真实标签（0 = 良性软件，1 = 勒索软件），**不参与检测过程，仅用于评估**

> **体积较大的完整数据集不纳入 Git 仓库**，需从原始科研项目目录（`../crypto_confidence/wild/`）查找或由项目成员分享。
>
> **小部分数据示例已提交到 Git 仓库**（位于 `data/sample_confidence.csv`），用于：
>
> - 让其他开发者无需下载完整数据集即可快速了解数据格式与字段含义
> - 支撑 Web UI 演示和本地功能测试的最小可运行数据
> - 作为批量检测功能的输入示例，方便 Code Review 时对照理解逻辑
>
> 示例数据要求：
>
> - 条数控制在 1000 条（兼顾演示效果与文件大小）
> - 文件大小约 200KB，**不超过 10MB**
> - 已脱敏处理，仅包含文件哈希、模型评分和标签，不包含任何原始代码或隐私信息
> - 文件命名建议：`sample_confidence.csv` / `sample_wild_label.csv`

### 数据集下载与放置

```bash
# 1. 进入项目目录


# 2. 初始化示例数据（自动完成以下操作）
#    - 尝试从原始科研项目目录查找完整数据
#    - 如果找到，自动截取前 1000 条并复制到 data/
#    - 如果找不到，自动生成同等规模的合成示例数据（保证离线演示可用）
python main.py init-data

# 3. 确认数据文件存在
#    data/sample_confidence.csv ← 已自动生成
#    data/sample_wild_label.csv ← 已自动生成
```

数据集目录结构：

```
最终/
├── data/                       # ✅ 示例数据（已通过 init-data 自动生成）
│   ├── sample_confidence.csv   # ✅ 1000 条示例置信度数据，已提交到仓库
│   └── sample_wild_label.csv    # ✅ 对应真实标签，已提交到仓库
├── ../crypto_confidence/       # 原始科研数据集（父目录中，不纳入版本控制）
│   └── wild/
│       ├── merged_wild.csv     # 完整数据集（init-data 的优先数据源）
│       └── ...
├── app.py                      # Flask 后端（Web UI + API，~260 行）
├── main.py                     # CLI 命令行入口（init-data/scan/batch/serve，~230 行）
├── detector.py                 # 核心检测引擎（规则引擎 + 加权融合 + LLM 集成，~410 行）
├── features.py                 # 6 维特征提取模块（CFG/EH/GI/OP/PF/RB，~310 行）
├── llm_engine.py               # LLM 推理引擎封装（Mock/OpenAI/DeepSeek 等，~500 行）
├── models_inference.py         # 深度学习模型推理（CNN/MalConv，可选模块）
├── config.json                 # 产品配置文件（产品名/版本/阈值/权重/LLM 设置）
├── requirements.txt            # Python 依赖清单（7 个核心包）
├── run.bat                     # Windows 一键启动脚本
├── prompt.txt                  # LLM 提示词模板（用于专家推理）
├── templates/
│   └── index.html              # Web UI 前端页面（单页应用）
├── static/
│   ├── style.css               # 前端样式（自定义样式表）
│   └── app.js                  # 前端逻辑（文件上传、API 调用、结果渲染）
└── README.md                   # 本文件：项目说明文档
```

> `data/` 目录中的示例文件已包含在仓库中。如果需要使用完整的原始科研数据集（5000+ 条样本），可前往virusshare中下载样本自行标注，或联系作者获取。

## 快速开始

```bash
# 1. 克隆或获取项目代码（如果使用 Git）
git clone https://github.com/yourname/ransomguard-ai.git
cd ransomguard-ai

# 2. 安装 Python 依赖
pip install -r requirements.txt
# 国内用户建议：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 3. 初始化示例数据（可选，首次启动 serve 时会自动检查）
python main.py init-data

# 4a. 启动 Web UI（推荐）
python main.py serve
# 然后浏览器访问 http://127.0.0.1:5000
# 如需自定义端口: python main.py serve --host 0.0.0.0 --port 8080

# 4b. 或使用命令行扫描单个文件
python main.py scan path/to/sample.exe
# 使用 DeepSeek 大模型:
python main.py scan sample.exe --provider deepseek --api-key sk-your-key-here

# 4c. 或使用命令行批量检测
python main.py batch data/sample_confidence.csv --limit 100

# 4d. 或使用 REST API 直接调用
# POST /api/scan    → 上传 PE 文件
# POST /api/batch   → 上传 CSV 批量检测
# GET  /api/demo    → 内置示例数据演示
```

### Windows 一键启动

```bat
:: 直接双击 run.bat 即可自动安装依赖并启动 Web UI
run.bat
```

启动后浏览器将自动打开 `http://127.0.0.1:5000`，可在页面上：

1. 上传单个 PE 文件进行实时检测
2. 上传 CSV 进行批量检测（需包含 `hash, CFG, EH, GI, OP, PF, RB` 列）
3. 一键加载内置示例数据进行演示（1000 条样本）
4. 在 Web UI 中切换 LLM 提供方并配置 API Key
5. 下载批量检测的 CSV 结果报告

### 命令行参数说明

```bash
# 扫描单个文件 — 可用参数
python main.py scan <file_path> \
    [--no-llm]                    # 禁用 LLM，仅使用本地规则和加权融合（更快、离线可用）
    [--provider <provider>]       # LLM 提供方: mock/openai/deepseek/anthropic/dashscope/local_http
    [--api-key <key>]             # LLM API Key
    [--base-url <url>]            # 自定义 Base URL
    [--model <model_name>]        # 自定义模型名（如 gpt-4o-mini / deepseek-chat）

# 批量检测 — 可用参数
python main.py batch <csv_path> \
    [--limit <number>]            # 处理行数上限（默认 1000）
    [--output <output_csv>]       # 输出 CSV 路径（默认自动命名）
    [--no-llm / --provider / --api-key / --base-url / --model]  # 同上 LLM 配置

# 启动 Web UI — 可用参数
python main.py serve \
    [--host <ip_address>]         # 监听地址（默认 127.0.0.1）
    [--port <port>]               # 监听端口（默认 5000）
    [--debug]                     # 开启 Flask debug 模式（开发用）
```

## 项目结构

```
最终/
├── app.py                  # Flask 后端（Web UI + API，~260 行）
├── main.py                 # CLI 命令行入口（init-data/scan/batch/serve，~230 行）
├── detector.py             # 核心检测引擎：规则引擎 + 加权融合 + LLM 集成（~410 行）
├── features.py             # 6 维特征提取模块：CFG/EH/GI/OP/PF/RB（~310 行）
├── llm_engine.py           # LLM 推理引擎封装：Mock/OpenAI/DeepSeek/Anthropic/Local（~500 行）
├── models_inference.py     # 深度学习模型推理（CNN/MalConv，可选模块，有 GPU 可用）
├── config.json             # 产品配置文件（产品名/版本/检测阈值/模型权重/LLM 设置）
├── requirements.txt        # Python 依赖清单（7 个核心包）
├── run.bat                 # Windows 一键启动脚本（自动安装依赖 + 启动 Web UI）
├── prompt.txt              # LLM 提示词模板（专家分层决策系统用）
├── templates/
│   └── index.html          # Web UI 前端页面（单页应用，支持文件上传、批量检测）
├── static/
│   ├── style.css           # 前端样式（深色主题、响应式设计）
│   └── app.js              # 前端逻辑（文件上传、API 调用、检测结果动态渲染）
├── data/
│   ├── sample_confidence.csv   # 内置示例：1000 条置信度数据（自动生成）
│   └── sample_wild_label.csv    # 内置示例：1000 条对应标签（用于评估）
└── README.md               # 本文件：项目说明文档
```

### 模块职责说明

| 模块            | 主要类/函数                                                  | 核心职责                                                |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| `app.py`        | Flask `app`、`run()`、路由函数                               | 提供 HTTP 服务、处理文件上传、返回 JSON 检测结果        |
| `main.py`       | `init_sample_data()`、`scan_single_file()`、`scan_batch_csv()`、`main()` | 命令行参数解析、数据初始化、CLI 检测流程                |
| `detector.py`   | `ModelConfidence`、`DetectionResult`、`detect_sample()`、`detect_from_csv()`、`evaluate()` | 核心检测逻辑：规则判定、加权融合、LLM 集成、结果评估    |
| `features.py`   | `extract_all()`、`extract_info()`、`extract_cfg/eh/gi/op/pf/rb()` | PE 文件 6 维特征提取，支持启发式 + 深度学习双模式       |
| `llm_engine.py` | `LLMEngine`、`safe_load_json()`、`extract_structured_response()` | 大模型调用封装、多提供商支持、JSON 解析与修复、优雅降级 |

---

## 🔍 核心检测方法（6 维特征）

| 代号    | 维度         | 方法                                                   |
| ------- | ------------ | ------------------------------------------------------ |
| **CFG** | 控制流图     | 启发式近似：统计 call / jmp / jcc / ret 指令字节密度   |
| **EH**  | 字节熵直方图 | 按滑动窗口计算熵分布，衡量加壳/加密程度                |
| **GI**  | 灰度图特征   | 将文件二进制重采样为 135×63 灰度图 → 计算纹理复杂度    |
| **OP**  | 操作码频率   | 统计加密/循环相关字节（REP/STOS/MOVS 等）出现密度      |
| **PF**  | PE 结构特征  | 节区名/属性、导入表危险函数、可写+可执行节区、加壳特征 |
| **RB**  | 原始字节分布 | 256 维字节值直方图 → 归一化熵；有 GPU 可启用 MalConv   |

各维度产出 0~1 的恶意倾向评分后，通过 **规则快速通道 + 加权融合 + 可选 LLM 推理** 的三级决策得到最终判决。

## 🤖 大语言模型（LLM）集成

- **默认使用本地规则引擎（mock 模式）**：无需 API Key，启动即用。
- 支持对接 **OpenAI / DeepSeek / Anthropic / 阿里云百炼 / 本地 OpenAI 兼容服务**（ollama/vllm 等）。
- 对每个样本，LLM 会：
  1. 读取 6 个模型的 confidence
  2. 分析风险模式（全模型一致高 / 全模型一致低 / 存在分歧）
  3. 输出结构化结论（`final_decision` / `reason` / `ensemble_probability` / `suggestion` / `false_positive_risk`）
- 当 API 不可用时自动 fallback 到本地加权融合。

### 三级决策流水线

1. **规则快速通道**：对明显样本（所有模型极低或多数极高）直接判定，节省 LLM 资源。
2. **加权融合**：对规则无法判定的样本，进行加权平均 + 几何均值的复合计算。
3. **LLM 专家推理**（可选）：采用多层专家决策系统，层层递进分析各维度证据。
   - 全局结构层（RB + PF）
   - 行为共识层（CFG + OP + GI）
   - 统计边界层（EH）
   - 加权融合层（综合概率）
   - 专家分析层（决策树 + 定性评估）

核心原则：**优先控制误报**——只有当恶意解释显著比良性解释更可信时才判定为恶意。

### 切换到真实 LLM

```bash
python main.py scan sample.exe \
    --provider deepseek \
    --api-key sk-xxxxxxxxxxxxxxxx \
    --base-url https://api.deepseek.com/chat/completions \
    --model deepseek-chat
```

Web UI 里也可以直接从下拉菜单选择 `DeepSeek / OpenAI / Anthropic / 本地 HTTP`，并在输入框填 Key。

### 修改融合规则

在 `detector.py` 中调整权重分配和检测阈值：

```python
# 默认等权（在 config.json 中可配置）
weights = {"CFG": 1.0, "EH": 1.0, "GI": 1.0, "OP": 1.0, "PF": 1.0, "RB": 1.0}

# 可根据实际测试效果调整为非等权：
# 例如全局结构特征更可靠时可提高 RB/PF 权重
```

## 📊 性能评估

对示例数据运行 LLM 后，使用 Web UI 中的演示功能会返回：

- **Accuracy / Precision / Recall / F1** 四大分类指标
- **TP / FP / TN / FN** 明细统计
- 每个样本的完整检测结果，支持 CSV 下载
- 单文件检测时提供可视化的风险雷达图

## ⚠️ 免责声明

本产品用于**科研演示与教学**目的，特征提取为启发式近似实现，
检测结果仅供参考，不构成对任何样本的"安全/威胁"担保结论。
真实生产环境请使用专业杀毒引擎并结合沙箱与人工分析。

---

*© RansomGuard AI · 基于多模型融合 + AI 大语言模型的勒索软件检测系统*
