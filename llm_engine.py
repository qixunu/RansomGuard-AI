"""
LLM 推理引擎

- 支持多种提供商: openai / anthropic / deepseek / dashscope / local_http / mock
- 默认走 OpenAI SDK，对 DashScope/DeepSeek/本地 Ollama 均适用
- mock 模式: 5 层专家推理 + 决策树，严格按 "优先控制误报" 原则
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Dict, Optional


# ============ LLM 提示词 ============

EXPERT_PROMPT = """你是一位资深恶意代码分析师，负责在"专家分层决策系统"中完成核心推理。

=== 推理中必须包含的步骤 ===
1. 对每个特征（CFG/EH/GI/OP/PF/RB）分别列出"良性解释"与"恶意解释"；
2. 评估良性解释是否更合理；
3. 若存在合理的良性解释，则不能判定为恶意；
4. 只有当恶意解释显著比良性解释更可信时，才判定为恶意。

=== 关键原则：优先控制误报 ===
判定为"恶意"必须同时满足：
1. 多个相互独立的特征信号强烈且一致地指向恶意行为；
2. 这些信号不能被正常软件行为合理解释；
3. 高置信恶意信号的数量与强度显著超过低置信或矛盾信号；
4. 总体风险评估达到足以造成实际危害的程度。

=== 分类器说明 ===
- CFG (Control Flow Graph): 反汇编提取代码块关系 → GNN
- EH (Extended Entropy Histogram): 字节值频率 + 局部熵模式 → MLP
- GI (Grayscale Image): 二进制降采样为灰度图 → CNN
- OP (Opcode Sequence): 操作码序列 → BiLSTM
- PF (PE Structure): PE 头 + 节布局 → LightGBM
- RB (Raw Bytes / MalConv): 原始字节 1D 卷积

=== 分层决策结构 ===
[Layer 1] 全局结构层 (RB, PF): 双高≥0.7→恶意; 双低≤0.15→良性
[Layer 2] 行为共识层 (CFG, OP, GI): 至少两个 ≥0.6→恶意; 全≤0.2→良性
[Layer 3] 统计边界层 (EH): EH≥0.5 且其他任一≥0.35 且无良性解释→偏恶意
[Layer 4] 加权融合层: P=0.22*RB+0.20*PF+0.18*CFG+0.15*OP+0.13*GI+0.12*EH
         P≥0.6→恶意; P≤0.38→良性
[Layer 5] 专家分析层: 决策树 + 定性评估

=== 样本数据 ===
hash: {sample_id}
CFG={cfg:.4f} EH={eh:.4f} GI={gi:.4f} OP={op:.4f} PF={pf:.4f} RB={rb:.4f}

=== 输出格式（必须严格遵守） ===
第 1 行: `<thinking>`
    列出每个维度的良性/恶意解释，按层序决策，保持客观
    对证据不足或存在良性解释的样本优先判良性
倒数几行: `</thinking>`
然后: `<output>`
    合法 JSON，字段:
    - hash (string, 原样回显)
    - final_decision (1=恶意, 0=良性, -1=不确定)
    - decision_level (1~5)
    - reason (中文简要说明)
    - ensemble_probability (0~1)
    - details (含 CFG/EH/GI/OP/PF/RB 的原始值)
    - false_positive_risk ("low"/"medium"/"high")
    - suggestion (中文建议)
最后: `</output>`
"""


# ============ 本地 mock: 5 层专家推理 ============

def _build_expert_response(conf, sample_id):
    cfg = float(conf.get("CFG", 0.0))
    eh  = float(conf.get("EH",  0.0))
    gi  = float(conf.get("GI",  0.0))
    op  = float(conf.get("OP",  0.0))
    pf  = float(conf.get("PF",  0.0))
    rb  = float(conf.get("RB",  0.0))
    details = {"CFG": round(cfg, 4), "EH": round(eh, 4), "GI": round(gi, 4),
               "OP": round(op, 4), "PF": round(pf, 4), "RB": round(rb, 4)}
    P = 0.22*rb + 0.20*pf + 0.18*cfg + 0.15*op + 0.13*gi + 0.12*eh

    def bn(k, v):
        if v <= 0.25: return f"{k}={v:.2f} 明显良性区间"
        if v <= 0.4: return f"{k}={v:.2f} 偏低，可能是压缩资源段或加壳安装包"
        if v <= 0.6: return f"{k}={v:.2f} 中等，可被正常混淆行为解释"
        return f"{k}={v:.2f} 偏高，单一维度不足以判恶意"

    def ml(k, v):
        if v >= 0.75: return f"{k}={v:.2f} 强阳性"
        if v >= 0.6: return f"{k}={v:.2f} 中高阳性"
        if v >= 0.4: return f"{k}={v:.2f} 弱阳性"
        return f"{k}={v:.2f} 无明显恶意迹象"

    evidence = "\n".join(
        f"  · {k}={float(conf.get(k, 0.0)):.2f}: "
        f"[良性] {bn(k, float(conf.get(k, 0.0)))} | "
        f"[恶意] {ml(k, float(conf.get(k, 0.0)))}"
        for k in ["RB", "PF", "CFG", "OP", "GI", "EH"]
    )

    # Layer 1
    benign_l1 = (rb >= 0.7 and pf < 0.4) or (pf >= 0.7 and rb < 0.3)
    if rb >= 0.7 and pf >= 0.7 and not benign_l1:
        return _pack(sample_id, 1, 1,
                     f"[Layer1] RB={rb:.2f} 且 PF={pf:.2f} 双高 ≥ 0.7，"
                     f"全局字节分布与 PE 结构均显著偏离正常软件；无合理良性解释。",
                     P, details, "low",
                     "建议立即隔离至沙箱，核查 CPU 持续高占、异常矿池 DNS、隐藏进程。",
                     evidence)
    if rb <= 0.15 and pf <= 0.15:
        return _pack(sample_id, 0, 1,
                     f"[Layer1] RB={rb:.2f} 且 PF={pf:.2f} 双低 ≤ 0.15，"
                     f"原始字节与 PE 结构均呈现典型良性模式。",
                     P, details, "low",
                     "无需特殊处置；若来源可疑可在常规监控中观察。",
                     evidence)

    # Layer 2
    high_l2 = sum(1 for x in [cfg, op, gi] if x >= 0.6)
    all_low_l2 = all(x <= 0.2 for x in [cfg, op, gi])
    benign_l2 = (high_l2 == 1 and max(cfg, op, gi) < 0.85)
    if high_l2 >= 2 and not benign_l2:
        names = [k for k, x in [("CFG", cfg), ("OP", op), ("GI", gi)] if x >= 0.6]
        return _pack(sample_id, 1, 2,
                     f"[Layer2] {', '.join(names)} 等 ≥ 2 个维度同时 ≥ 0.6，"
                     f"控制流/操作码/二进制图像出现一致性恶意模式。",
                     P, details, "low",
                     "建议立即隔离至沙箱，核查反汇编控制流、CPU 高占用与异常网络外联。",
                     evidence)
    if all_low_l2:
        return _pack(sample_id, 0, 2,
                     f"[Layer2] CFG={cfg:.2f}, OP={op:.2f}, GI={gi:.2f} 全 ≤ 0.2，"
                     f"空间与行为维度无明显恶意迹象。",
                     P, details, "low",
                     "无需特殊处置。",
                     evidence)

    # Layer 3 (EH)
    any_other_h = any(x >= 0.35 for x in [cfg, op, gi, pf, rb])
    benign_l3 = (eh >= 0.5 and not any_other_h) or \
                (0.5 <= eh < 0.7 and all(x < 0.5 for x in [cfg, op, gi, pf, rb]))
    if eh >= 0.5 and any_other_h and not benign_l3:
        return _pack(sample_id, 1, 3,
                     f"[Layer3] EH={eh:.2f} ≥ 0.5，且存在其他维度 ≥ 0.35 的辅助证据；"
                     f"熵异常与结构性异常同时出现，无法被单纯压缩/混淆行为合理解释。",
                     P, details, "medium",
                     "建议在沙箱中观测 10-30 分钟，重点观察 CPU、文件加密与网络外联。",
                     evidence)

    # Layer 4 (加权融合)
    if P >= 0.6:
        return _pack(sample_id, 1, 4,
                     f"[Layer4] P={P:.3f} ≥ 0.6，加权后综合恶意证据达到风险阈值。",
                     P, details, "medium",
                     "建议在沙箱中运行 15 分钟以上，核查 CPU/DNS/进程行为。",
                     evidence)
    if P <= 0.38:
        return _pack(sample_id, 0, 4,
                     f"[Layer4] P={P:.3f} ≤ 0.38，各维度综合加权后恶意证据不足。",
                     P, details, "low",
                     "无需特殊处置；如来源不明可保留常规监控。",
                     evidence)

    # Layer 5 (决策树 + 定性评估)
    td = _decision_tree(rb, pf, cfg, gi, op, eh)
    if td == 1 and P >= 0.5:
        return _pack(sample_id, 1, 5,
                     f"[Layer5] P={P:.3f} 处于模糊区间，但决策树支持恶意判断；"
                     f"综合评估恶意解释较良性解释更可信。",
                     P, details, "high",
                     "建议上传至威胁情报平台交叉比对并人工复检。",
                     evidence)
    return _pack(sample_id, 0, 5,
                 f"[Layer5] P={P:.3f} 处于模糊区间，"
                 f"决策树支持良性判断；在证据不足情况下优先判良性避免误报。",
                 P, details, "medium",
                 "建议保留常规监控；如出现行为异常可重新提交检测。",
                 evidence)


def _decision_tree(rb, pf, cfg, gi, op, eh):
    if rb <= 0.08:
        if pf <= 0.37: return 0
        if cfg <= 0.39: return 0
        if gi <= 0.26: return 0
        return 1
    if gi <= 0.83:
        if op <= 0.67:
            if cfg <= 0.34: return 0
            return 1
        return 1
    if cfg <= 0.29: return 0
    return 1


def _pack(sample_id, decision, level, reason, P, details, fp_risk,
          suggestion, evidence):
    return {
        "hash": sample_id,
        "final_decision": int(decision),
        "decision_level": int(level),
        "reason": reason,
        "ensemble_probability": round(float(P), 6),
        "details": details,
        "false_positive_risk": fp_risk,
        "suggestion": suggestion,
        "_evidence": evidence,
    }


# ============ 可靠的 JSON 解析与修复（参考 agent_runner.py） ============

def safe_load_json(text):
    """8 层 JSON 修复逻辑，专门应对大模型常见的格式输出问题"""
    if not text:
        return None
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)

    if "'" in text and '"' not in text[:10]:
        text = re.sub(r"'", '"', text)
    text = re.sub(r",\s*([\]}])", r"\1", text)
    text = text.replace("None", "null").replace("True", "true").replace("False", "false")
    o, c = text.count("{"), text.count("}")
    if o > c: text += "}" * (o - c)
    o, c = text.count("["), text.count("]")
    if o > c: text += "]" * (o - c)
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e + 1])
    except Exception:
        pass
    try:
        import json_repair as jr
        return jr.loads(text)
    except Exception:
        pass
    return None


def extract_structured_response(text):
    if not text:
        return None
    m = re.search(r"<output>\s*(.*?)\s*</output>", text, re.DOTALL | re.IGNORECASE)
    if m:
        parsed = safe_load_json(m.group(1))
        if parsed is not None:
            return parsed
    return safe_load_json(text)


# ============ LLM 引擎类（统一用 OpenAI SDK） ============

class LLMConfigError(RuntimeError):
    pass


class LLMEngine:
    """通用 LLM 调用封装 —— 除 Anthropic 外统一走 OpenAI SDK"""

    _DEFAULTS = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat", "deepseek-reasoner", "deepseek-v3"],
        },
        "dashscope": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "models": ["deepseek-v3.2", "deepseek-v3",
                       "qwen-plus", "qwen-max", "qwen-long"],
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1",
            "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
        },
        "local_http": {
            "base_url": "http://localhost:11434/v1",
            "models": ["qwen2.5:7b", "llama3:8b", "qwen3:8b"],
        },
    }

    _ALIASES = {
        "local": "mock", "offline": "mock", "demo": "mock",
        "gpt": "openai", "openai-api": "openai",
        "claude": "anthropic",
        "qwen": "dashscope", "qwen-plus": "dashscope", "qwen-max": "dashscope",
        "tongyi": "dashscope", "tongyi-qianwen": "dashscope",
        "通义千问": "dashscope", "百炼": "dashscope",
        "bailian": "dashscope", "aliyun": "dashscope",
        "dashscope-api": "dashscope",
        "ollama": "local_http", "vllm": "local_http",
    }

    def __init__(self, provider="mock", api_key=None, base_url=None,
                 model=None, max_retries=4, retry_delay=1.0, timeout=60.0):
        raw = (provider or "").lower().strip() or "mock"
        self.provider = self._ALIASES.get(raw, raw)

        if self.provider == "mock":
            self.api_key = self.base_url = self.model = None
            self.max_retries = 0
            self.timeout = timeout
            return

        if self.provider not in self._DEFAULTS:
            valid = ", ".join(list(self._DEFAULTS.keys()) + ["mock"])
            raise ValueError(f"不支持的 LLM 提供商: '{provider}'。可选: {valid}")

        cfg = self._DEFAULTS[self.provider]
        self.api_key = api_key or os.environ.get("RANSOMGUARD_API_KEY", "")
        _url = base_url or cfg["base_url"]
        # 去除 URL 两侧的反引号、空白（Markdown 里的 `url` 格式）
        self.base_url = _url.strip().strip("`").strip().rstrip("/")
        self.model = (model or cfg["models"][0]).strip()
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.timeout = float(timeout)

    def call(self, conf, sample_id):
        if self.provider == "mock":
            time.sleep(random.uniform(0.05, 0.15))
            return _build_expert_response(conf, sample_id)

        prompt = EXPERT_PROMPT.format(
            sample_id=sample_id,
            cfg=float(conf.get("CFG", 0.0)),
            eh=float(conf.get("EH", 0.0)),
            gi=float(conf.get("GI", 0.0)),
            op=float(conf.get("OP", 0.0)),
            pf=float(conf.get("PF", 0.0)),
            rb=float(conf.get("RB", 0.0)),
        )

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._do_call(prompt, attempt)
            except LLMConfigError:
                raise  # 配置错误直接抛，让 detector.py 降级
            except Exception as e:
                last_err = e
                backoff = self.retry_delay * (2 ** (attempt - 1)) + 0.1 * attempt
                time.sleep(backoff)

        raise RuntimeError(f"LLM 调用连续 {self.max_retries} 次失败: {last_err}")

    def _do_call(self, prompt, attempt):
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_openai_sdk(prompt, attempt)

    def _call_openai_sdk(self, prompt, attempt):
        """统一用 OpenAI SDK 调用（DashScope / DeepSeek / OpenAI / Ollama 都兼容）"""
        from openai import OpenAI

        if not self.api_key:
            raise LLMConfigError(
                f"[{self.provider}] 未填写 API Key。请在 Web UI 的 'LLM 配置' 中填入，"
                f"或设置环境变量 RANSOMGUARD_API_KEY。"
            )

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "你是一位专业的恶意代码威胁情报分析师。"
                                "输出必须以 <thinking> 标签包裹推理链，"
                                "然后以 <output> 标签包裹一个合法 JSON 对象。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=30000,
            )

            # 兼容多种响应格式
            content = None
            try:
                content = resp.choices[0].message.content  # SDK 对象风格
            except Exception:
                try:
                    content = resp["choices"][0]["message"]["content"]  # dict 风格
                except Exception:
                    content = str(resp)

            if not content or not str(content).strip():
                models = ", ".join(self._DEFAULTS[self.provider]["models"])
                raise LLMConfigError(
                    f"[{self.provider}] 模型 '{self.model}' 返回空内容。"
                    f"请确认模型名是否正确，候选: {models}"
                )

            parsed = extract_structured_response(str(content))
            if parsed is None:
                raise LLMConfigError(
                    f"[{self.provider}] 模型返回内容无法解析为 JSON。"
                    f"原始输出前 200 字符: {str(content)[:200]}"
                )
            return parsed

        except Exception as e:
            msg = str(e).lower()
            raw = str(e)

            if "401" in msg or "unauthorized" in msg or "authentication" in msg:
                raise LLMConfigError(
                    f"[{self.provider}] API Key 无效或未授权 (HTTP 401)。\n"
                    f"  · 检查 Key 是否正确（不要包含多余空格/换行）\n"
                    f"  · 是否已开通 '{self.provider}' 对应服务\n"
                    f"  · Base URL: {self.base_url}\n"
                    f"  · 服务器返回: {raw[:300]}"
                )

            if "404" in msg or "not_found" in msg or "not found" in msg:
                models = ", ".join(self._DEFAULTS[self.provider]["models"])
                raise LLMConfigError(
                    f"[{self.provider}] 请求地址或模型名错误 (HTTP 404)。\n"
                    f"  · 当前模型名: '{self.model}'\n"
                    f"  · 候选模型名: {models}\n"
                    f"  · 当前 Base URL: {self.base_url}\n"
                    f"  · 提示: DashScope/百炼 需要在控制台先 '开通' 对应模型；"
                    f"DeepSeek 官方的默认模型是 'deepseek-chat'。"
                )

            if "429" in msg or "rate limit" in msg or "quota" in msg:
                raise RuntimeError(
                    f"[{self.provider}] 请求过于频繁或额度不足 (HTTP 429)。"
                    f"若持续出现请检查账户余额或套餐配额。"
                )

            if ("timeout" in msg or "timed out" in msg
                    or "502" in msg or "503" in msg or "500" in msg):
                raise RuntimeError(
                    f"[{self.provider}] 服务端或网络问题（会自动重试）: {raw[:200]}"
                )

            if "invalid" in msg and ("api" in msg or "key" in msg or "url" in msg):
                raise LLMConfigError(
                    f"[{self.provider}] 配置错误: {raw[:300]}\n"
                    f"  · Base URL: {self.base_url}\n"
                    f"  · 当前模型: {self.model}"
                )

            # 默认：普通错误让外层重试或降级
            raise RuntimeError(f"[{self.provider}] 调用失败 (attempt {attempt}): {raw[:300]}")

    def _call_anthropic(self, prompt):
        import requests
        if not self.api_key:
            raise LLMConfigError("[Anthropic] 未填写 API Key。")

        resp = requests.post(
            self.base_url.rstrip("/") + "/v1/messages",
            headers={"x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 2048,
                  "system": "你是一位专业的威胁情报分析师，严格按要求输出。",
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            detail = resp.text[:500]
            raise LLMConfigError(
                f"[Anthropic] HTTP {resp.status_code}: {detail}"
            )
        body = resp.json()
        text = "".join(
            (b.get("text") or "") for b in body.get("content", [])
            if isinstance(b, dict)
        )
        parsed = extract_structured_response(text)
        if parsed is None:
            raise LLMConfigError(f"[Anthropic] 返回内容无法解析为 JSON: {text[:200]}")
        return parsed