"""
RansomGuard AI - 核心检测引擎
多模型融合 (CFG / EH / GI / OP / PF / RB) + LLM 推理
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ===================== 数据结构 =====================

@dataclass
class ModelConfidence:
    """单个模型的置信度结果"""
    CFG: float = 0.0
    EH: float = 0.0
    GI: float = 0.0
    OP: float = 0.0
    PF: float = 0.0
    RB: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"CFG": self.CFG, "EH": self.EH, "GI": self.GI,
                "OP": self.OP, "PF": self.PF, "RB": self.RB}

    def to_array(self) -> np.ndarray:
        return np.array([self.CFG, self.EH, self.GI,
                         self.OP, self.PF, self.RB], dtype=float)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ModelConfidence":
        return cls(
            CFG=float(d.get("CFG", 0.0)),
            EH=float(d.get("EH", 0.0)),
            GI=float(d.get("GI", 0.0)),
            OP=float(d.get("OP", 0.0)),
            PF=float(d.get("PF", 0.0)),
            RB=float(d.get("RB", 0.0)),
        )


@dataclass
class DetectionResult:
    """完整的检测结果"""
    sample_hash: str
    is_ransomware: bool
    ensemble_probability: float
    risk_level: str                 # low / medium / high
    model_confidence: ModelConfidence
    reasoning: str                  # 推理过程文本
    recommendation: str             # 建议处置
    false_positive_risk: str        # low / medium / high
    llm_enabled: bool = True
    llm_verdict: Optional[str] = None
    per_model_evidence: Dict[str, Dict[str, str]] = field(default_factory=dict)
    verdict_source: str = "ensemble"   # ensemble / llm / rule

    def to_dict(self) -> dict:
        return {
            "sample_hash": self.sample_hash,
            "is_ransomware": self.is_ransomware,
            "ensemble_probability": round(float(self.ensemble_probability), 6),
            "risk_level": self.risk_level,
            "model_confidence": self.model_confidence.as_dict(),
            "reasoning": self.reasoning,
            "recommendation": self.recommendation,
            "false_positive_risk": self.false_positive_risk,
            "llm_enabled": self.llm_enabled,
            "llm_verdict": self.llm_verdict,
            "verdict_source": self.verdict_source,
        }


# ===================== 规则 / 融合引擎 =====================

def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def rule_based_check(conf: ModelConfidence) -> Tuple[Optional[bool], str]:
    """
    基于规则的快速判断（可作为 LLM 不可用时的 fallback）
    返回 (decision, reason)，decision=None 表示规则无法判断，需交给融合模型
    """
    d = conf.as_dict()
    high_values = [v for v in d.values() if v >= 0.8]
    low_values = [v for v in d.values() if v <= 0.2]

    # 规则1：极端良性（所有模型 <= 0.15）
    if all(v <= 0.15 for v in d.values()):
        return False, "所有6个检测模型均给出极低恶意评分 (≤0.15)，无任何勒索软件特征信号。"

    # 规则2：极端恶意（>=5个模型 >=0.85）
    if sum(1 for v in d.values() if v >= 0.85) >= 5:
        return True, ">=5个检测模型给出极高恶意评分 (≥0.85)，涵盖结构、字节、操作码等多个维度。"

    # 规则3：强矛盾信号 (>=2个 >=0.7 且 >=2个 <=0.3)
    if len(high_values) >= 2 and len(low_values) >= 2:
        return None, "各模型存在明显分歧，需由集成模型综合判断。"

    return None, ""


def weighted_ensemble(conf: ModelConfidence, weights: Optional[Dict[str, float]] = None) -> float:
    """加权平均融合。默认等权。"""
    if weights is None:
        weights = {"CFG": 1.0, "EH": 1.0, "GI": 1.0, "OP": 1.0, "PF": 1.0, "RB": 1.0}

    arr = conf.to_array()
    w = np.array([weights[k] for k in ["CFG", "EH", "GI", "OP", "PF", "RB"]], dtype=float)
    w = w / w.sum()
    return float(np.clip(float(np.dot(arr, w)), 0.0, 1.0))


def geometric_mean_ensemble(conf: ModelConfidence) -> float:
    """几何平均——对高维一致恶意的样本惩罚更小，更保守。"""
    arr = conf.to_array()
    # 避免 log(0)
    arr = np.clip(arr, 1e-6, 1.0)
    return float(np.clip(float(np.exp(np.mean(np.log(arr)))), 0.0, 1.0))


def risk_level(prob: float) -> str:
    if prob < 0.33:
        return "low"
    elif prob < 0.67:
        return "medium"
    return "high"


def fp_risk(prob: float, spread: float) -> str:
    """根据概率离边界的距离和模型间的分歧度估算误报风险"""
    distance = abs(prob - 0.5)
    if distance >= 0.3 and spread <= 0.3:
        return "low"
    elif distance >= 0.15 or spread <= 0.4:
        return "medium"
    return "high"


def build_reasoning(conf: ModelConfidence, ensemble_prob: float,
                    decision: bool) -> Tuple[str, str, Dict[str, Dict[str, str]]]:
    """
    构造可解释的推理文本（不调用 LLM 时的 fallback reasoning）
    """
    d = conf.as_dict()
    evidence = {}

    # 按模型生成"证据"
    reasons = []
    for name, v in d.items():
        if v >= 0.75:
            tone = "强阳性"
            detail = f"{name} 评分 {v:.3f}，明显高于良性阈值 0.5"
        elif v >= 0.5:
            tone = "弱阳/可疑"
            detail = f"{name} 评分 {v:.3f}，存在可疑模式但非决定性"
        elif v >= 0.25:
            tone = "弱阴/不确定"
            detail = f"{name} 评分 {v:.3f}，与典型勒索软件特征有一定差异"
        else:
            tone = "强阴性"
            detail = f"{name} 评分 {v:.3f}，未检测到对应维度的恶意特征"
        evidence[name] = {"score": f"{v:.3f}", "tone": tone, "detail": detail}

    if decision:
        reasons.append(f"综合评分 {ensemble_prob:.3f} ≥ 0.5，判定为勒索软件。")
        high_models = [k for k, v in d.items() if v >= 0.7]
        if high_models:
            reasons.append(f"强阳性模型: {', '.join(high_models)}。")
        low_models = [k for k, v in d.items() if v <= 0.3]
        if low_models:
            reasons.append(f"但 {', '.join(low_models)} 为弱信号，存在一定误报可能。")
    else:
        reasons.append(f"综合评分 {ensemble_prob:.3f} < 0.5，判定为良性。")
        low_models = [k for k, v in d.items() if v <= 0.2]
        if low_models:
            reasons.append(f"强阴性模型: {', '.join(low_models)}。")
        high_models = [k for k, v in d.items() if v >= 0.6]
        if high_models:
            reasons.append(f"需要注意: {', '.join(high_models)} 存在弱阳信号，建议关注。")

    reasoning_text = " ".join(reasons)
    recommendation = (
        "立即隔离该文件，提交至沙箱进一步动态分析，并检查主机近期是否存在异常文件加密、"
        "大量文件重命名、CPU/GPU 占用异常升高等行为。"
        if decision else
        "无特殊处置，可纳入常规监控；若来自外部来源或可疑路径，建议保持谨慎观察。"
    )
    return reasoning_text, recommendation, evidence


# ===================== 主检测函数 =====================

def detect_sample(hash_or_id: str, conf: ModelConfidence,
                  llm_engine=None, llm_mode: str = "auto") -> DetectionResult:
    """
    对单样本做完整的检测判决。

    llm_engine: 可选的 LLM 引擎对象（需实现 call(conf_dict, sample_id) -> dict）
                为 None 时完全使用本地规则+融合。
    llm_mode: "auto"（默认，LLM可用则用） | "llm_only" | "local_only"
    """
    # 1. 规则快速检查
    rule_decision, rule_reason = rule_based_check(conf)

    # 2. 集成模型计算
    w_mean = weighted_ensemble(conf)
    g_mean = geometric_mean_ensemble(conf)
    # 最终概率取加权平均（实际生产中可用训练好的 LogReg/MLP 替代）
    ensemble_prob = (w_mean + g_mean) / 2.0

    spread = float(conf.to_array().max() - conf.to_array().min())
    rl = risk_level(ensemble_prob)
    fp = fp_risk(ensemble_prob, spread)

    # 3. LLM 推理（如果启用且提供）
    llm_verdict: Optional[str] = None
    reasoning, recommendation, evidence = "", "", {}
    use_llm = (llm_engine is not None) and (llm_mode != "local_only")

    if use_llm:
        try:
            llm_out = llm_engine.call(conf.as_dict(), hash_or_id)
            # 结构化字段
            llm_decision = bool(llm_out.get("final_decision",
                                            1 if ensemble_prob >= 0.5 else 0))
            llm_verdict = llm_out.get("reason", f"LLM 判定: {'勒索软件' if llm_decision else '良性'}")
            llm_prob = float(llm_out.get("ensemble_probability", ensemble_prob))

            # LLM 与集成模型分歧时，以 LLM 置信度为准做加权校正
            if abs(llm_prob - ensemble_prob) > 0.3:
                # 存在较大分歧 → 给 LLM 更高话语权
                ensemble_prob = 0.6 * llm_prob + 0.4 * ensemble_prob
                rl = risk_level(ensemble_prob)
                fp = fp_risk(ensemble_prob, spread)
                final_decision = ensemble_prob >= 0.5
                source = "llm+ensemble"
            else:
                final_decision = ensemble_prob >= 0.5
                source = "ensemble"

            reasoning = (
                f"[LLM 推理] {llm_verdict}  [本地集成评分: {w_mean:.3f} / 几何均值: {g_mean:.3f}]"
            )
            recommendation = llm_out.get("suggestion", recommendation or "")
            # 构造 evidence
            for model_name, v in conf.as_dict().items():
                evidence[model_name] = {
                    "score": f"{v:.3f}",
                    "tone": "强阳性" if v >= 0.75 else ("强阴性" if v <= 0.25 else "不确定"),
                    "detail": f"{model_name} 评分 {v:.3f}",
                }

            return DetectionResult(
                sample_hash=hash_or_id,
                is_ransomware=final_decision,
                ensemble_probability=ensemble_prob,
                risk_level=rl,
                model_confidence=conf,
                reasoning=reasoning,
                recommendation=recommendation,
                false_positive_risk=fp,
                llm_enabled=True,
                llm_verdict=llm_verdict,
                per_model_evidence=evidence,
                verdict_source=source,
            )
        except Exception as e:
            # —— LLM 阶段的任何错误都不该让整个检测崩溃 ——
            # 区分"配置错误"和"临时网络错误"但采取相同策略：
            #   · 回退到本地加权融合
            #   · 在 reasoning 开头附加一条清晰的中文提示，告诉用户发生了什么
            err_text = str(e).lower()
            config_error_markers = ("llmconfigerror", "401", "404", "429",
                                    "api key", "apikey", "未填写",
                                    "不支持", "未知的 provider", "json 解析失败")
            is_user_config_error = any(m in err_text for m in config_error_markers)

            reasoning, recommendation, evidence = build_reasoning(conf, ensemble_prob,
                                                                  ensemble_prob >= 0.5)
            if is_user_config_error:
                prefix = ("[LLM 跳过：配置错误] "
                         f"{e}。提示：检查 API Key、Model 名称和 Base URL。"
                         "  当前检测结果来自本地 6 模型加权融合。\n")
            else:
                prefix = f"[LLM 服务不可用，已回退本地集成] {e}\n"
            reasoning = prefix + reasoning

            # 注意：直接 return，不进入下面的"规则判定"分支（rule_decision 不是 None 时也不进）
            return DetectionResult(
                sample_hash=hash_or_id,
                is_ransomware=ensemble_prob >= 0.5,
                ensemble_probability=ensemble_prob,
                risk_level=rl,
                model_confidence=conf,
                reasoning=reasoning,
                recommendation=recommendation,
                false_positive_risk=fp,
                llm_enabled=False,
                llm_verdict=None,
                per_model_evidence=evidence,
                verdict_source="ensemble (llm_fallback)",
            )

    # 4. 本地（规则 + 集成）路径
    if rule_decision is not None:
        reasoning = f"[规则判定] {rule_reason}\n"
        local_reason, recommendation, evidence = build_reasoning(conf, ensemble_prob, rule_decision)
        reasoning += local_reason
        return DetectionResult(
            sample_hash=hash_or_id,
            is_ransomware=rule_decision,
            ensemble_probability=ensemble_prob,
            risk_level=rl,
            model_confidence=conf,
            reasoning=reasoning,
            recommendation=recommendation,
            false_positive_risk=fp,
            llm_enabled=False,
            llm_verdict=None,
            per_model_evidence=evidence,
            verdict_source="rule",
        )

    reasoning, recommendation, evidence = build_reasoning(conf, ensemble_prob,
                                                          ensemble_prob >= 0.5)
    return DetectionResult(
        sample_hash=hash_or_id,
        is_ransomware=ensemble_prob >= 0.5,
        ensemble_probability=ensemble_prob,
        risk_level=rl,
        model_confidence=conf,
        reasoning=reasoning,
        recommendation=recommendation,
        false_positive_risk=fp,
        llm_enabled=False,
        llm_verdict=None,
        per_model_evidence=evidence,
        verdict_source="ensemble",
    )


# ===================== 批量检测 =====================

def detect_from_csv(csv_path: str, llm_engine=None,
                    label_col: Optional[str] = None,
                    limit: Optional[int] = None) -> List[DetectionResult]:
    """
    从 CSV 文件批量读取置信度并检测。
    CSV 需包含: hash + [CFG, EH, GI, OP, PF, RB]
    """
    df = pd.read_csv(csv_path)
    required = ["CFG", "EH", "GI", "OP", "PF", "RB"]
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV 必须包含列: {required}，实际列: {list(df.columns)}")
    if "hash" not in df.columns:
        raise ValueError("CSV 必须包含 'hash' 列")

    if limit is not None:
        df = df.head(limit)

    results = []
    for _, row in df.iterrows():
        conf = ModelConfidence(
            CFG=_clamp(row["CFG"]), EH=_clamp(row["EH"]),
            GI=_clamp(row["GI"]), OP=_clamp(row["OP"]),
            PF=_clamp(row["PF"]), RB=_clamp(row["RB"]),
        )
        r = detect_sample(str(row["hash"]), conf, llm_engine=llm_engine)
        results.append(r)

    return results


def evaluate(results: List[DetectionResult], labels: List[int]) -> dict:
    """计算精度指标，labels 必须与 results 同长度"""
    preds = [1 if r.is_ransomware else 0 for r in results]
    y = np.array(labels, dtype=int)
    p = np.array(preds, dtype=int)

    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    tn = int(np.sum((p == 0) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))

    accuracy = (tp + tn) / max(1, len(y))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "total": len(y),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }