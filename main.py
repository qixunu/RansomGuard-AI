"""
RansomGuard AI · CLI 入口

支持:
  python main.py --init-data          # 从原科研项目读取 1000 条示例 confidence 数据
  python main.py --scan file.exe      # 扫描单个文件
  python main.py --run-server         # 启动 Web UI
  python main.py --batch data.csv     # 批量扫描 CSV
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

# Windows 下强制以 UTF-8 输出到控制台，避免 GBK 编码错误
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from detector import ModelConfidence, detect_sample, detect_from_csv, evaluate
from features import extract_info, extract_all
from llm_engine import LLMEngine


# ====================== 示例数据初始化 ======================

def _try_locate_original_confidence() -> Path | None:
    """尝试在父目录（原科研项目）里找到 'crypto_confidence/wild/merged_wild.csv' """
    candidates = [
        BASE_DIR.parent / "crypto_confidence" / "wild" / "merged_wild.csv",
        BASE_DIR.parent.parent / "crypto_confidence" / "wild" / "merged_wild.csv",
        BASE_DIR / "crypto_confidence" / "wild" / "merged_wild.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def init_sample_data():
    """把原科研项目里的 1000 条数据复制到 data/sample_confidence.csv"""
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    src = _try_locate_original_confidence()
    sample_csv = data_dir / "sample_confidence.csv"
    sample_label = data_dir / "sample_wild_label.csv"

    if src is None:
        print("[!] 未找到原始数据 (crypto_confidence/wild/merged_wild.csv)，")
        print("    尝试从内置的示例 wild_label.csv 合成示例数据...")

        # 生成 1000 条合成数据用于演示
        import numpy as np
        rng = np.random.default_rng(42)
        n = 1000
        # 生成一部分"明显良性"、一部分"明显恶意"、一部分"模糊"
        labels = rng.binomial(1, 0.55, size=n)
        conf = []
        for y in labels:
            if y == 1:
                base = 0.55 + rng.beta(3, 1.5) * 0.45
                noise = rng.normal(0, 0.08, size=6)
            else:
                base = 0.1 + rng.beta(1.5, 3) * 0.45
                noise = rng.normal(0, 0.08, size=6)
            arr = [max(0.0, min(1.0, base + n)) for n in noise]
            conf.append(arr)
        df = pd.DataFrame(conf, columns=["CFG", "EH", "GI", "OP", "PF", "RB"])
        df.insert(0, "hash", [f"demo_{i:04d}" for i in range(n)])
        df["label"] = labels.astype(int)
        df.to_csv(sample_csv, index=False)
        df[["hash", "label"]].to_csv(sample_label, index=False)
        print(f"    已生成 {n} 条合成示例数据 → {sample_csv}")
        return

    df = pd.read_csv(src)
    # 期望列: hash, CFG, EH, GI, OP, PF, RB, label
    print(f"[✓] 找到源数据 {src} ({len(df)} 条)")
    df2 = df.head(1000).copy()
    df2.to_csv(sample_csv, index=False)
    print(f"[✓] 已写入 {sample_csv} ({len(df2)} 条)")

    if "label" in df2.columns:
        df2[["hash", "label"]].to_csv(sample_label, index=False)
        print(f"[✓] 已写入 {sample_label}")
    else:
        print("[i] 源数据无 label 列 → 跳过生成 sample_wild_label.csv")


# ====================== 其他子命令 ======================

def scan_single_file(path: str, use_llm: bool, provider: str, api_key: str,
                     base_url: str, model: str):
    info = extract_info(path)
    s = info["scores"]
    conf = ModelConfidence(
        CFG=float(s["CFG"]), EH=float(s["EH"]), GI=float(s["GI"]),
        OP=float(s["OP"]), PF=float(s["PF"]), RB=float(s["RB"]),
    )
    engine = LLMEngine(provider=provider, api_key=api_key,
                       base_url=base_url or None, model=model or None) if use_llm else None
    result = detect_sample(
        Path(path).name if False else f"file:{os.path.basename(path)}",
        conf, llm_engine=engine,
    )
    print("=" * 60)
    print(f"文件: {path}")
    print(f"大小: {info['size_mb']:.2f} MB   是PE: {info['is_pe']}")
    print("-" * 60)
    print("各模型评分:")
    for k in ["CFG", "EH", "GI", "OP", "PF", "RB"]:
        print(f"  {k}: {s[k]:.3f}")
    print("-" * 60)
    verdict = "勒索软件" if result.is_ransomware else "良性"
    print(f"最终判决: {verdict}")
    print(f"综合恶意概率: {result.ensemble_probability:.3f}")
    print(f"风险等级: {result.risk_level.upper()}")
    print(f"误报风险: {result.false_positive_risk}")
    print(f"判决来源: {result.verdict_source}")
    print("-" * 60)
    print(f"推理过程: {result.reasoning}")
    print(f"LLM 观点: {result.llm_verdict or '(未启用)'}")
    print(f"处置建议: {result.recommendation}")


def scan_batch_csv(csv_path: str, limit: int, use_llm: bool, provider: str,
                   api_key: str, base_url: str, model: str, output: str):
    engine = LLMEngine(provider=provider, api_key=api_key,
                       base_url=base_url or None, model=model or None) if use_llm else None
    results = detect_from_csv(csv_path, llm_engine=engine, limit=limit)
    df_out = pd.DataFrame([
        {
            "hash": r.sample_hash,
            "CFG": r.model_confidence.CFG,
            "EH":  r.model_confidence.EH,
            "GI":  r.model_confidence.GI,
            "OP":  r.model_confidence.OP,
            "PF":  r.model_confidence.PF,
            "RB":  r.model_confidence.RB,
            "ensemble_probability": r.ensemble_probability,
            "is_ransomware": int(r.is_ransomware),
            "risk_level": r.risk_level,
        }
        for r in results
    ])

    # 若原 CSV 含 label 则计算指标
    src_df = pd.read_csv(csv_path)
    metrics = None
    if "label" in src_df.columns:
        labels = src_df["label"].astype(int).tolist()[: len(results)]
        if len(labels) == len(results):
            metrics = evaluate(results, labels)
            print("\n== 指标 (与 label 列对比) ==")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

    out_path = output or (Path(csv_path).stem + "_ransomguard_result.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n[✓] 共检测 {len(results)} 个样本 → {out_path}")
    print(f"    勒索软件: {sum(1 for r in results if r.is_ransomware)}")
    print(f"    良性:     {sum(1 for r in results if not r.is_ransomware)}")


# ====================== 主入口 ======================

def main():
    ap = argparse.ArgumentParser(description="RansomGuard AI - 基于多模型融合 + LLM 的勒索软件检测")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-data", help="初始化示例数据 (1000 条)")

    p_scan = sub.add_parser("scan", help="扫描单个文件")
    p_scan.add_argument("file", help="要扫描的 PE 文件路径")
    _add_llm_args(p_scan)

    p_batch = sub.add_parser("batch", help="批量扫描 CSV (需要 CFG/EH/GI/OP/PF/RB 列)")
    p_batch.add_argument("csv", help="CSV 文件路径")
    p_batch.add_argument("--limit", type=int, default=1000, help="处理行数上限")
    p_batch.add_argument("--output", default="", help="输出 CSV 路径")
    _add_llm_args(p_batch)

    p_srv = sub.add_parser("serve", help="启动 Web UI")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=5000)
    p_srv.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    if args.cmd == "init-data":
        init_sample_data()
    elif args.cmd == "scan":
        scan_single_file(args.file, not args.no_llm, args.provider,
                          args.api_key, args.base_url, args.model)
    elif args.cmd == "batch":
        scan_batch_csv(args.csv, args.limit, not args.no_llm, args.provider,
                       args.api_key, args.base_url, args.model, args.output)
    elif args.cmd == "serve":
        # 首次启动前确保有示例数据
        if not (BASE_DIR / "data" / "sample_confidence.csv").exists():
            init_sample_data()
        from app import run
        run(host=args.host, port=args.port, debug=args.debug)


def _add_llm_args(p: argparse.ArgumentParser):
    p.add_argument("--no-llm", action="store_true",
                   help="禁用 LLM，仅使用本地规则和加权融合 (更快、离线可用)")
    p.add_argument("--provider", default="mock",
                   help="LLM 提供方: mock (默认)/ openai / deepseek / anthropic / local_http")
    p.add_argument("--api-key", default=os.environ.get("RANSOMGUARD_API_KEY", ""),
                   help="LLM API Key (也可通过环境变量 RANSOMGUARD_API_KEY 配置)")
    p.add_argument("--base-url", default="", help="自定义 Base URL")
    p.add_argument("--model", default="", help="自定义模型名 (如 gpt-4o-mini)")


if __name__ == "__main__":
    main()