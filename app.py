"""
RansomGuard AI · Flask 后端

提供接口:
- GET  /                   Web UI
- POST /api/scan           上传 PE 文件 → 检测 → 返回 DetectionResult JSON
- POST /api/batch          上传 CSV (含6模型置信度) → 批量检测
- GET  /api/demo           使用内置示例数据 (1000条) 做批量检测演示
- GET  /api/download/<id>  下载批量检测结果 CSV
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

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

# 保证同级目录可被 import（无论以何种方式启动）
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from flask import Flask, jsonify, request, render_template, send_file, abort  # noqa: E402
import pandas as pd  # noqa: E402

from detector import (  # noqa: E402
    DetectionResult, ModelConfidence, detect_sample,
    detect_from_csv, evaluate,
)
from features import extract_all, extract_info  # noqa: E402
from llm_engine import LLMEngine  # noqa: E402


app = Flask(__name__,
            template_folder=str(BASE_DIR / "templates"),
            static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB 上传上限

# --- 批量检测结果缓存 (用于下载) ---
_RESULT_STORE: dict[str, list[DetectionResult]] = {}
_RESULT_STORE_LOCK = threading.Lock()


# ==================== 工具函数 ====================

def _build_llm_engine() -> LLMEngine | None:
    """根据请求参数构造 LLM 引擎，未启用时返回 None"""
    use_llm = request.form.get("use_llm", "1") == "1"
    if not use_llm:
        return None
    provider = request.form.get("provider", "mock").strip() or "mock"
    api_key = request.form.get("api_key", "").strip()
    base_url = request.form.get("base_url", "").strip() or None
    model = request.form.get("model", "").strip() or None
    return LLMEngine(provider=provider, api_key=api_key,
                     base_url=base_url, model=model)


def _file_to_hash(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


# ==================== 页面 ====================

@app.get("/")
def index():
    return render_template("index.html")


# ==================== API: 单文件扫描 ====================

@app.post("/api/scan")
def api_scan():
    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "文件名为空"}), 400

    try:
        # 保存到临时文件以便读取 bytes
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.save(tmp.name)
            tmp.flush()
            tmp.close()
            with open(tmp.name, "rb") as fh:
                data = fh.read()
            info = extract_info(tmp.name)
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass

        scores = info["scores"]
        conf = ModelConfidence(
            CFG=float(scores["CFG"]), EH=float(scores["EH"]), GI=float(scores["GI"]),
            OP=float(scores["OP"]),  PF=float(scores["PF"]),  RB=float(scores["RB"]),
        )
        llm_engine = _build_llm_engine()
        result = detect_sample(_file_to_hash(data), conf, llm_engine=llm_engine)

        # 附加 "是否使用了 LLM"
        r_dict = result.to_dict()
        r_dict["verdict_source"] = result.verdict_source

        return jsonify({
            "meta": {
                "filename": f.filename,
                "size_bytes": info["size_bytes"],
                "size_mb": info["size_mb"],
                "is_pe": info["is_pe"],
            },
            "result": r_dict,
            "llm_used": (llm_engine is not None) and result.llm_enabled,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": f"{e.__class__.__name__}: {e}",
                        "trace": traceback.format_exc()}), 500


# ==================== API: 批量检测 ====================

@app.post("/api/batch")
def api_batch():
    if "csv" not in request.files:
        return jsonify({"error": "未上传 CSV"}), 400
    f = request.files["csv"]
    try:
        # 读取为 DataFrame
        df = pd.read_csv(f.stream)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
        df.to_csv(tmp.name, index=False)
        tmp.close()

        limit = int(request.form.get("limit", 1000) or 1000)
        llm_engine = _build_llm_engine()

        results = detect_from_csv(tmp.name, llm_engine=llm_engine, limit=limit)
        os.unlink(tmp.name)

        # 若有 label 列则计算指标
        metrics = None
        if "label" in df.columns:
            labels = df["label"].astype(int).tolist()[: len(results)]
            if len(labels) == len(results):
                metrics = evaluate(results, labels)

        # 保存以供下载
        rid = uuid.uuid4().hex
        with _RESULT_STORE_LOCK:
            _RESULT_STORE[rid] = results

        return jsonify({
            "results": [r.to_dict() for r in results[:200]],  # 给前端前 200 条
            "total": len(results),
            "metrics": metrics,
            "download_url": f"/api/download/{rid}",
        })
    except Exception as e:
        import traceback
        return jsonify({"error": f"{e.__class__.__name__}: {e}",
                        "trace": traceback.format_exc()}), 500


# ==================== API: 示例数据 ====================

@app.get("/api/demo")
def api_demo():
    try:
        sample_path = BASE_DIR / "data" / "sample_confidence.csv"
        if not sample_path.exists():
            return jsonify({"error": "示例数据未找到，请先运行 `python main.py --init-data`"}), 404

        # 本地 mock LLM
        llm_engine = LLMEngine(provider="mock")
        results = detect_from_csv(str(sample_path), llm_engine=llm_engine)

        metrics = None
        label_path = BASE_DIR / "data" / "sample_wild_label.csv"
        if label_path.exists():
            labels_df = pd.read_csv(label_path)
            # 以 hash 对齐
            hash_to_label = dict(zip(labels_df["hash"].astype(str),
                                     labels_df["label"].astype(int)))
            labels = [hash_to_label.get(r.sample_hash, -1) for r in results]
            valid = [(r, y) for r, y in zip(results, labels) if y in (0, 1)]
            if valid:
                metrics = evaluate([v[0] for v in valid], [v[1] for v in valid])

        rid = uuid.uuid4().hex
        with _RESULT_STORE_LOCK:
            _RESULT_STORE[rid] = results

        return jsonify({
            "results": [r.to_dict() for r in results[:200]],
            "total": len(results),
            "metrics": metrics,
            "download_url": f"/api/download/{rid}",
        })
    except Exception as e:
        import traceback
        return jsonify({"error": f"{e.__class__.__name__}: {e}",
                        "trace": traceback.format_exc()}), 500


# ==================== API: 下载结果 ====================

@app.get("/api/download/<rid>")
def api_download(rid: str):
    with _RESULT_STORE_LOCK:
        results = _RESULT_STORE.get(rid)
    if not results:
        abort(404)

    rows = []
    for r in results:
        d = r.model_confidence
        rows.append({
            "hash": r.sample_hash,
            "CFG": d.CFG, "EH": d.EH, "GI": d.GI,
            "OP": d.OP, "PF": d.PF, "RB": d.RB,
            "ensemble_probability": r.ensemble_probability,
            "is_ransomware": int(r.is_ransomware),
            "risk_level": r.risk_level,
            "false_positive_risk": r.false_positive_risk,
            "verdict_source": r.verdict_source,
        })
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"ransomguard_result_{rid[:8]}.csv",
    )


def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    print(f"🚀 RansomGuard AI 已启动: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()