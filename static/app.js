(function () {
  "use strict";

  const statusBadge = document.getElementById("status-badge");
  statusBadge.textContent = "就绪";

  const uploadForm = document.getElementById("upload-form");
  const useLLM = document.getElementById("use-llm");
  const provider = document.getElementById("llm-provider");
  const apiKeyRow = document.getElementById("api-key-row");
  const apiKeyInput = document.getElementById("api-key");
  const apiBaseInput = document.getElementById("api-base");
  const apiModelInput = document.getElementById("api-model");
  const submitBtn = document.getElementById("submit-btn");

  const progressCard = document.getElementById("progress-card");
  const progressBar = document.getElementById("progress-bar");
  const progressLog = document.getElementById("progress-log");

  const resultCard = document.getElementById("result-card");
  const resultSlot = document.getElementById("result-slot");

  provider.addEventListener("change", () => {
    apiKeyRow.classList.toggle("hidden", provider.value === "mock");
  });

  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = document.getElementById("file-input").files[0];
    if (!file) return;

    submitBtn.disabled = true;
    submitBtn.textContent = "分析中...";
    progressCard.classList.remove("hidden");
    resultCard.classList.add("hidden");
    progressBar.style.width = "5%";
    progressLog.textContent = `[1/4] 已选择文件: ${file.name} (${(file.size / 1024).toFixed(1)} KB)\n`;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("use_llm", useLLM.checked ? "1" : "0");
    formData.append("provider", provider.value);
    formData.append("api_key", apiKeyInput.value || "");
    formData.append("base_url", apiBaseInput.value || "");
    formData.append("model", apiModelInput.value || "");

    try {
      progressBar.style.width = "20%";
      progressLog.textContent += "[2/4] 正在提取 6 个维度的特征...\n";
      const res = await fetch("/api/scan", { method: "POST", body: formData });
      progressBar.style.width = "70%";
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      progressLog.textContent += "[3/4] 多模型融合完成\n";
      progressLog.textContent += (data.llm_used
        ? "[4/4] LLM 推理完成\n"
        : "[4/4] 本地规则/集成完成\n");
      progressBar.style.width = "100%";

      renderSingleResult(data);
    } catch (err) {
      progressLog.textContent += "❌ 错误: " + err.message + "\n";
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "开始检测";
    }
  });

  function renderSingleResult(data) {
    const r = data.result;
    const d = r.model_confidence;
    const prob = r.ensemble_probability;
    const isRansom = r.is_ransomware;

    let heroClass = "warn";
    let emoji = "⚠️";
    let verdict = "可疑";
    if (isRansom) { heroClass = "danger"; emoji = "🚨"; verdict = "检测到勒索软件"; }
    else { heroClass = "safe"; emoji = "✅"; verdict = "未检测到勒索软件"; }

    const bars = [
      ["CFG", d.CFG], ["EH", d.EH], ["GI", d.GI],
      ["OP", d.OP],  ["PF", d.PF], ["RB", d.RB],
    ].map(([k, v]) => `
      <div class="bar-row">
        <span class="label">${k}</span>
        <span class="track"><span class="fill" style="width:${(v * 100).toFixed(1)}%"></span></span>
        <span class="val">${v.toFixed(3)}</span>
      </div>
    `).join("");

    resultSlot.innerHTML = `
      <div class="result-hero ${heroClass}">
        <div class="emoji">${emoji}</div>
        <div class="verdict">${verdict}</div>
        <div class="prob">综合恶意概率: ${(prob * 100).toFixed(2)}% · 风险: ${r.risk_level.toUpperCase()} · 误报风险: ${r.false_positive_risk}</div>
        <div style="margin-top:8px;font-size:12px;color:#8c95b5">判决来源: ${r.verdict_source} · 样本哈希: ${r.sample_hash}</div>
      </div>

      <div class="grid2">
        <div class="box">
          <h3>各模型评分</h3>
          ${bars}
        </div>
        <div class="box">
          <h3>推理过程</h3>
          <div class="mono">${escapeHtml(r.reasoning || "(空)")}</div>
        </div>
        <div class="box">
          <h3>LLM 观点</h3>
          <div class="mono">${escapeHtml(r.llm_verdict || "(未启用 LLM / 回退本地)")}</div>
        </div>
        <div class="box">
          <h3>处置建议</h3>
          <p style="margin:0;font-size:13px;">${escapeHtml(r.recommendation)}</p>
        </div>
      </div>

      <div style="margin-top:14px;font-size:12px;color:#8c95b5;">
        文件元信息: ${JSON.stringify(data.meta || {})}
      </div>
    `;
    resultCard.classList.remove("hidden");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  // ---- CSV 批量 ----
  const csvForm = document.getElementById("csv-form");
  const csvResult = document.getElementById("csv-result");
  csvForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = document.getElementById("csv-input").files[0];
    const limit = document.getElementById("csv-limit").value;
    if (!file) return;
    const fd = new FormData();
    fd.append("csv", file);
    fd.append("limit", limit);
    csvResult.innerHTML = `<div class="mono">⏳ 正在处理 ${file.name}...</div>`;
    try {
      const res = await fetch("/api/batch", { method: "POST", body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      csvResult.innerHTML = renderBatchResult(data);
    } catch (err) {
      csvResult.innerHTML = `<div class="mono">❌ 错误: ${err.message}</div>`;
    }
  });

  function renderBatchResult(data) {
    const rows = data.results.map(r => {
      const d = r.model_confidence;
      const cls = r.is_ransomware ? "pos" : "neg";
      const tag = r.is_ransomware
        ? '<span class="tag danger">勒索</span>'
        : '<span class="tag safe">良性</span>';
      return `<tr>
        <td title="${r.sample_hash}">${r.sample_hash.substr(0, 16)}...</td>
        <td>${d.CFG.toFixed(2)}</td><td>${d.EH.toFixed(2)}</td>
        <td>${d.GI.toFixed(2)}</td><td>${d.OP.toFixed(2)}</td>
        <td>${d.PF.toFixed(2)}</td><td>${d.RB.toFixed(2)}</td>
        <td class="${cls}">${(r.ensemble_probability * 100).toFixed(1)}%</td>
        <td>${tag}</td>
      </tr>`;
    }).join("");

    const stats = data.metrics
      ? `<div class="summary-stats">
           <div class="stat"><div class="n">${data.metrics.total}</div><div class="k">总数</div></div>
           <div class="stat"><div class="n">${(data.metrics.accuracy * 100).toFixed(1)}%</div><div class="k">Accuracy</div></div>
           <div class="stat"><div class="n">${(data.metrics.precision * 100).toFixed(1)}%</div><div class="k">Precision</div></div>
           <div class="stat"><div class="n">${(data.metrics.recall * 100).toFixed(1)}%</div><div class="k">Recall</div></div>
           <div class="stat"><div class="n">${(data.metrics.f1 * 100).toFixed(1)}%</div><div class="k">F1</div></div>
         </div>`
      : "";

    const metricsLine = data.metrics
      ? `<p style="color:#8c95b5;font-size:12px;">TP=${data.metrics.tp} · FP=${data.metrics.fp} · TN=${data.metrics.tn} · FN=${data.metrics.fn}</p>`
      : "<p style=\"color:#8c95b5;font-size:12px;\">CSV 中没有 label 列，跳过准确率计算。</p>";

    const downloadBtn = `<a href="${data.download_url}" download="ransomguard_batch_result.csv" style="display:inline-block;margin-top:10px;padding:8px 14px;background:#232a4a;border:1px solid var(--border);border-radius:8px;color:var(--text);text-decoration:none;">⬇️ 下载完整结果 CSV</a>`;

    return `
      ${stats}
      ${metricsLine}
      <div class="table-wrap">
        <table>
          <thead><tr><th>hash</th><th>CFG</th><th>EH</th><th>GI</th><th>OP</th><th>PF</th><th>RB</th><th>恶意概率</th><th>判决</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      ${downloadBtn}
    `;
  }

  // ---- 示例数据 ----
  document.getElementById("demo-btn").addEventListener("click", async () => {
    const demoResult = document.getElementById("demo-result");
    demoResult.innerHTML = `<div class="mono">⏳ 正在加载示例数据...</div>`;
    try {
      const res = await fetch("/api/demo");
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      demoResult.innerHTML = renderBatchResult(data);
    } catch (err) {
      demoResult.innerHTML = `<div class="mono">❌ ${err.message}</div>`;
    }
  });
})();