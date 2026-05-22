"""
step2_compare.py
================
Loads metrics.json from all step2 runs and produces:
  1. A minimal terminal summary
  2. A self-contained HTML report with bar charts and a full table
     → opens automatically in the browser

Usage
-----
    from step2_compare import compare_all
    compare_all(Path("results/step2"), platform="reddit")

Or directly:
    python step2_compare.py
"""

from __future__ import annotations

import json
import logging
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

METRICS_TO_SHOW = [
    "macro_f1", "roc_auc",
    "f1_pos", "f1_neg",
    "macro_precision", "macro_recall",
    "precision_pos", "recall_pos",
    "precision_neg", "recall_neg",
]

METRIC_LABELS = {
    "macro_f1":        "Macro F1",
    "roc_auc":         "ROC AUC",
    "f1_pos":          "F1 (keep)",
    "f1_neg":          "F1 (remove)",
    "macro_precision": "Macro Prec.",
    "macro_recall":    "Macro Rec.",
    "precision_pos":   "Prec. (keep)",
    "recall_pos":      "Rec. (keep)",
    "precision_neg":   "Prec. (remove)",
    "recall_neg":      "Rec. (remove)",
}

RUN_LABELS = {
    "reddit_cn":              "CN MF  (+votes)",
    "reddit_cn_inverted":     "CN MF  (−votes)",
    "wikipedia":              "CN MF  Wikipedia",
    "BL1_net_score":          "BL1 Net score",
    "BL2_weighted_net_score": "BL2 Weighted",
    "BL3_subreddit_adjusted": "BL3 Subreddit-adj.",
}


# ── data loading ──────────────────────────────────────────────────────────────
def load_metrics(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not load %s: %s", path, e)
        return None


def collect_runs(results_dir: Path, platform: Optional[str]) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(results_dir.rglob("metrics.json")):
        rel      = metrics_path.parent.relative_to(results_dir)
        run_name = "/".join(rel.parts)
        if platform and platform not in run_name:
            continue
        m = load_metrics(metrics_path)
        if m is None:
            continue
        label = RUN_LABELS.get(rel.parts[-1], rel.parts[-1])
        row   = {"run": run_name, "label": label}
        for key in METRICS_TO_SHOW:
            val = m.get(key, m.get(f"{key}_mean", float("nan")))
            row[key] = round(float(val), 4)
        row["alpha"]    = m.get("alpha", "—")
        thr = m.get("threshold", m.get("threshold_mean"))
        row["threshold"] = round(float(thr), 4) if isinstance(thr, (int, float)) else "—"
        polarity = m.get("polarity_correct", m.get("polarity_correct_mean"))
        row["polarity"] = "✓" if polarity else "✗"
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df


# terminal summary
def print_summary(df: pd.DataFrame) -> None:
    cols = ["label", "macro_f1", "roc_auc", "f1_pos", "f1_neg", "polarity"]
    pad  = [40, 12, 12, 12, 12, 8]
    header = "".join(c.ljust(p) for c, p in zip(cols, pad))
    sep    = "─" * sum(pad)
    print(f"\n{sep}\n{header}\n{sep}")
    for _, row in df.iterrows():
        line = (
            str(row["label"]).ljust(pad[0])
            + f"{row['macro_f1']:.4f}".ljust(pad[1])
            + f"{row['roc_auc']:.4f}".ljust(pad[2])
            + f"{row['f1_pos']:.4f}".ljust(pad[3])
            + f"{row['f1_neg']:.4f}".ljust(pad[4])
            + str(row["polarity"]).ljust(pad[5])
        )
        print(line)
    print(sep + "\n")


# ── HTML report ───────────────────────────────────────────────────────────────
def _bar_chart(df: pd.DataFrame, metric: str, title: str, colors: dict) -> str:
    rows_html = ""
    max_val   = df[metric].max() if not df[metric].isna().all() else 1.0
    for _, row in df.iterrows():
        val   = row[metric]
        pct   = round(100 * val / max_val, 1) if pd.notna(val) and max_val > 0 else 0
        color = colors[row["label"]]
        rows_html += f"""
        <div class="bar-row">
          <div class="bar-label">{row['label']}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct}%;background:{color}"></div>
            <span class="bar-value">{val:.4f}</span>
          </div>
        </div>"""
    return f"""
    <div class="chart-card">
      <h3>{title}</h3>
      <div class="bar-chart">{rows_html}</div>
    </div>"""


def build_html(df: pd.DataFrame, platform: str, results_dir: Path) -> str:
    palette = [
        "#4f8ef7","#f7874f","#4fc98e","#f7d24f",
        "#c44ff7","#f74f6b","#4ff7f0","#a0a0a0",
    ]
    colors = {row["label"]: palette[i % len(palette)] for i, row in df.iterrows()}

    primary = [
        ("macro_f1", "Macro F1"),
        ("roc_auc",  "ROC AUC"),
        ("f1_pos",   "F1 — keep (+1)"),
        ("f1_neg",   "F1 — remove (−1)"),
    ]
    charts_html = "".join(_bar_chart(df, m, t, colors) for m, t in primary)

    # full table
    table_cols = ["label"] + METRICS_TO_SHOW + ["alpha", "threshold", "polarity"]
    table_cols = [c for c in table_cols if c in df.columns]

    thead = "<tr>" + "".join(
        f"<th>{METRIC_LABELS.get(c, c.replace('_',' ').title())}</th>"
        for c in table_cols
    ) + "</tr>"

    tbody = ""
    for i, row in df.iterrows():
        tr_class = "best-row" if i == 0 else ""
        cells = ""
        for c in table_cols:
            val = row[c]
            cells += f"<td>{val:.4f}</td>" if isinstance(val, float) else f"<td>{val}</td>"
        tbody += f"<tr class='{tr_class}'>{cells}</tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model Comparison — {platform}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  :root{{
    --bg:#0f1117;--surface:#181c27;--border:#272d3d;
    --text:#e2e8f0;--muted:#64748b;--accent:#4f8ef7;--best:#1a2e1a;
  }}
  body{{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',sans-serif;
        font-size:14px;padding:40px 48px;line-height:1.6}}
  header{{border-bottom:1px solid var(--border);padding-bottom:24px;margin-bottom:40px}}
  header h1{{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;
             color:var(--accent);letter-spacing:-0.5px}}
  header p{{color:var(--muted);font-size:13px;margin-top:6px;font-family:'IBM Plex Mono',monospace}}
  h2{{font-size:12px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
      color:var(--muted);margin-bottom:20px}}
  h3{{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;color:var(--muted);
      text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}}
  section{{margin-bottom:52px}}
  .charts-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}}
  .chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px 24px}}
  .bar-chart{{display:flex;flex-direction:column;gap:10px}}
  .bar-row{{display:grid;grid-template-columns:180px 1fr;align-items:center;gap:12px}}
  .bar-label{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--text);
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:right}}
  .bar-track{{background:#1e2435;border-radius:3px;height:22px;position:relative;display:flex;align-items:center}}
  .bar-fill{{height:100%;border-radius:3px;min-width:2px}}
  .bar-value{{position:absolute;right:8px;font-family:'IBM Plex Mono',monospace;
              font-size:11px;color:var(--text);font-weight:600}}
  .table-wrap{{overflow-x:auto;border:1px solid var(--border);border-radius:8px}}
  table{{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:12px}}
  thead tr{{background:#12151e;border-bottom:1px solid var(--border)}}
  th{{padding:10px 14px;text-align:left;color:var(--muted);font-weight:600;
      letter-spacing:.05em;white-space:nowrap;text-transform:uppercase;font-size:11px}}
  td{{padding:9px 14px;color:var(--text);border-bottom:1px solid var(--border);white-space:nowrap}}
  tbody tr:last-child td{{border-bottom:none}}
  tbody tr:hover{{background:#1a1f2e}}
  .best-row td{{background:var(--best);font-weight:600}}
  .best-row td:first-child{{border-left:3px solid #4fc98e}}
  .note{{color:var(--muted);font-size:12px;margin-bottom:14px}}
</style>
</head>
<body>
<header>
  <h1>model_comparison / {platform}</h1>
  <p>results dir: {results_dir} &nbsp;·&nbsp; {len(df)} runs loaded</p>
</header>

<section>
  <h2>Primary Metrics</h2>
  <div class="charts-grid">{charts_html}</div>
</section>

<section>
  <h2>Full Results Table</h2>
  <p class="note">Sorted by Macro F1 descending &nbsp;·&nbsp; Best row highlighted</p>
  <div class="table-wrap">
    <table>
      <thead>{thead}</thead>
      <tbody>{tbody}</tbody>
    </table>
  </div>
</section>
</body>
</html>"""
    return html


# ── main ──────────────────────────────────────────────────────────────────────
def compare_all(
    results_dir:  Path,
    platform:     Optional[str] = None,
    open_browser: bool = True,
) -> pd.DataFrame:
    df = collect_runs(results_dir, platform)
    if df.empty:
        log.warning("No runs found under %s (platform=%s)", results_dir, platform)
        return df

    label = platform or "all"
    print_summary(df)

    html     = build_html(df, label, results_dir)
    out_path = results_dir.resolve() / f"comparison_{label}.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("HTML report → %s", out_path)

    if open_browser:
        webbrowser.open(out_path.as_uri())

    csv_path = results_dir / f"comparison_{label}.csv"
    df.to_csv(csv_path, index=False)
    log.info("CSV  → %s", csv_path)

    return df


if __name__ == "__main__":
    compare_all(Path("results/step2"), platform="reddit")
    compare_all(Path("results/step2"), platform="wikipedia")