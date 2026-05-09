from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

METRICS = [
    "macro_f1", "roc_auc",
    "f1_pos", "f1_neg",
    "macro_precision", "macro_recall",
    "precision_pos", "recall_pos",
    "precision_neg", "recall_neg",
]

RUN_LABELS = {
    "reddit_cn": "CN+",
    "reddit_cn_inverted": "CN-",
    "BL1_net_score": "Net",
    "BL2_weighted_net_score": "WNet",
    "BL3_subreddit_adjusted": "SubAdj",
}

COLORS = {
    "CN+": "#0647b8",
    "CN-": "#ee1d40",
    "Net": "#0da25a",
    "WNet": "#d8b226",
    "SubAdj": "#b541e6",
}


# ── load ────────────────────────────────────────────
def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def collect(results_dir: Path):
    rows = []

    for p in results_dir.rglob("metrics.json"):
        run = "/".join(p.parent.relative_to(results_dir).parts)

        if "wikipedia" in run:
            continue

        m = load_metrics(p)

        label_key = p.parent.name
        label = RUN_LABELS.get(label_key, label_key)

        row = {"run": label}

        for k in METRICS:
            row[k] = m.get(k, None)
            row[k + "_std"] = m.get(k + "_std", None)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("macro_f1", ascending=False)
    return df


# ── print ────────────────────────────────────────────
def print_summary(df):
    print("\n=== SUMMARY (no wikipedia) ===\n")
    print(df[["run", "macro_f1", "roc_auc", "f1_pos", "f1_neg"]])


# ── plot with CI ─────────────────────────────────────
def plot_metric(df, metric):
    plt.figure(figsize=(8,4))

    colors = [COLORS.get(r, "gray") for r in df["run"]]

    y = df[metric]
    yerr = df.get(metric + "_std")

    # fallback se std non presente
    if yerr is None or yerr.isna().all():
        yerr = None

    plt.bar(
        df["run"],
        y,
        color=colors,
        yerr=yerr,
        capsize=5,          # 👈 CI bars
        alpha=0.9
    )

    plt.title(metric)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)

    offset = 0.03

    for i, v in enumerate(y):
        if pd.notna(v):
            err = 0
            if yerr is not None and not pd.isna(yerr.iloc[i]):
                err = yerr.iloc[i]

            plt.text(
                i,
                v + err + offset,   # 👈 sposta sopra CI
                f"{v:.2f}",
                ha="center",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()


# ── main ────────────────────────────────────────────
def compare(results_dir: Path):
    df = collect(results_dir)

    if df.empty:
        print("No data found")
        return df

    print_summary(df)

    # main plots with CI
    for metric in ["macro_f1", "roc_auc", "f1_pos", "f1_neg"]:
        plot_metric(df, metric)

    return df


# ── run ─────────────────────────────────────────────
compare(Path("results/step2"))