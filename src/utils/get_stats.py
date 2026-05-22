from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PATH = Path("data/processed/final_intersection_dataset.csv")
MIN_VOTES_PER_POST = 5
MIN_VOTES_PER_USER = 10


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df["vote"]  = pd.to_numeric(df["vote"],  errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["vote"].isin([1, -1]) & df["label"].isin([1, -1])]
    df = df.dropna(subset=["username", "item_id", "vote", "label"])
    df = df.drop_duplicates(subset=["username", "item_id"])
    return df.reset_index(drop=True)


def density_filter(df):
    f = df.copy()
    for _ in range(100):
        prev = len(f)

        post_counts = f["item_id"].value_counts()
        f = f[f["item_id"].isin(post_counts[post_counts >= MIN_VOTES_PER_POST].index)]

        user_counts = f["username"].value_counts()
        f = f[f["username"].isin(user_counts[user_counts >= MIN_VOTES_PER_USER].index)]

        if len(f) == prev:
            break
    return f


def row(label, raw, filt):
    return f"  {label:<10} {raw:>10,}  ->  {filt:>10,}    {100*filt/raw:.1f}% retained"


def describe_votes(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
    }


def print_distribution_stats(name, raw_series, filt_series):
    raw_stats = describe_votes(raw_series)
    filt_stats = describe_votes(filt_series)

    print(f"\n{name}:")
    print(f"  {'stat':<8} {'before':>10} {'after':>10}")
    print(f"  {'-'*32}")
    for k in ["mean", "median", "min", "max"]:
        print(f"  {k:<8} {raw_stats[k]:>10.2f} {filt_stats[k]:>10.2f}")


# 🆕 NEW: outcome distribution
def print_outcome_stats(name, raw_series, filt_series):
    raw_counts = raw_series.value_counts().sort_index()
    filt_counts = filt_series.value_counts().sort_index()

    print(f"\n{name} distribution:")
    print(f"  {'value':<8} {'before':>10} {'after':>10}")
    print(f"  {'-'*32}")

    for val in [-1, 1]:
        r = raw_counts.get(val, 0)
        f = filt_counts.get(val, 0)
        perc = (f / r * 100) if r > 0 else 0
        print(f"  {val:<8} {r:>10,} {f:>10,}   ({perc:.1f}%)")


# ---- RUN ----
raw = load(INPUT_PATH)
fil = density_filter(raw)

print(f"\nFilter: >= {MIN_VOTES_PER_POST} votes/post, >= {MIN_VOTES_PER_USER} votes/user\n")

# --- base counts ---
print(f"{'':12} {'before':>10} {'after':>10}")
print(f"{'-'*36}")
print(row("votes",  len(raw),                 len(fil)))
print(row("posts",  raw['item_id'].nunique(), fil['item_id'].nunique()))
print(row("users",  raw['username'].nunique(), fil['username'].nunique()))

# --- outcome stats ---
print_outcome_stats("Label", raw["label"], fil["label"])
print_outcome_stats("Vote", raw["vote"], fil["vote"])

# --- votes per user ---
raw_user_votes = raw.groupby("username").size()
fil_user_votes = fil.groupby("username").size()

print_distribution_stats("Votes per user", raw_user_votes, fil_user_votes)

# --- votes per post ---
raw_post_votes = raw.groupby("item_id").size()
fil_post_votes = fil.groupby("item_id").size()

print_distribution_stats("Votes per post", raw_post_votes, fil_post_votes)

# --- density ---
def density(df):
    return len(df) / (df["username"].nunique() * df["item_id"].nunique())

print("\nMatrix density (user-item):")
print(f"  before: {density(raw):.6f}")
print(f"  after : {density(fil):.6f}")

# --- sparsity reduction ---
print("\nSparsity reduction:")
print(f"  removed votes: {len(raw) - len(fil):,}")
print(f"  removed users: {raw['username'].nunique() - fil['username'].nunique():,}")
print(f"  removed posts: {raw['item_id'].nunique() - fil['item_id'].nunique():,}")

print()