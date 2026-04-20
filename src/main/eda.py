import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.main.data import PROJECT_ROOT, FMA_METADATA_PATH, _load_tracks


RESULTS_DIR = PROJECT_ROOT / "results" / "eda"


def genre_distribution():
    tracks = _load_tracks()
    genre_counts = tracks["track", "genre_top"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    genre_counts.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Genre")
    ax.set_title("Genre Distribution — FMA Medium")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "genre_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved genre_distribution.png")


def pca_variance():
    features = pd.read_csv(FMA_METADATA_PATH / "features.csv", index_col=0, header=[0, 1, 2])
    tracks = _load_tracks()
    features = features[features.index.isin(tracks.index)]

    X = features.values.astype(np.float32)
    X = StandardScaler().fit_transform(X)

    pca = PCA().fit(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100

    # Find components needed for 90% variance
    n_90 = np.argmax(cumulative >= 90) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumulative) + 1), cumulative, color="steelblue", linewidth=2)
    ax.axhline(y=90, color="red", linestyle="--", alpha=0.7, label=f"90% variance ({n_90} components)")
    ax.axvline(x=n_90, color="red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("PCA — Cumulative Variance")
    ax.legend()
    ax.set_xlim(1, len(cumulative))
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pca_variance.png", bbox_inches="tight")
    plt.close()
    print(f"Saved pca_variance.png (90% variance at {n_90} components)")


def feature_boxplots():
    features = pd.read_csv(FMA_METADATA_PATH / "features.csv", index_col=0, header=[0, 1, 2])
    tracks = _load_tracks()
    features = features[features.index.isin(tracks.index)]
    tracks = tracks.loc[features.index]

    genres = tracks["track", "genre_top"]

    # Top features by between-genre variance ratio, one per feature group
    selected = {
        "MFCC 4 Mean": ("mfcc", "mean", "04"),
        "MFCC 1 Mean": ("mfcc", "mean", "01"),
        "Spectral Rolloff Skew": ("spectral_rolloff", "skew", "01"),
        "Spectral Contrast 7 Max": ("spectral_contrast", "max", "07"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Sort genres by frequency for consistent ordering
    genre_order = genres.value_counts().index.tolist()

    for i, (label, col) in enumerate(selected.items()):
        data = pd.DataFrame({"value": features[col].values, "genre": genres.values})
        boxplot_data = [data[data["genre"] == g]["value"].dropna().values for g in genre_order]
        bp = axes[i].boxplot(boxplot_data, tick_labels=genre_order, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
        axes[i].set_xticklabels(genre_order, rotation=45, ha="right", fontsize=8)
        axes[i].set_title(label)
        axes[i].set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_boxplots.png", bbox_inches="tight")
    plt.close()
    print("Saved feature_boxplots.png")


def missing_values():
    features = pd.read_csv(FMA_METADATA_PATH / "features.csv", index_col=0, header=[0, 1, 2])
    tracks = _load_tracks()
    features = features[features.index.isin(tracks.index)]

    missing = features.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found in features.")
        summary = pd.DataFrame({"missing_count": [0], "note": ["No missing values"]})
    else:
        summary = missing.reset_index()
        summary.columns = ["feature_group", "feature_type", "feature_stat", "missing_count"]

    summary.to_csv(RESULTS_DIR / "missing_values.csv", index=False)
    print("Saved missing_values.csv")


def feature_stats_by_genre():
    tracks = _load_tracks()
    features = pd.read_csv(FMA_METADATA_PATH / "features.csv", index_col=0, header=[0, 1, 2])
    features = features[features.index.isin(tracks.index)]
    tracks = tracks.loc[features.index]

    genres = tracks["track", "genre_top"]

    # Use mean statistics across feature groups
    mean_cols = [col for col in features.columns if col[2] == "mean"]
    subset = features[mean_cols].copy()
    subset.columns = [f"{c[0]}_{c[1]}" for c in mean_cols]
    subset["genre"] = genres.values

    stats = subset.groupby("genre").mean()
    stats.to_csv(RESULTS_DIR / "feature_stats.csv")
    print("Saved feature_stats.csv")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    genre_distribution()
    pca_variance()
    feature_boxplots()
    missing_values()
    feature_stats_by_genre()
    print(f"\nEDA results saved to {RESULTS_DIR}")
