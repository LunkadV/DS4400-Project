import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import pandas as pd

from src.main.data import PROJECT_ROOT, load_tabular, FMA_METADATA_PATH
from src.main.evaluate import load_results, compare_models, plot_per_class_f1, run_ttests, save_results


RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "comparison"

MODEL_FILES = {
    "Random Forest": RESULTS_DIR / "random_forest.json",
    "XGBoost": RESULTS_DIR / "xgboost.json",
    "Neural Network": RESULTS_DIR / "neural_network.json",
    "CNN": RESULTS_DIR / "cnn.json",
}


def load_all_results():
    merged = {}
    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping {name}")
            continue
        data = load_results(str(path))
        # Each file has a single top-level key — extract the inner dict
        inner = list(data.values())[0]
        merged[name] = inner
    return merged


def plot_feature_importance(save_path=None):
    rf_path = RESULTS_DIR / "random_forest.joblib"
    if not os.path.exists(rf_path):
        print("Warning: Random Forest model not found, skipping feature importance")
        return

    model = joblib.load(rf_path)
    features = pd.read_csv(FMA_METADATA_PATH / "features.csv", index_col=0, header=[0, 1, 2])
    labels = [f"{c[0]}_{c[1]}_{c[2]}" for c in features.columns]

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-20:]
    top_labels = [labels[i] for i in top_idx]
    top_importances = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_labels)), top_importances, color="steelblue")
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest — Top 20 Feature Importances")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_ttest_heatmap(results, save_path=None):
    from scipy import stats

    model_names = list(results.keys())
    n = len(model_names)
    p_matrix = np.ones((n, n))
    diff_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                scores_i = results[model_names[i]]["f1_scores"]
                scores_j = results[model_names[j]]["f1_scores"]
                _, p_value = stats.ttest_ind(scores_i, scores_j)
                p_matrix[i][j] = p_value
                diff_matrix[i][j] = np.mean(scores_i) - np.mean(scores_j)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(p_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1)
    plt.colorbar(im, ax=ax, label="p-value")

    # Annotate cells with p-value and mean diff
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=10)
            else:
                p = p_matrix[i][j]
                diff = diff_matrix[i][j]
                sig = "*" if p < 0.05 else ""
                ax.text(j, i, f"p={p:.4f}{sig}\n({diff:+.3f})",
                        ha="center", va="center", fontsize=8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_yticklabels(model_names)
    ax.set_title("Pairwise T-Test Results (CV Macro F1)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_hyperparameter_table(results, save_path=None):
    rows = []
    for name, data in results.items():
        params = data.get("best_params", {})
        for key, value in params.items():
            rows.append({"Model": name, "Hyperparameter": key, "Value": str(value)})
    df = pd.DataFrame(rows)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = load_all_results()
    if not results:
        print("No results found. Train models first.")
        exit(1)

    print(f"Loaded results for: {', '.join(results.keys())}")

    _, _, encoder = load_tabular()

    compare_models(results, save_path=str(OUTPUT_DIR / "model_comparison.png"))
    print("Saved model_comparison.png")

    plot_per_class_f1(results, encoder, save_path=str(OUTPUT_DIR / "per_class_f1.png"))
    print("Saved per_class_f1.png")

    ttest_results = run_ttests(results)
    save_results(ttest_results, str(OUTPUT_DIR / "ttests.json"))
    print("Saved ttests.json")

    plot_ttest_heatmap(results, save_path=str(OUTPUT_DIR / "ttest_heatmap.png"))
    print("Saved ttest_heatmap.png")

    plot_feature_importance(save_path=str(OUTPUT_DIR / "feature_importance.png"))
    print("Saved feature_importance.png")

    save_hyperparameter_table(results, save_path=str(OUTPUT_DIR / "hyperparameters.csv"))
    print("Saved hyperparameters.csv")

    for name, data in results.items():
        test = data.get("test_metrics", {})
        print(f"\n{name}:")
        print(f"  Test macro F1: {test.get('macro_f1', 'N/A'):.4f}")
        print(f"  Test accuracy: {test.get('accuracy', 'N/A'):.4f}")

    print(f"\nComparison results saved to {OUTPUT_DIR}")
