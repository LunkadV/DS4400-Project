import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.main.data import load_tabular, split, PROJECT_ROOT
from src.main.models.neuralNetwork.model import get_model as get_mlp_model
from src.main.evaluate import compute_metrics, save_results, plot_confusion_matrix, load_results


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT_ROOT / "results" / "novel"
MODEL_NAMES = ["Random Forest", "XGBoost", "MLP"]


def load_base_models():
    rf = joblib.load(PROJECT_ROOT / "results" / "random_forest.joblib")

    xgb = joblib.load(PROJECT_ROOT / "results" / "xgboost.joblib")

    with open(PROJECT_ROOT / "results" / "neural_network.json") as f:
        mlp_params = json.load(f)["neural_network"]["best_params"]
    mlp = get_mlp_model(mlp_params)
    mlp.load_state_dict(torch.load(PROJECT_ROOT / "results" / "neural_network.pt", map_location=DEVICE))
    mlp.to(DEVICE)
    mlp.eval()

    return rf, xgb, mlp


def get_base_probabilities(rf, xgb, mlp, X, scaler):
    rf_prob = rf.predict_proba(X)
    xgb_prob = xgb.predict_proba(X)

    X_scaled = scaler.transform(X)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.zeros(len(X_scaled))),
        batch_size=64
    )
    mlp_probs = []
    with torch.no_grad():
        for X_batch, _ in loader:
            logits = mlp(X_batch.to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            mlp_probs.append(probs.cpu().numpy())
    mlp_prob = np.vstack(mlp_probs)

    return [rf_prob, xgb_prob, mlp_prob]


def load_cv_per_genre_f1():
    """Load per-class F1 from cross-validation results to avoid overfitting."""
    result_files = [
        PROJECT_ROOT / "results" / "random_forest.json",
        PROJECT_ROOT / "results" / "xgboost.json",
        PROJECT_ROOT / "results" / "neural_network.json",
    ]
    per_genre_f1 = []
    for path in result_files:
        data = load_results(str(path))
        inner = list(data.values())[0]
        per_genre_f1.append(inner["per_class_f1"])
    return np.array(per_genre_f1)


def weighted_soft_vote(probabilities, per_genre_f1):
    n_samples = probabilities[0].shape[0]
    n_classes = probabilities[0].shape[1]
    blended = np.zeros((n_samples, n_classes))

    for class_idx in range(n_classes):
        # Weights for this class: each model's F1 on this genre
        weights = per_genre_f1[:, class_idx]
        weight_sum = weights.sum()
        if weight_sum == 0:
            weights = np.ones(len(MODEL_NAMES)) / len(MODEL_NAMES)
        else:
            weights = weights / weight_sum

        for m in range(len(MODEL_NAMES)):
            blended[:, class_idx] += weights[m] * probabilities[m][:, class_idx]

    return blended.argmax(axis=1)


def plot_per_genre_f1(per_genre_f1, encoder, save_path=None):
    genres = encoder.classes_
    x = np.arange(len(genres))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(14, 8))
    for i, model_name in enumerate(MODEL_NAMES):
        ax.bar(x + i * width, per_genre_f1[i], width, label=model_name, color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(genres, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Genre F1 Score by Model (Cross-Validation)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_routing_map(per_genre_f1, encoder, save_path=None):
    genres = encoder.classes_
    best_models = np.argmax(per_genre_f1, axis=0)
    colors_map = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(genres)), per_genre_f1.max(axis=0), color=[colors_map[b] for b in best_models])
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels(genres, fontsize=9)
    ax.set_xlabel("Best F1 Score")
    ax.set_title("Per-Genre Best Model Assignment")

    # Add model name labels on bars
    for i, (score, model_idx) in enumerate(zip(per_genre_f1.max(axis=0), best_models)):
        ax.text(score + 0.01, i, MODEL_NAMES[model_idx], va="center", fontsize=8)

    ax.set_xlim(0, 1)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Use medium_only to match the subset the base models were trained on
    X, y, encoder = load_tabular(subset="medium_only")
    X_train, X_test, y_train, y_test = split(X, y)
    n_classes = len(encoder.classes_)

    # Fit scaler on training data (needed for MLP)
    scaler = StandardScaler()
    scaler.fit(X_train)

    print("Loading base models...")
    rf, xgb, mlp = load_base_models()

    per_genre_f1 = load_cv_per_genre_f1()

    print("Per-genre best model (from CV):")
    for genre_idx, genre_name in enumerate(encoder.classes_):
        scores = per_genre_f1[:, genre_idx]
        best = np.argmax(scores)
        print(f"  {genre_name:<25} -> {MODEL_NAMES[best]} "
              f"(RF={scores[0]:.3f}, XGB={scores[1]:.3f}, MLP={scores[2]:.3f})")

    print("\nGenerating base model predictions on test set...")
    test_probs = get_base_probabilities(rf, xgb, mlp, X_test, scaler)

    weighted_pred = weighted_soft_vote(test_probs, per_genre_f1)
    weighted_metrics = compute_metrics(y_test, weighted_pred, encoder)

    base_metrics = {}
    for name, probs in zip(MODEL_NAMES, test_probs):
        pred = probs.argmax(axis=1)
        base_metrics[name] = compute_metrics(y_test, pred, encoder)

    print(f"\n{'Model':<30} {'Macro F1':>10} {'Accuracy':>10}")
    print("-" * 52)
    for name in MODEL_NAMES:
        m = base_metrics[name]
        print(f"{name:<30} {m['macro_f1']:>10.4f} {m['accuracy']:>10.4f}")
    print("-" * 52)
    print(f"{'Weighted Soft Vote':<30} {weighted_metrics['macro_f1']:>10.4f} {weighted_metrics['accuracy']:>10.4f}")

    plot_per_genre_f1(per_genre_f1, encoder, save_path=str(RESULTS_DIR / "per_genre_f1_by_model.png"))
    print("\nSaved per_genre_f1_by_model.png")

    plot_routing_map(per_genre_f1, encoder, save_path=str(RESULTS_DIR / "routing_map.png"))
    print("Saved routing_map.png")

    plot_confusion_matrix(y_test, weighted_pred, encoder, title="Weighted Soft Vote — Confusion Matrix",
                          save_path=str(RESULTS_DIR / "confusion_matrix.png"))
    print("Saved confusion_matrix.png")

    results = {
        "weighted_soft_vote": {
            "base_metrics": base_metrics,
            "weighted_soft_vote_metrics": weighted_metrics,
            "per_genre_best_model": {
                encoder.classes_[g]: MODEL_NAMES[np.argmax(per_genre_f1[:, g])]
                for g in range(n_classes)
            },
        }
    }
    save_results(results, str(RESULTS_DIR / "routing_results.json"))
    print(f"\nResults saved to {RESULTS_DIR}")
