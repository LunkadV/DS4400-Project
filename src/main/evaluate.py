import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats


def compute_metrics(y_true, y_pred, encoder):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "per_class_f1": f1_score(y_true, y_pred, average=None).tolist(),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, encoder, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(encoder.classes_)))
    ax.set_yticks(range(len(encoder.classes_)))
    ax.set_xticklabels(encoder.classes_, rotation=45, ha="right")
    ax.set_yticklabels(encoder.classes_)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def cross_validate_model(model, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_macro_f1 = []
    fold_per_class_f1 = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        fold_macro_f1.append(f1_score(y_val, y_pred, average="macro"))
        fold_per_class_f1.append(f1_score(y_val, y_pred, average=None).tolist())

    return {
        "f1_scores": fold_macro_f1,
        "per_class_f1": np.mean(fold_per_class_f1, axis=0).tolist(),
    }


def compare_models(results_dict, save_path=None):
    model_names = list(results_dict.keys())
    means = [np.mean(results_dict[m]["f1_scores"]) for m in model_names]
    stds = [np.std(results_dict[m]["f1_scores"]) for m in model_names]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, means, yerr=stds, capsize=8)
    plt.title("Model Comparison — Macro F1 Score (Mean ± Std across folds)")
    plt.xlabel("Model")
    plt.ylabel("Macro F1 Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(results_dict, encoder, save_path=None):
    model_names = list(results_dict.keys())
    per_class = np.array([results_dict[m]["per_class_f1"] for m in model_names])

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(per_class, interpolation="nearest", cmap=plt.cm.RdYlGn, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(encoder.classes_)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(encoder.classes_, rotation=45, ha="right")
    ax.set_yticklabels(model_names)
    ax.set_title("Per-Class F1 Score by Model")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_ttests(results_dict):
    model_names = list(results_dict.keys())
    ttest_results = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            scores1 = results_dict[m1]["f1_scores"]
            scores2 = results_dict[m2]["f1_scores"]
            t_stat, p_value = stats.ttest_ind(scores1, scores2)
            ttest_results[f"{m1} vs {m2}"] = {
                "mean_diff": np.mean(scores1) - np.mean(scores2),
                "p_value": round(p_value, 4),
            }

    return ttest_results


def save_results(results_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)
