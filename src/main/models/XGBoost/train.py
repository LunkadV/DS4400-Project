import os
import joblib
import optuna
from sklearn.model_selection import cross_val_score

from src.main.data import load_tabular, split, PROJECT_ROOT
from src.main.models.XGBoost.model import get_model
from src.main.evaluate import cross_validate_model, compute_metrics, plot_confusion_matrix, save_results


RESULTS_PATH = str(PROJECT_ROOT / "results" / "xgboost.json")
MODEL_PATH = str(PROJECT_ROOT / "results" / "xgboost.joblib")


def tune(X_train, y_train):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 500]),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "random_state": 42,
            "eval_metric": "mlogloss",
        }
        model = get_model(params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print(f"Best params: {study.best_params}")
    print(f"Best CV macro F1: {study.best_value:.4f}")
    return study.best_params


if __name__ == "__main__":
    X, y, encoder = load_tabular()
    X_train, X_test, y_train, y_test = split(X, y)

    best_params = tune(X_train, y_train)
    best_params["random_state"] = 42
    best_params["eval_metric"] = "mlogloss"
    import torch
    best_params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(best_params)

    cv_results = cross_validate_model(model, X_train, y_train)
    print(f"CV macro F1: {sum(cv_results['f1_scores']) / len(cv_results['f1_scores']):.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, encoder)
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test micro F1: {metrics['micro_f1']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    plot_confusion_matrix(y_test, y_pred, encoder, title="XGBoost — Confusion Matrix",
                          save_path=os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    results = {
        "xgboost": {
            "best_params": best_params,
            "f1_scores": cv_results["f1_scores"],
            "per_class_f1": cv_results["per_class_f1"],
            "test_metrics": metrics,
        }
    }
    save_results(results, RESULTS_PATH)
