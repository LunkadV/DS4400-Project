import torch
from xgboost import XGBClassifier

def get_model(params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "eval_metric": "mlogloss",
        }
    return XGBClassifier(**params)
