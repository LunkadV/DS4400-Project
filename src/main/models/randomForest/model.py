from sklearn.ensemble import RandomForestClassifier

def get_model(params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
    return RandomForestClassifier(**params)
