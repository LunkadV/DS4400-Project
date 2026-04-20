import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from src.main.data import load_tabular, split, PROJECT_ROOT
from src.main.models.neuralNetwork.model import get_model
from src.main.evaluate import compute_metrics, plot_confusion_matrix, save_results


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

RESULTS_PATH = str(PROJECT_ROOT / "results" / "neural_network.json")
MODEL_PATH = str(PROJECT_ROOT / "results" / "neural_network.pt")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.extend(model(X_batch.to(DEVICE)).argmax(dim=1).cpu().numpy())
    return np.array(preds)


def cross_validate(X, y, params, lr, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_macro_f1 = []
    fold_per_class_f1 = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)

        model = get_model(params).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer)

        y_pred = predict(model, val_loader)
        fold_macro_f1.append(f1_score(y_val, y_pred, average="macro"))
        fold_per_class_f1.append(f1_score(y_val, y_pred, average=None).tolist())

    return {
        "f1_scores": fold_macro_f1,
        "per_class_f1": np.mean(fold_per_class_f1, axis=0).tolist(),
    }


if __name__ == "__main__":
    X, y, encoder = load_tabular()
    X_train, X_test, y_train, y_test = split(X, y)

    # Normalize features — required for MLP unlike tree-based models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def objective(trial):
        params = {
            "input_size": 518,
            "hidden_sizes": trial.suggest_categorical("hidden_sizes", [[256], [512, 256], [512, 256, 128]]),
            "num_classes": 16,
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_f1 = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE)
            m = get_model(params).to(DEVICE)
            opt = optim.Adam(m.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            for _ in range(EPOCHS):
                train_epoch(m, train_loader, criterion, opt)
            fold_f1.append(f1_score(y_val, predict(m, val_loader), average="macro"))
        return np.mean(fold_f1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print(f"Best params: {study.best_params}")
    print(f"Best CV macro F1: {study.best_value:.4f}")

    best_params = {
        "input_size": 518,
        "hidden_sizes": study.best_params["hidden_sizes"],
        "num_classes": 16,
        "dropout": study.best_params["dropout"],
    }
    best_lr = study.best_params["lr"]

    cv_results = cross_validate(X_train, y_train, best_params, lr=study.best_params["lr"])
    print(f"CV macro F1: {sum(cv_results['f1_scores']) / len(cv_results['f1_scores']):.4f}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=BATCH_SIZE)

    model = get_model(best_params).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{EPOCHS}")

    y_pred = predict(model, test_loader)
    metrics = compute_metrics(y_test, y_pred, encoder)
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test micro F1: {metrics['micro_f1']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    plot_confusion_matrix(y_test, y_pred, encoder, title="Neural Network — Confusion Matrix",
                          save_path=os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'))

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    results = {
        "neural_network": {
            "best_params": best_params,
            "f1_scores": cv_results["f1_scores"],
            "per_class_f1": cv_results["per_class_f1"],
            "test_metrics": metrics,
        }
    }
    save_results(results, RESULTS_PATH)
