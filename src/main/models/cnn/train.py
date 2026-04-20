import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.main.data import load_spectrograms, split, PROJECT_ROOT
from src.main.models.cnn.model import get_model
from src.main.evaluate import compute_metrics, plot_confusion_matrix, save_results


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
TUNING_EPOCHS = 10
BATCH_SIZE = 32
SPEC_HEIGHT = 128
SPEC_WIDTH = 1296

RESULTS_PATH = str(PROJECT_ROOT / "results" / "cnn.json")
MODEL_PATH = str(PROJECT_ROOT / "results" / "cnn.pt")


class SpectrogramDataset(Dataset):
    """Dataset for loading spectrogram .npy files and their labels."""

    def __init__(self, paths, labels):
        valid = [(p, l) for p, l in zip(paths, labels) if os.path.exists(p)]
        self.paths = [v[0] for v in valid]
        self.labels = np.array([v[1] for v in valid])
        print(f"SpectrogramDataset: {len(self.paths)}/{len(paths)} files found")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            spec = np.load(self.paths[idx])
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

            if spec.shape[1] < SPEC_WIDTH:
                spec = np.pad(spec, ((0, 0), (0, SPEC_WIDTH - spec.shape[1])), mode="constant")
            else:
                spec = spec[:, :SPEC_WIDTH]

            tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, 128, W)
            return tensor, self.labels[idx]
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            return None


def collate_fn(batch):
    """Filter out None entries from failed loads."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    for X_batch, y_batch in loader:
        if X_batch.numel() == 0:
            continue
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
            if X_batch.numel() == 0:
                continue
            preds.extend(model(X_batch.to(DEVICE)).argmax(dim=1).cpu().numpy())
    return np.array(preds)


def cross_validate(paths, labels, params, lr, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_macro_f1 = []
    fold_per_class_f1 = []

    for train_idx, val_idx in skf.split(paths, labels):
        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_dataset = SpectrogramDataset(train_paths, train_labels)
        val_dataset = SpectrogramDataset(val_paths, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        model = get_model(params).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer)

        y_pred = predict(model, val_loader)
        fold_macro_f1.append(f1_score(val_dataset.labels, y_pred, average="macro"))
        fold_per_class_f1.append(f1_score(val_dataset.labels, y_pred, average=None).tolist())

    return {
        "f1_scores": fold_macro_f1,
        "per_class_f1": np.mean(fold_per_class_f1, axis=0).tolist(),
    }


if __name__ == "__main__":
    all_paths, y, encoder = load_spectrograms()
    all_paths = np.array(all_paths)

    indices = np.arange(len(y))
    train_idx, test_idx, y_train, y_test = split(indices, y)

    train_paths = all_paths[train_idx].tolist()
    test_paths = all_paths[test_idx].tolist()

    def objective(trial):
        params = {
            "num_layers": 3,
            "embedding_size": trial.suggest_categorical("embedding_size", [128, 256, 512]),
            "num_classes": 16,
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_f1 = []
        for tr_idx, val_idx in skf.split(train_paths, y_train):
            tr_paths = [train_paths[i] for i in tr_idx]
            val_paths_fold = [train_paths[i] for i in val_idx]
            tr_labels = y_train[tr_idx]
            val_labels = y_train[val_idx]

            tr_dataset = SpectrogramDataset(tr_paths, tr_labels)
            val_dataset = SpectrogramDataset(val_paths_fold, val_labels)
            tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

            m = get_model(params).to(DEVICE)
            opt = optim.Adam(m.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            for _ in range(TUNING_EPOCHS):
                train_epoch(m, tr_loader, criterion, opt)
            fold_f1.append(f1_score(val_dataset.labels, predict(m, val_loader), average="macro"))
        return np.mean(fold_f1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=8)
    print(f"Best params: {study.best_params}")
    print(f"Best CV macro F1: {study.best_value:.4f}")

    best_params = {
        "num_layers": 3,
        "embedding_size": study.best_params["embedding_size"],
        "num_classes": 16,
        "dropout": study.best_params["dropout"],
    }
    best_lr = study.best_params["lr"]

    cv_results = cross_validate(train_paths, y_train, best_params, lr=best_lr)
    print(f"CV macro F1: {sum(cv_results['f1_scores']) / len(cv_results['f1_scores']):.4f}")

    train_dataset = SpectrogramDataset(train_paths, y_train)
    test_dataset = SpectrogramDataset(test_paths, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = get_model(best_params).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{EPOCHS}")

    y_pred = predict(model, test_loader)
    metrics = compute_metrics(test_dataset.labels, y_pred, encoder)
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test micro F1: {metrics['micro_f1']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    plot_confusion_matrix(test_dataset.labels, y_pred, encoder, title="CNN — Confusion Matrix",
                          save_path=os.path.join(os.path.dirname(__file__), 'confusion_matrix.png'))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    results = {
        "cnn": {
            "best_params": best_params,
            "f1_scores": cv_results["f1_scores"],
            "per_class_f1": cv_results["per_class_f1"],
            "test_metrics": metrics,
        }
    }
    save_results(results, RESULTS_PATH)
