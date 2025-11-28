#!/usr/bin/env python3
"""Train classical + neural prototypes for Notebook 09."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import LinearSVC
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def find_project_root(marker: str = ".git") -> Path:
    path = Path.cwd()
    for parent in [path, *path.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError("No se encontró la raíz del proyecto")


def boundaries_to_segments(y_boundaries: np.ndarray) -> np.ndarray:
    y_boundaries = np.asarray(y_boundaries, dtype=int)
    n_sentences = len(y_boundaries) + 1
    segments = np.zeros(n_sentences, dtype=int)
    current = 0
    segments[0] = current
    for i, val in enumerate(y_boundaries):
        if val:
            current += 1
        segments[i + 1] = current
    return segments


def pk(reference: np.ndarray, hypothesis: np.ndarray, k: int) -> float:
    reference = np.asarray(reference)
    hypothesis = np.asarray(hypothesis)
    if reference.shape != hypothesis.shape:
        raise ValueError("reference y hypothesis deben tener la misma longitud")
    if k <= 0 or len(reference) <= k:
        return 0.0
    disagreements = 0
    total = 0
    for i in range(len(reference) - k):
        same_ref = reference[i] == reference[i + k]
        same_hyp = hypothesis[i] == hypothesis[i + k]
        disagreements += int(same_ref != same_hyp)
        total += 1
    return disagreements / total if total else 0.0


def windowdiff(reference: np.ndarray, hypothesis: np.ndarray, k: int) -> float:
    reference = np.asarray(reference, dtype=int)
    hypothesis = np.asarray(hypothesis, dtype=int)
    if reference.shape != hypothesis.shape:
        raise ValueError("reference y hypothesis deben tener la misma longitud")
    n_sentences = len(reference) + 1
    if k <= 0 or n_sentences <= k:
        return 0.0
    total = n_sentences - k
    errors = 0
    for start in range(total):
        end = start + k - 1
        ref_count = reference[start:end].sum()
        hyp_count = hypothesis[start:end].sum()
        errors += int(ref_count != hyp_count)
    return errors / total if total else 0.0


def evaluate_predictions(df_boundaries: pd.DataFrame, y_pred: np.ndarray) -> dict:
    df = df_boundaries.sort_values(["level", "doc_id", "boundary_id"]).reset_index(drop=True)
    y_true = df["y"].to_numpy().astype(int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    doc_records = []
    seg_lengths = []
    for (_, _doc_id), group in df.groupby(["level", "doc_id"]):
        idx = group.index.to_numpy()
        y_doc = group["y"].to_numpy()
        doc_records.append((idx, y_doc))
        seg = boundaries_to_segments(y_doc)
        _, counts = np.unique(seg, return_counts=True)
        seg_lengths.extend(counts.tolist())
    mean_segment_length = float(np.mean(seg_lengths)) if seg_lengths else 1.0
    k = max(1, int(round(mean_segment_length / 2)))
    pk_scores = []
    wd_scores = []
    for idx, y_doc in doc_records:
        y_pred_doc = y_pred[idx]
        seg_true = boundaries_to_segments(y_doc)
        seg_pred = boundaries_to_segments(y_pred_doc)
        pk_scores.append(pk(seg_true, seg_pred, k=k))
        wd_scores.append(windowdiff(y_doc, y_pred_doc, k=k))
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "pk": float(np.mean(pk_scores)) if pk_scores else np.nan,
        "windowdiff": float(np.mean(wd_scores)) if wd_scores else np.nan,
        "k_window": k,
    }


def sklearn_run(model, X_train, y_train, X_val, y_val, bounds_train, bounds_val):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    metrics = {
        "metrics_train": evaluate_predictions(bounds_train, y_train_pred),
        "metrics_val": evaluate_predictions(bounds_val, y_val_pred),
        "classification_report": classification_report(y_val, y_val_pred, digits=3),
    }
    return metrics, y_val_pred


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TinyLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1]).squeeze(-1)


class TinyCNN(nn.Module):
    def __init__(self, input_dim: int, num_filters: int = 128):
        super().__init__()
        self.conv3 = nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        x = x.transpose(1, 2)  # -> (batch, dim, seq_len)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv5(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_x = torch.from_numpy(X).float()
    tensor_y = torch.from_numpy(y).float()
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epochs(model, train_loader, val_loader, device, epochs: int, lr: float):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    history = {"val_loss": []}
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        val_loss = evaluate_loss(model, val_loader, device, criterion)
        history["val_loss"].append(float(val_loss))
    return history


def evaluate_loss(model, loader, device, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
    return np.mean(losses)


def predict_proba(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelos prototipo Notebook 09")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reports-file", type=str, default="reports/09_metrics.json")
    args = parser.parse_args()

    root = find_project_root()
    processed = root / "data" / "processed"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = processed / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    boundaries_train = pd.read_csv(processed / "boundaries" / "boundaries_train.csv")
    boundaries_val = pd.read_csv(processed / "boundaries" / "boundaries_validation.csv")

    df_w2v_train = pd.read_parquet(processed / "delta_w2v_train.parquet")
    df_w2v_val = pd.read_parquet(processed / "delta_w2v_val.parquet")
    df_bert_train = pd.read_parquet(processed / "delta_bert_train.parquet")
    df_bert_val = pd.read_parquet(processed / "delta_bert_val.parquet")

    seq_train_npz = np.load(processed / "bert_sequences_train.npz")
    seq_val_npz = np.load(processed / "bert_sequences_val.npz")
    seq_train = seq_train_npz["X"].astype(np.float32)
    seq_val = seq_val_npz["X"].astype(np.float32)
    y_seq_train = seq_train_npz["y"].astype(np.float32)
    y_seq_val = seq_val_npz["y"].astype(np.float32)

    X_w2v_train = df_w2v_train.drop(columns=["y"]).to_numpy(dtype=np.float32)
    y_w2v_train = df_w2v_train["y"].to_numpy(dtype=int)
    X_w2v_val = df_w2v_val.drop(columns=["y"]).to_numpy(dtype=np.float32)
    y_w2v_val = df_w2v_val["y"].to_numpy(dtype=int)

    X_bert_train = df_bert_train.drop(columns=["y"]).to_numpy(dtype=np.float32)
    y_bert_train = df_bert_train["y"].to_numpy(dtype=int)
    X_bert_val = df_bert_val.drop(columns=["y"]).to_numpy(dtype=np.float32)
    y_bert_val = df_bert_val["y"].to_numpy(dtype=int)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Entrenando en dispositivo: {device}")

    results = {}

    logreg = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    w2v_logreg_metrics, w2v_logreg_pred = sklearn_run(
        logreg, X_w2v_train, y_w2v_train, X_w2v_val, y_w2v_val, boundaries_train, boundaries_val
    )
    results["w2v_logreg"] = w2v_logreg_metrics
    np.savez_compressed(
        preds_dir / "w2v_logreg_val.npz",
        y_true=y_w2v_val.astype(np.int64),
        y_pred=w2v_logreg_pred.astype(np.int64),
    )

    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    w2v_svm_metrics, w2v_svm_pred = sklearn_run(
        svm, X_w2v_train, y_w2v_train, X_w2v_val, y_w2v_val, boundaries_train, boundaries_val
    )
    results["w2v_svm"] = w2v_svm_metrics
    np.savez_compressed(
        preds_dir / "w2v_svm_val.npz",
        y_true=y_w2v_val.astype(np.int64),
        y_pred=w2v_svm_pred.astype(np.int64),
    )

    mlp_w2v = TinyMLP(input_dim=X_w2v_train.shape[1], hidden_dim=128, dropout=0.1)
    train_loader_w2v = make_loader(X_w2v_train, y_w2v_train.astype(np.float32), args.batch_size, shuffle=True)
    train_eval_w2v = make_loader(X_w2v_train, y_w2v_train.astype(np.float32), args.batch_size, shuffle=False)
    val_loader_w2v = make_loader(X_w2v_val, y_w2v_val.astype(np.float32), args.batch_size, shuffle=False)
    train_epochs(mlp_w2v, train_loader_w2v, val_loader_w2v, device, args.epochs, args.lr)
    train_probs_w2v = predict_proba(mlp_w2v, train_eval_w2v, device)
    val_probs_w2v = predict_proba(mlp_w2v, val_loader_w2v, device)
    w2v_mlp_preds = (val_probs_w2v >= 0.5).astype(int)
    results["w2v_mlp"] = {
        "metrics_train": evaluate_predictions(boundaries_train, (train_probs_w2v >= 0.5).astype(int)),
        "metrics_val": evaluate_predictions(boundaries_val, w2v_mlp_preds),
        "classification_report": classification_report(
            y_w2v_val, w2v_mlp_preds, digits=3
        ),
    }
    np.savez_compressed(
        preds_dir / "w2v_mlp_val.npz",
        y_true=y_w2v_val.astype(np.int64),
        y_pred=w2v_mlp_preds.astype(np.int64),
    )

    mlp_model = TinyMLP(input_dim=X_bert_train.shape[1], hidden_dim=256, dropout=0.2)
    train_loader_mlp = make_loader(X_bert_train, y_bert_train.astype(np.float32), args.batch_size, shuffle=True)
    train_eval_loader_mlp = make_loader(X_bert_train, y_bert_train.astype(np.float32), args.batch_size, shuffle=False)
    val_loader_mlp = make_loader(X_bert_val, y_bert_val.astype(np.float32), args.batch_size, shuffle=False)
    train_epochs(mlp_model, train_loader_mlp, val_loader_mlp, device, args.epochs, args.lr)
    train_probs = predict_proba(mlp_model, train_eval_loader_mlp, device)
    val_probs = predict_proba(mlp_model, val_loader_mlp, device)
    bert_mlp_preds = (val_probs >= 0.5).astype(int)
    results["bert_mlp"] = {
        "metrics_train": evaluate_predictions(boundaries_train, (train_probs >= 0.5).astype(int)),
        "metrics_val": evaluate_predictions(boundaries_val, bert_mlp_preds),
        "classification_report": classification_report(
            y_bert_val, bert_mlp_preds, digits=3
        ),
    }
    np.savez_compressed(
        preds_dir / "bert_mlp_val.npz",
        y_true=y_bert_val.astype(np.int64),
        y_pred=bert_mlp_preds.astype(np.int64),
    )

    lstm_model = TinyLSTM(input_dim=seq_train.shape[2], hidden_dim=128)
    train_loader_seq = make_loader(seq_train, y_seq_train, args.batch_size, shuffle=True)
    train_eval_seq = make_loader(seq_train, y_seq_train, args.batch_size, shuffle=False)
    val_loader_seq = make_loader(seq_val, y_seq_val, args.batch_size, shuffle=False)
    train_epochs(lstm_model, train_loader_seq, val_loader_seq, device, args.epochs, args.lr)
    train_probs_seq = predict_proba(lstm_model, train_eval_seq, device)
    val_probs_seq = predict_proba(lstm_model, val_loader_seq, device)
    bert_lstm_preds = (val_probs_seq >= 0.5).astype(int)
    results["bert_lstm"] = {
        "metrics_train": evaluate_predictions(boundaries_train, (train_probs_seq >= 0.5).astype(int)),
        "metrics_val": evaluate_predictions(boundaries_val, bert_lstm_preds),
        "classification_report": classification_report(
            y_seq_val.astype(int), bert_lstm_preds, digits=3
        ),
    }
    np.savez_compressed(
        preds_dir / "bert_lstm_val.npz",
        y_true=y_seq_val.astype(np.int64),
        y_pred=bert_lstm_preds.astype(np.int64),
    )

    cnn_model = TinyCNN(input_dim=seq_train.shape[2], num_filters=128)
    train_epochs(cnn_model, train_loader_seq, val_loader_seq, device, args.epochs, args.lr)
    train_probs_cnn = predict_proba(cnn_model, train_eval_seq, device)
    val_probs_cnn = predict_proba(cnn_model, val_loader_seq, device)
    bert_cnn_preds = (val_probs_cnn >= 0.5).astype(int)
    results["bert_cnn"] = {
        "metrics_train": evaluate_predictions(boundaries_train, (train_probs_cnn >= 0.5).astype(int)),
        "metrics_val": evaluate_predictions(boundaries_val, bert_cnn_preds),
        "classification_report": classification_report(
            y_seq_val.astype(int), bert_cnn_preds, digits=3
        ),
    }
    np.savez_compressed(
        preds_dir / "bert_cnn_val.npz",
        y_true=y_seq_val.astype(np.int64),
        y_pred=bert_cnn_preds.astype(np.int64),
    )

    reports_path = root / args.reports_file
    reports_path.write_text(json.dumps(results, indent=2))
    print(f"Métricas guardadas en {reports_path}")


if __name__ == "__main__":
    main()
