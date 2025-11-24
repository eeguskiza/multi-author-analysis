#!/usr/bin/env python3
"""Train shallow models on TF-IDF deltas for Notebook 08."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import LinearSVC


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


def load_boundaries(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(processed_dir / "boundaries_train_sw.parquet")
    val = pd.read_parquet(processed_dir / "boundaries_val_sw.parquet")
    return train, val


def build_delta_matrix(boundaries: pd.DataFrame, X_char, char_index: pd.DataFrame, window: int):
    df = boundaries.sort_values(["level", "doc_id", "boundary_id"]).reset_index(drop=True)
    char_ix = char_index.sort_values(["level", "doc_id", "sent_id"]).reset_index(drop=True)
    rows = []
    for (level, doc_id), doc_bounds in tqdm(df.groupby(["level", "doc_id"], sort=False), desc="delta docs"):
        doc_sents = char_ix[(char_ix["level"] == level) & (char_ix["doc_id"] == doc_id)]
        row_idxs = doc_sents["row_idx"].to_numpy()
        X_doc = X_char[row_idxs, :]
        n_sent = X_doc.shape[0]
        n_boundaries = max(0, n_sent - 1)
        if n_boundaries != len(doc_bounds):
            raise ValueError(f"Dimensiones incompatibles para {(level, doc_id)}")
        for b in range(n_boundaries):
            left_start = max(0, b - window + 1)
            left_end = b + 1
            right_start = b + 1
            right_end = min(n_sent, b + 1 + window)
            v_left = X_doc[left_start:left_end].mean(axis=0)
            v_right = X_doc[right_start:right_end].mean(axis=0)
            rows.append(v_right - v_left)
    X_delta = sparse.vstack(rows).tocsr()
    y = df["y"].to_numpy().astype(int)
    return X_delta, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--window", type=int, default=None)
    args = parser.parse_args()

    root = find_project_root()
    processed = root / "data" / "processed"
    features_dir = root / "features" / "tfidf" / "char"
    reports = root / "reports"
    reports.mkdir(exist_ok=True, parents=True)

    boundaries_train, boundaries_val = load_boundaries(processed)
    if args.window is None:
        best = json.loads((reports / "08_sw_best.json").read_text())
        window = int(best["window"])
    else:
        window = args.window

    X_char_train = sparse.load_npz(features_dir / "X_train_char.npz")
    X_char_val = sparse.load_npz(features_dir / "X_val_char.npz")
    idx_char_train = pd.read_csv(features_dir / "index_train.csv").reset_index().rename(columns={"index": "row_idx"})
    idx_char_val = pd.read_csv(features_dir / "index_val.csv").reset_index().rename(columns={"index": "row_idx"})

    X_train_delta, y_train = build_delta_matrix(boundaries_train, X_char_train, idx_char_train, window)
    X_val_delta, y_val = build_delta_matrix(boundaries_val, X_char_val, idx_char_val, window)

    svd = TruncatedSVD(n_components=args.components, random_state=42)
    X_train_red = svd.fit_transform(X_train_delta)
    X_val_red = svd.transform(X_val_delta)

    logreg = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    logreg.fit(X_train_red, y_train)
    y_train_log = logreg.predict(X_train_red)
    y_val_log = logreg.predict(X_val_red)

    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    svm.fit(X_train_red, y_train)
    y_train_svm = svm.predict(X_train_red)
    y_val_svm = svm.predict(X_val_red)

    metrics_val_log = evaluate_predictions(boundaries_val, y_val_log)
    metrics_val_svm = evaluate_predictions(boundaries_val, y_val_svm)

    (reports / "08_logreg_metrics.json").write_text(json.dumps({
        "label": "logreg_tfidf",
        "metrics_val": metrics_val_log,
        "classification_report": classification_report(y_val, y_val_log, digits=3),
    }, indent=2))

    (reports / "08_svm_metrics.json").write_text(json.dumps({
        "label": "svm_tfidf",
        "metrics_val": metrics_val_svm,
        "classification_report": classification_report(y_val, y_val_svm, digits=3),
    }, indent=2))

    sparse.save_npz(processed / "X_train_delta.npz", X_train_delta)
    sparse.save_npz(processed / "X_val_delta.npz", X_val_delta)
    np.save(processed / "y_train_delta.npy", y_train)
    np.save(processed / "y_val_delta.npy", y_val)

    print("Modelos entrenados. Métricas guardadas en reports/08_logreg_metrics.json y 08_svm_metrics.json")


if __name__ == "__main__":
    main()
