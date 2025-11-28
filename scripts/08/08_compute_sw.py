#!/usr/bin/env python3
"""Compute sliding-window cosine scores + threshold search for Notebook 08."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


def find_project_root(marker: str = ".git") -> Path:
    path = Path.cwd()
    for parent in [path, *path.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"No se encontró {marker}")


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


def cosine_scores_for_doc(X_doc: sparse.csr_matrix, window: int) -> np.ndarray:
    n_sent = X_doc.shape[0]
    n_boundaries = max(0, n_sent - 1)
    scores = np.zeros(n_boundaries, dtype=float)
    if n_boundaries == 0:
        return scores
    for b in range(n_boundaries):
        left_start = max(0, b - window + 1)
        left_end = b + 1
        right_start = b + 1
        right_end = min(n_sent, b + 1 + window)
        v_left = np.asarray(X_doc[left_start:left_end].mean(axis=0)).reshape(1, -1)
        v_right = np.asarray(X_doc[right_start:right_end].mean(axis=0)).reshape(1, -1)
        sim = cosine_similarity(v_left, v_right)[0, 0]
        scores[b] = float(sim)
    return scores


def add_cosine_scores(boundaries: pd.DataFrame, X_char, char_index: pd.DataFrame, window: int) -> pd.Series:
    df = boundaries.sort_values(["level", "doc_id", "boundary_id"]).reset_index(drop=True)
    char_ix = char_index.sort_values(["level", "doc_id", "sent_id"]).reset_index(drop=True)
    scores = np.zeros(len(df), dtype=float)
    offset = 0
    for (level, doc_id), doc_bounds in tqdm(df.groupby(["level", "doc_id"], sort=False), desc=f"window {window}"):
        doc_sents = char_ix[(char_ix["level"] == level) & (char_ix["doc_id"] == doc_id)]
        row_idxs = doc_sents["row_idx"].to_numpy()
        X_doc = X_char[row_idxs, :]
        doc_scores = cosine_scores_for_doc(X_doc, window=window)
        scores[offset : offset + len(doc_scores)] = doc_scores
        offset += len(doc_scores)
    return pd.Series(scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window cosine search")
    parser.add_argument("--window-sizes", nargs="*", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--tau-min", type=float, default=0.70)
    parser.add_argument("--tau-max", type=float, default=0.98)
    parser.add_argument("--tau-steps", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    root = find_project_root()
    processed = root / "data" / "processed"
    boundaries_dir = processed / "boundaries"
    features_dir = root / "features" / "tfidf" / "char"
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True, parents=True)

    boundaries_train = pd.read_csv(boundaries_dir / "boundaries_train.csv")
    boundaries_val = pd.read_csv(boundaries_dir / "boundaries_validation.csv")
    X_char_train = sparse.load_npz(features_dir / "X_train_char.npz")
    X_char_val = sparse.load_npz(features_dir / "X_val_char.npz")
    idx_char_train = pd.read_csv(features_dir / "index_train.csv").reset_index().rename(columns={"index": "row_idx"})
    idx_char_val = pd.read_csv(features_dir / "index_val.csv").reset_index().rename(columns={"index": "row_idx"})

    train_sw = boundaries_train.copy()
    val_sw = boundaries_val.copy()
    for w in args.window_sizes:
        col = f"cos_char_w{w}"
        train_sw[col] = add_cosine_scores(train_sw, X_char_train, idx_char_train, w)
        val_sw[col] = add_cosine_scores(val_sw, X_char_val, idx_char_val, w)

    train_sw.to_parquet(processed / "boundaries_train_sw.parquet")
    val_sw.to_parquet(processed / "boundaries_val_sw.parquet")

    taus = np.linspace(args.tau_min, args.tau_max, args.tau_steps)
    results = []
    for w in args.window_sizes:
        col = f"cos_char_w{w}"
        for tau in taus:
            y_pred = (train_sw[col].to_numpy() < tau).astype(int)
            metrics = evaluate_predictions(train_sw, y_pred)
            d = {"window": w, "tau": float(tau)}
            d.update(metrics)
            results.append(d)
    results_df = pd.DataFrame(results)
    results_df.to_csv(reports_dir / "08_sw_results.csv", index=False)
    best = results_df.sort_values(["f1_macro", "pk", "windowdiff"], ascending=[False, True, True]).iloc[0]
    (reports_dir / "08_sw_best.json").write_text(json.dumps(best.to_dict(), indent=2))

    print("Proceso completado. Resultados en reports/08_sw_results.csv y 08_sw_best.json")


if __name__ == "__main__":
    main()
