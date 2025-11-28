#!/usr/bin/env python3
"""Precompute delta embeddings for Notebook 09 (Word2Vec + DistilBERT)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def find_project_root(marker: str = ".git") -> Path:
    path = Path.cwd()
    for parent in [path, *path.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError("No se encontró la raíz del proyecto")


def detect_dir(root: Path, preferred: str, fallback: str) -> Path:
    cand = root / preferred
    if cand.exists():
        return cand
    alt = root / fallback
    if alt.exists():
        return alt
    raise FileNotFoundError(f"No se encontró ni {preferred} ni {fallback}")


def load_embeddings(emb_dir: Path, prefix: str, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    split_name = "train" if split == "train" else "validation"
    matrix_path = emb_dir / f"{prefix}_{split_name}.npy"
    index_path = emb_dir / f"{prefix}_{split_name}_index.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    matrix = np.load(matrix_path)
    index_df = pd.read_csv(index_path)
    index_df = index_df.reset_index().rename(columns={"index": "row_idx"})
    index_df["pair_key"] = list(zip(index_df["level"], index_df["doc_id"]))
    index_df["pos_in_doc"] = index_df.groupby("pair_key").cumcount()
    return matrix.astype(np.float32, copy=False), index_df


def build_delta_vectors(boundaries: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    idx_left = boundaries["idx_left"].to_numpy(dtype=int)
    idx_right = boundaries["idx_right"].to_numpy(dtype=int)
    vec_left = embeddings[idx_left]
    vec_right = embeddings[idx_right]
    return (vec_right - vec_left).astype(np.float32, copy=False)


def build_context_sequences(
    boundaries: pd.DataFrame,
    embeddings: np.ndarray,
    index_df: pd.DataFrame,
    half_window: int,
) -> np.ndarray:
    seq_len = half_window * 2
    emb_dim = embeddings.shape[1]
    sequences = np.zeros((len(boundaries), seq_len, emb_dim), dtype=np.float32)
    key_groups = index_df.groupby("pair_key")["row_idx"].apply(list).to_dict()
    pos_lookup = np.zeros(len(index_df), dtype=int)
    pos_lookup[index_df["row_idx"].to_numpy()] = index_df["pos_in_doc"].to_numpy()
    for i, row in enumerate(tqdm(boundaries.itertuples(index=False), total=len(boundaries), desc="context seqs")):
        key = (row.level, row.doc_id)
        doc_rows = key_groups.get(key)
        if doc_rows is None:
            raise KeyError(f"No hay embeddings para {key}")
        pos_right = pos_lookup[row.idx_right]
        left_start = max(0, pos_right - half_window)
        left_rows = doc_rows[left_start:pos_right]
        right_end = min(len(doc_rows), pos_right + half_window)
        right_rows = doc_rows[pos_right:right_end]
        selected = left_rows + right_rows
        for j in range(min(len(selected), seq_len)):
            sequences[i, j] = embeddings[selected[j]]
    return sequences


def to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Precompute delta embeddings for Notebook 09")
    parser.add_argument("--half-window", type=int, default=2, help="Sentencias por lado para secuencias contextuales")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directorio donde guardar los .parquet")
    args = parser.parse_args()

    root = find_project_root()
    processed = root / args.output_dir
    boundaries_dir = processed / "boundaries"
    w2v_dir = detect_dir(root / "features", "word2vec", "embeddings_static")
    bert_dir = detect_dir(root / "features", "contextual", "embeddings_contextual")

    train_bounds = pd.read_csv(boundaries_dir / "boundaries_train.csv")
    val_bounds = pd.read_csv(boundaries_dir / "boundaries_validation.csv")

    w2v_train, idx_w2v_train = load_embeddings(w2v_dir, "S", "train")
    w2v_val, idx_w2v_val = load_embeddings(w2v_dir, "S", "validation")
    bert_train, idx_bert_train = load_embeddings(bert_dir, "C", "train")
    bert_val, idx_bert_val = load_embeddings(bert_dir, "C", "validation")

    print("Construyendo deltas Word2Vec…")
    delta_w2v_train = build_delta_vectors(train_bounds, w2v_train)
    delta_w2v_val = build_delta_vectors(val_bounds, w2v_val)

    print("Construyendo deltas DistilBERT…")
    delta_bert_train = build_delta_vectors(train_bounds, bert_train)
    delta_bert_val = build_delta_vectors(val_bounds, bert_val)

    w2v_cols = [f"w2v_f{i:04d}" for i in range(delta_w2v_train.shape[1])]
    bert_cols = [f"bert_f{i:04d}" for i in range(delta_bert_train.shape[1])]

    df_w2v_train = pd.DataFrame(delta_w2v_train, columns=w2v_cols)
    df_w2v_train["y"] = train_bounds["y"].to_numpy(dtype=int)
    df_w2v_val = pd.DataFrame(delta_w2v_val, columns=w2v_cols)
    df_w2v_val["y"] = val_bounds["y"].to_numpy(dtype=int)

    df_bert_train = pd.DataFrame(delta_bert_train, columns=bert_cols)
    df_bert_train["y"] = train_bounds["y"].to_numpy(dtype=int)
    df_bert_val = pd.DataFrame(delta_bert_val, columns=bert_cols)
    df_bert_val["y"] = val_bounds["y"].to_numpy(dtype=int)

    print("Guardando .parquet…")
    to_parquet(df_w2v_train, processed / "delta_w2v_train.parquet")
    to_parquet(df_w2v_val, processed / "delta_w2v_val.parquet")
    to_parquet(df_bert_train, processed / "delta_bert_train.parquet")
    to_parquet(df_bert_val, processed / "delta_bert_val.parquet")

    print("Construyendo secuencias contextuales para prototipos neuronales…")
    seq_train = build_context_sequences(train_bounds, bert_train, idx_bert_train, args.half_window).astype(np.float16)
    seq_val = build_context_sequences(val_bounds, bert_val, idx_bert_val, args.half_window).astype(np.float16)
    np.savez_compressed(processed / "bert_sequences_train.npz", X=seq_train, y=train_bounds["y"].to_numpy(dtype=np.int64))
    np.savez_compressed(processed / "bert_sequences_val.npz", X=seq_val, y=val_bounds["y"].to_numpy(dtype=np.int64))

    print("Listo. Archivos guardados en data/processed/")


if __name__ == "__main__":
    main()
