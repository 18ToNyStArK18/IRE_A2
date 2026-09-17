"""Article embeddings: load where provided (EB-NeRD), compute via BERT where
not (MIND), and write both into a unified per-dataset feature-store table.

MIND ships no article embeddings, so we compute them ourselves with
config.MIND_SEMANTIC_MODEL (MiniLM, A1's bootstrap-significant winner over
bert-base-uncased) over title+abstract, mean-pooled across tokens with the
attention mask and L2-normalized.

EB-NeRD's RecSys24 artifacts already include per-article multilingual BERT
embeddings (`google-bert/bert-base-multilingual-cased`, 768-dim) -- we load
and L2-normalize those rather than recomputing, per the assignment's "load
the provided embeddings where available".

Output: data/processed/<dataset>/feature_store/embeddings.parquet with
columns [article_id, embedding] (embedding = list[float32]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.inverted_index import concat_fields

EMBEDDING_FIELDS = ["title", "abstract"]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def compute_bert_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
    max_length: int = config.EMBEDDING_MAX_LENGTH,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch = [t if t.strip() else "[empty]" for t in batch]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            )
            out = model(**enc)
            hidden = out.last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / counts).cpu().numpy()
            all_embeddings.append(pooled)
            done = start + len(batch)
            if done % (batch_size * 20) == 0 or done == len(texts):
                print(f"  embedded {done}/{len(texts)}")

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def build_mind_embeddings(
    processed_dir=config.MIND_PROCESSED_DIR,
    model_name: str = config.MIND_SEMANTIC_MODEL,
) -> pd.DataFrame:
    """Defaults to MIND_SEMANTIC_MODEL (MiniLM), A1's ablation winner for
    retrieval -- not MIND_BERT_MODEL, which stays reserved for the NRMS
    baseline's tokenizer/word vectors."""
    articles = pd.read_parquet(processed_dir / "articles.parquet")
    texts = [concat_fields(r, EMBEDDING_FIELDS) for r in articles.to_dict("records")]

    print(f"[mind] computing embeddings for {len(texts)} articles with {model_name} ...")
    vectors = compute_bert_embeddings(texts, model_name)
    vectors = _l2_normalize(vectors)

    out = pd.DataFrame({
        "article_id": articles["article_id"].values,
        "embedding": list(vectors),
    })
    fs_dir = processed_dir / "feature_store"
    fs_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fs_dir / "embeddings.parquet", index=False)
    print(f"[mind] saved {len(out)} embeddings ({vectors.shape[1]}-dim) -> {fs_dir / 'embeddings.parquet'}")
    return out


def load_ebnerd_provided_embeddings(
    processed_dir=config.EBNERD_PROCESSED_DIR,
    artifact: str = config.EBNERD_PROVIDED_ARTIFACT,
) -> pd.DataFrame:
    artifact_dir = config.EBNERD_RAW_DIR / "artifacts" / artifact
    parquet_files = list(artifact_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet found under {artifact_dir}; run download.download_ebnerd_artifact('{artifact}') first."
        )
    raw = pd.read_parquet(parquet_files[0])
    embedding_col = [c for c in raw.columns if c != "article_id"][0]

    articles = pd.read_parquet(processed_dir / "articles.parquet")
    catalog_ids = set(articles["article_id"])

    raw["article_id"] = raw["article_id"].astype(str)
    raw = raw[raw["article_id"].isin(catalog_ids)]

    vectors = _l2_normalize(np.stack(raw[embedding_col].to_numpy()).astype(np.float32))
    out = pd.DataFrame({"article_id": raw["article_id"].values, "embedding": list(vectors)})

    fs_dir = processed_dir / "feature_store"
    fs_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fs_dir / "embeddings.parquet", index=False)

    coverage = len(out) / len(catalog_ids)
    print(
        f"[ebnerd] loaded {len(out)} provided embeddings from '{artifact}' "
        f"({vectors.shape[1]}-dim), catalog coverage={coverage:.3f} -> {fs_dir / 'embeddings.parquet'}"
    )
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("mind", "all"):
        build_mind_embeddings()
    if args.dataset in ("ebnerd", "all"):
        load_ebnerd_provided_embeddings()


if __name__ == "__main__":
    main()
