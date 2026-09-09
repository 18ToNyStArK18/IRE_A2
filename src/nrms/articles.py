"""Article text -> the two tensors NRMS needs: a token matrix indexed by
article code, and the pretrained word-embedding matrix that initialises the
news encoder's embedding layer.

This mirrors ebnerd-benchmark's pipeline:
    concat_str_columns -> convert_text2encoding_with_transformers
    -> create_article_id_to_value_mapping        (the token matrix)
    get_transformers_word_embeddings             (the embedding weights)

Row 0 of the token matrix is all-zeros and belongs to the reserved padding
code (see src/nrms/ids.py). Articles that exist only in the behaviour logs and
carry no catalogue text also land on an all-zero row -- the same "unknown
article is a zero vector" convention the benchmark uses.

The token matrix is cached to disk because tokenising ~65k MIND articles or
~125k EB-NeRD articles is slow enough to be annoying to repeat, and it is
deterministic given (codec, text columns, tokenizer, title_size).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.nrms.ids import ArticleCodec


def build_token_matrix(
    article_text,
    codec: ArticleCodec,
    tokenizer,
    title_size: int,
    batch_size: int = 1024,
) -> np.ndarray:
    """(n_articles + 1, title_size) int32 token ids, indexed by article code.

    Uses padding="max_length" + truncation so every article is exactly
    title_size tokens, as their `convert_text2encoding_with_transformers` does.

    Batched because truncation happens *after* encoding: EB-NeRD's ~125k
    articles carry full bodies, and handing all of them to the tokenizer at
    once builds every full-length encoding in memory before discarding all but
    the first `title_size` tokens.
    """
    matrix = np.zeros((codec.n_articles + 1, title_size), dtype=np.int32)

    ids = [str(a) for a in article_text["article_id"]]
    texts = [str(t) for t in article_text["text"]]

    codes = codec.encode(ids)
    known = codes != 0  # articles absent from the codec keep their zero row
    if not known.any():
        return matrix

    known_codes = codes[known]
    known_texts = [t for t, keep in zip(texts, known) if keep]

    for start in range(0, len(known_texts), batch_size):
        chunk = known_texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=title_size,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        matrix[known_codes[start : start + len(chunk)]] = encoded

    return matrix


def load_word_embeddings(model_name: str) -> np.ndarray:
    """The transformer's input word-embedding matrix, (vocab_size, dim).

    Equivalent to their `get_transformers_word_embeddings`. NRMS initialises
    its own trainable embedding from this rather than running the transformer
    -- the transformer is used purely as a source of word vectors and a
    tokenizer, which is why the baseline is cheap to train.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    return model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def token_matrix_fingerprint(model_name: str, text_columns, title_size: int) -> str:
    """Short hash of everything the matrix contents depend on besides the codec.

    Without this the cache is keyed on title_size alone and validated only by
    shape -- so swapping TEXT_ENCODER or changing TEXT_COLUMNS silently reuses
    the old tokenisation, because the shape still matches. Folding those into
    the filename turns a stale cache into a plain cache miss.
    """
    payload = "|".join([model_name, ",".join(text_columns), str(title_size)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def load_or_build_token_matrix(
    artifact_dir: Path,
    article_text,
    codec: ArticleCodec,
    tokenizer,
    title_size: int,
    model_name: str,
    text_columns,
    rebuild: bool = False,
) -> np.ndarray:
    fingerprint = token_matrix_fingerprint(model_name, text_columns, title_size)
    path = artifact_dir / f"article_tokens_{title_size}_{fingerprint}.npy"
    if path.exists() and not rebuild:
        matrix = np.load(path)
        if matrix.shape == (codec.n_articles + 1, title_size):
            return matrix
        # Catalogue grew or shrank since the cache was written; rebuild rather
        # than silently indexing into a stale matrix.
    matrix = build_token_matrix(article_text, codec, tokenizer, title_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix
