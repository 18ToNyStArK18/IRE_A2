"""Central paths and constants for the data pipeline.

Directory layout:
    data/raw/<dataset>/...        untouched downloaded files (zips extracted)
    data/interim/<dataset>/...    parsed into the unified schema, not yet split
    data/processed/<dataset>/...  temporally split behaviors + feature store + popularity
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MIND_RAW_DIR = RAW_DIR / "mind"
EBNERD_RAW_DIR = RAW_DIR / "ebnerd"

MIND_INTERIM_DIR = INTERIM_DIR / "mind"
EBNERD_INTERIM_DIR = INTERIM_DIR / "ebnerd"

MIND_PROCESSED_DIR = PROCESSED_DIR / "mind"
EBNERD_PROCESSED_DIR = PROCESSED_DIR / "ebnerd"

DATASETS = ("mind", "ebnerd")

# --- Download sources -------------------------------------------------

MIND_HF_REPO = "yjw1029/MIND"
MIND_FILES = {
    "train": "MINDsmall_train.zip",
    "dev": "MINDsmall_dev.zip",
}

EBNERD_BASE_URL = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"
EBNERD_BUNDLE = "demo"  # "demo" (fast, default) or "small" (larger, closer to final scale)
EBNERD_BUNDLE_FILES = {
    "demo": "ebnerd_demo.zip",
    "small": "ebnerd_small.zip",
}
EBNERD_ARTIFACT_FILES = {
    "bert": "artifacts/google_bert_base_multilingual_cased.zip",
    "word2vec": "artifacts/Ekstra_Bladet_word2vec.zip",
}

# --- Phase 3: semantic embeddings -------------------------------------
MIND_BERT_MODEL = "bert-base-uncased"  # monolingual English, faster + more accurate than multilingual for MIND
EBNERD_PROVIDED_ARTIFACT = "bert"  # which EBNERD_ARTIFACT_FILES entry to use as "provided" embeddings
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_LENGTH = 64

# --- Temporal split -----------------------------------------------------
# Both datasets ship two chronologically ordered periods (MIND: train < dev;
# EB-NeRD: train < validation). We treat the *later* provided period as the
# held-out TEST set (never touched for model selection), and carve a
# validation split from the *tail* of the earlier period by time. This keeps
# a strict temporal ordering train < val < test and never shuffles rows.
VAL_FRACTION_OF_TRAIN_PERIOD = 0.10  # last 10% of the early period, by time, becomes val

# --- Feature store / popularity -----------------------------------------
POPULARITY_TOP_N = 500

# --- A2 Q1: behavioural feature engineering ------------------------------
RECENCY_DECAY_RATE = 0.9  # positional decay for history_category_weights, and
                           # the MIND fallback (no per-click timestamps) for
                           # recency_weighted_engagement
# True time-decay half-life, used wherever per-click timestamps exist -- which
# is EB-NeRD only (MIND ships none and takes the positional fallback above, so
# this constant does not affect it). Fitted to EB-NeRD's actual history: the
# per-split history.parquet is a *frozen* snapshot of the 21 days preceding the
# behaviours period, so a click's age relative to the impression scoring it runs
# 116-672h (measured p25/p50/p75 = 116/238/372h at the period start, rising to
# 284/406/540h at its end). At the 24h we used first, every click at the end of
# the period is >=168h old and so contributes at most 2^-7 -- the score collapses
# to ~0.01 for everyone. 168h puts that measured interquartile range within 2-3
# half-lives, where an exponential weight actually discriminates between users.
# Revisit if the history ever becomes a rolling window instead of a snapshot.
RECENCY_HALFLIFE_HOURS = 168.0
SESSION_GAP_MINUTES = 30  # MIND has no native session_id; new session after this idle gap
CTR_PRIOR_STRENGTH = 50  # smoothed CTR = (clicks + CTR_PRIOR_STRENGTH*global_ctr) / (displays + CTR_PRIOR_STRENGTH)
                          # global_ctr is measured from train, not a hand-picked constant (was 10/100 = 10% flat, ~2.5x MIND's real ~4%)
POSITION_BIAS_LOG_BASE = "natural"  # 1/log(rank+2); "natural" or "2"

# --- A2 Q2: candidate generation -----------------------------------------
# Top-K retrieved per impression by A1's retriever and handed to the re-ranker.
# A2 Q2 asks for K ~ 100-200; we persist at 200 so K stays a tunable knob
# without re-running retrieval.
CANDIDATE_K = 200

# A1's ablation-winning configs (src/run_ablation_study.py -> the configs
# compare_final_configs.py locked in, which are also what A1 actually submitted
# to both leaderboards). BM25's winning query-construction method differs by
# dataset; both use title-only over a 10-click window.
BM25_CANDIDATE_CONFIG = {
    "mind": {"fields": ["title"], "window": 10, "method": "tfidf_keywords"},
    "ebnerd": {"fields": ["title"], "window": 10, "method": "recency_weighted"},
}
# Semantic: both pool over the FULL history (window=None); MIND mean-pools,
# EB-NeRD max-pools.
SEMANTIC_CANDIDATE_CONFIG = {
    "mind": {"pooling": "mean", "window": None},
    "ebnerd": {"pooling": "max", "window": None},
}

# MIND's semantic retriever uses MiniLM, NOT MIND_BERT_MODEL: A1 validated a
# bootstrap-significant AUC gain (0.575 -> 0.631, non-overlapping 95% CIs) over
# bert-base-uncased, and MiniLM is ~13x faster to encode a catalogue with.
# Deliberately a separate constant from MIND_BERT_MODEL, which the NRMS
# baseline uses as its tokenizer/word-vector source -- repointing that would
# silently change the Q3 baseline too.
MIND_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
