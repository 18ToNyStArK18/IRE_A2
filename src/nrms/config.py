"""NRMS hyperparameters, mirroring ebnerd-benchmark's `hparams_nrms`
(src/ebrec/models/newsrec/model_config.py) so our reproduction is comparable to
the published baseline. Values below are their defaults; deviations are called
out explicitly.
"""

from __future__ import annotations

from src import config as base_config

# --- architecture (their hparams_nrms defaults) --------------------------
TITLE_SIZE = 30  # tokens kept per article; their MAX_TITLE_LENGTH
HISTORY_SIZE = 20  # clicked articles kept per user, most-recent-last
HEAD_NUM = 20
HEAD_DIM = 20
ATTENTION_HIDDEN_DIM = 200
DROPOUT = 0.2

# The encoders emit HEAD_NUM * HEAD_DIM (=400), NOT ATTENTION_HIDDEN_DIM:
# additive attention returns a weighted *sum of its inputs*, so it preserves
# input width and ATTENTION_HIDDEN_DIM is only the internal projection size.
NEWS_VECTOR_DIM = HEAD_NUM * HEAD_DIM

# --- training (their defaults / reproducibility script) ------------------
LEARNING_RATE = 1e-4
NPRATIO = 4  # negatives sampled per positive (Wu et al. 2019)
BATCH_SIZE = 32
EPOCHS = 5
# Callback settings, matching their reproducibility script exactly:
#   EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True)
#   ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.2, patience=2, min_lr=1e-6)
EARLY_STOPPING_PATIENCE = 4
LR_PLATEAU_PATIENCE = 2
LR_PLATEAU_FACTOR = 0.2
LR_PLATEAU_MIN_LR = 1e-6
SEED = 42

# Q3 freshness arm (src/nrms/freshness.py). Deliberately tiny: the head is
# ~50 parameters against ~192M in the embedding matrix, so a gain from it
# cannot be attributed to added capacity.
FRESHNESS_HIDDEN_DIM = 16

# --- per-dataset text encoding -------------------------------------------
# EB-NeRD is Danish, so the benchmark uses multilingual XLM-RoBERTa. MIND is
# English and already has a monolingual choice recorded in src/config.py.
TEXT_ENCODER = {
    "mind": base_config.MIND_BERT_MODEL,
    "ebnerd": "FacebookAI/xlm-roberta-base",
}

# MIND ships no article body at all (see src/parse.py), so its text is
# title + abstract only; EB-NeRD gets the body too, as in their script.
TEXT_COLUMNS = {
    "mind": ("title", "abstract"),
    "ebnerd": ("title", "abstract", "body"),
}

PROCESSED_DIRS = {
    "mind": base_config.MIND_PROCESSED_DIR,
    "ebnerd": base_config.EBNERD_PROCESSED_DIR,
}


def artifact_dir(dataset: str):
    """Where this dataset's NRMS artifacts live: id map, token matrix,
    checkpoints, predictions, metrics. Under data/, so .gitignore covers it."""
    return PROCESSED_DIRS[dataset] / "nrms"
