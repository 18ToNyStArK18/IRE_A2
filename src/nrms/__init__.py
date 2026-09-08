"""NRMS baseline (Wu et al., 2019), reimplemented in PyTorch following the
official EB-NeRD starter benchmark (https://github.com/jppol-ai/ebnerd-benchmark).

Their implementation is TensorFlow/Keras-2 and leans on `keras.backend`
primitives (`K.dot`, `K.permute_dimensions`, ...) that Keras 3 removed, so it
cannot run on a current TF install. We therefore reproduce the *architecture*
and the *training protocol* rather than vendoring their code: same news/user
encoders, same Wu-2019 negative sampling, same metric protocol, driven by this
repo's unified parquet schema (see src/parse.py) instead of EB-NeRD's raw
layout.

Module map:
    config      hyperparameters, mirroring their `hparams_nrms`
    ids         article-id codec (our string ids -> the int ids NRMS needs)
    adapter     unified behaviors/articles parquet -> NRMS-ready frames
    articles    article text -> token matrix + pretrained embedding weights
    sampling    Wu et al. 2019 negative sampling
    layers      AdditiveAttention (their AttLayer2), MultiHeadSelfAttention
    model       NewsEncoder / UserEncoder / NRMS
    dataset     torch Datasets + collates for training and evaluation
    train       training loop
    evaluate    scoring + metrics (metrics themselves live in src/metrics.py)
"""
