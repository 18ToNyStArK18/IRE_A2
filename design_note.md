# Learning from Click-Logs on EB-NeRD and MIND
### CS4.406 Assignment 2 — Design Note

---

## 1. What we built

A two-stage retrieve-then-rank news recommender, evaluated on EB-NeRD (demo) and
MIND-small, plus a reproduced NRMS baseline and one principled improvement to it.

```
behaviour logs ──► unified schema ──► temporal split ──► feature store
                                            │
                    ┌───────────────────────┴────────────────────────┐
                    │                                                │
            Stage 1: candidate generation              NRMS baseline (Q3)
            (1-hour fresh pool, top-200)               scores the in-view list
                    │                                                │
            Stage 2: LambdaMART re-ranker                   + freshness arm
            (24 behavioural features)
                    │
              ranked list ──► evaluation harness ──► Codabench submission
```

The single most consequential decision in the project was not in the model. It
was recognising that Assignment 1's candidate generator was searching the wrong
space, and that fixing it moved recall@200 from **2.55% to 97.0%** — a change no
amount of re-ranker work could have matched.

---

## 2. Data pipeline and leakage discipline

Both datasets are parsed into one schema: an `articles` table (title, abstract,
body, category, entities, `published_time`) and a `behaviors` table (one row per
impression: history, candidates, labels, timestamp). EB-NeRD additionally
carries `session_id` and per-history-click dwell times; MIND has neither, and
that asymmetry recurs throughout the results.

Splits are **temporal, never random**: the later of each dataset's two shipped
periods becomes the held-out test set, and the last 10% of the earlier period by
time becomes validation. Assertions enforce
`max(train.time) ≤ min(val.time) ≤ min(test.time)`.

Leakage discipline was treated as a correctness property with tests behind it,
not a box to tick. Three mechanisms:

**As-of-time article statistics.** Popularity, CTR and freshness are answered by
`TrainEventIndex.as_of(article, t)`, which counts only events strictly before
*t*. A single whole-period aggregate would be safe for val/test but wrong for
training rows: an article clicked *in* impression *I* would have that very click
folded into the popularity figure used to score *I*. A regression test asserts
that a candidate's own click never enters its own feature.

**Click reporting lag.** A click happens *after* its impression, so clicks from
the preceding few minutes may not exist yet at serving time. Stage 1 counts
clicks only up to *t* − 10 min. Measured cost: 0 pp on EB-NeRD, ~1 pp on MIND at
K=200; a 30-minute lag would cost MIND ~4 pp.

**Backward-only session counters.** Within-session features count strictly
earlier impressions in the same session; an impression never sees its own
outcome or a later one.

**Serving-time feature availability (Q9).** EB-NeRD's raw article table ships
`total_inviews`/`total_pageviews`/`total_read_time`, aggregated over the whole
observation period. Our unified schema excludes them, and a test asserts it.
Assignment 1 quantified why: re-introducing `total_pageviews` as a ranking
signal inflated validation AUC by **+0.087 (+18%)**. We also report the
model-visible version of the same idea: MIND has no dwell time at all, so
`avg_history_dwell_time` carries an explicit `has_dwell_time` flag rather than a
zero that would read as "no engagement".

---

## 3. Stage 1: candidate generation, and the mistake that dominated everything

### 3.1 The diagnosis

Assignment 1 retrieved top-K by similarity between the user's click history and
the **whole catalogue**. Carried into Assignment 2 unchanged, it recovered the
clicked article for only 2.55% (EB-NeRD) / 2.90% (MIND) of test impressions at
K=200. No configuration in A1's 39-config ablation exceeded 4.6%.

That ceiling is not a property of the task. Two measurements explain it:

- **Clicks go to what is in circulation right now.** EB-NeRD's median
  clicked-article age at click time is **3.1 hours**; 92% are clicked within 24 h
  of publication. The catalogue reaches back to **2000**.
- **History similarity is the wrong ranking signal for a fresh pool.** On MIND a
  title query scores a median of only ~20 of ~2,000 pool articles above zero.

So two independent mistakes compounded: the wrong search space, and the wrong
signal to rank it by.

### 3.2 The fix, and the ablation that separates the two causes

Candidates are drawn from what the platform actually displayed in the hour
before the impression, ranked by recent (lagged) click counts, backfilled from a
24-hour window when the 1-hour pool is short.

**Test recall@K, full populations:**

| stage-1 arm | EB-NeRD @50 / @100 / @200 | MIND @50 / @100 / @200 |
|---|---|---|
| `bm25` — A1, whole catalogue | 0.82 / 1.41 / **2.55%** | 1.27 / 1.96 / **2.90%** |
| `bm25_fresh` — A1's scoring, 1-hour pool | 21.9 / 42.9 / **84.2%** | 10.0 / 14.2 / **21.0%** |
| `popular` — 1-hour pool, lagged clicks | 83.4 / 93.8 / **97.0%** | 81.6 / 88.6 / **93.9%** |

The middle row is the important one: it changes *only* the search space, keeping
A1's scoring untouched, and recovers most of the gap by itself. The third row
changes only the ranking signal. Attributing the whole gain to either alone
would be wrong.

One methodological note. `bm25_fresh`'s top-200 is ~90% ties, and its recall
swung between 10% and 92% depending purely on the tie-breaking rule. It now
breaks ties with a seeded random draw — the only rule under which the arm
measures BM25 and nothing else. Insertion order smuggles in recency; popularity
ties turn the arm into `popular`.

Window length was chosen on measurement: 1 hour beat 6 h and 24 h, because longer
windows let stale-but-popular articles crowd out fresh ones.

---

## 4. Stage 2: the re-ranker

### 4.1 Objective

**LambdaMART** (LightGBM `objective="lambdarank"`): RankNet pairwise gradients
weighted by |ΔnDCG|. Stated precisely, it is a *pairwise gradient weighted by a
listwise metric*, not a listwise loss — and for a GBDT it is not one of three
options, since RankNet is its ancestor and LambdaRank is the gradient trick
inside it.

It was chosen because the ΔnDCG weighting spends capacity at the top of the
list, which is what MRR and nDCG@5 reward. `lambdarank_truncation_level` is set
to 200 rather than LightGBM's default of 30: our positives began at median rank
80, so the default would have left most of them outside the gradient window.

### 4.2 Features (24)

| group | features |
|---|---|
| click history | click count, recency-weighted engagement, `has_history_timestamps` |
| session | hour, day-of-week, impression size, session position, impressions/clicks earlier in session, avg history dwell time + availability flag |
| article (as-of *t*) | click count, display count, log click count, smoothed CTR, log freshness, `has_known_publish_time` |
| history × candidate | category binary match, category affinity, **history-title BM25**, **history-embedding cosine** |
| stage-1 output | retrieval rank, position bias, retrieval score |

Two details worth reporting. **CTR is shrunk toward each dataset's own measured
base rate** (MIND 4.1%, EB-NeRD 8.9%) rather than a hand-picked constant; a flat
10% prior would have dominated the feature. **Recency weighting uses real
elapsed time** where per-click timestamps exist (EB-NeRD); an earlier version
weighted by list position, which for a plain count collapses to a function of
history length and carries no recency signal at all.

The two **history × candidate similarity** features close the Q1.1 "titles,
categories, embeddings" requirement for the shipped arm, and were the last thing
added: the fresh pool ranks by click counts, so without them history content
reached the model only through category. Both reuse Assignment 1 directly — the
same history-title BM25 query, and the same pooled user embedding — applied to
whatever candidates stage 1 returned. §5.2 shows they decide the MIND result.

### 4.3 Two bugs that had to be fixed before the re-ranker beat stage 1

**Train/serve skew in the article statistics.** The index originally froze
val/test statistics at the end of training. Training rows got live, accumulating
counts; test rows got a snapshot in which the fresh articles actually clicked all
read zero. The median `click_count_article` of clicked articles was **16 on
train and 0 on test** (99.2% of clicked test articles read exactly zero). The
model learned "many clicks → clicked" and then buried fresh articles, finishing
18–21% *below* stage 1. Fix: build the index over every earlier event, still
strictly before *t*, with the same click lag — which is what a live system's
counters actually hold.

**Hard-negative sampling.** Keeping each group's 24 best-ranked negatives
over-represents exactly the region where stage-1 rank separates the positive
least well, so the model learned rank was uninformative. Random negatives keep
the rank distribution representative, and cost a quarter of the rows.

---

## 5. Results: the two-stage pipeline

Test split, K=200, "before" = stage-1 popularity order, bootstrap 95% CIs over
impressions.

| | EB-NeRD | MIND |
|---|---|---|
| impressions / click in top-200 | 25,356 / 97.0% | 73,152 / 93.9% |
| AUC | 0.8815 → **0.9219** [0.9205, 0.9233] | 0.8655 → **0.8891** [0.8879, 0.8903] |
| MRR | 0.1639 → **0.2115** (+29.0%) | 0.2237 → **0.2403** (+7.4%) |
| nDCG@5 | 0.1471 → **0.2026** (+37.7%) | 0.2451 → **0.2618** (+6.8%) |
| nDCG@10 | 0.1994 → **0.2705** (+35.7%) | 0.2932 → **0.3150** (+7.4%) |
| top features (gain) | freshness 52%, position bias 20% | position bias 56%, retrieval rank 11%, history-embedding cosine 10% |

Paired over impressions, both gains over stage 1 are significant: EB-NeRD MRR
**+0.0476 [+0.0442, +0.0510]**, MIND **+0.0167 [+0.0149, +0.0183]**.

**Both datasets gain — but MIND only once the re-ranker could compare candidates
with the user's history.** With the 22-feature model MIND was a wash (MRR
0.2237 → 0.2239, CI spanning zero) and we concluded it had "learned to trust
stage 1". That conclusion was about a missing feature, not about MIND: adding the
two history-similarity features moved it to +0.0167 MRR, essentially all of the
dataset's gain (§5.1).

For context, the same re-ranker on catalogue-wide candidates scored MRR 0.0182
(EB-NeRD) and 0.0032 (MIND). **The 10× difference is stage-1 recall, not
ranking** — a re-ranker cannot rank what it was never handed.

### 5.1 What the history-similarity features are worth, and why the datasets differ

Same models with the two features removed, paired over test impressions:

| | EB-NeRD MRR | MIND MRR | MIND nDCG@5 |
|---|---|---|---|
| without them | 0.2102 | 0.2239 | 0.2450 |
| **with them** | **0.2115** | **0.2403** | **0.2618** |
| paired Δ, 95% CI | +0.0013 [−0.0008, +0.0034] | **+0.0164** [+0.0149, +0.0178] | **+0.0167** [+0.0151, +0.0184] |

**MIND gains; EB-NeRD gains nothing.** The cause is the embedding space, and it
is the same measurement that made §5.3's diversity numbers incomparable: two
random MIND articles have cosine 0.055, two random EB-NeRD articles 0.951. MIND's
MiniLM vectors are contrastively trained and spread out, so cosine to a user's
history discriminates; EB-NeRD's provided multilingual-BERT vectors occupy a
narrow cone where clicked and non-clicked candidates average 0.598 to three
decimals. `history_embedding_cosine` takes 10.4% of MIND's model gain and 0.8%
of EB-NeRD's. **A bi-encoder feature is only as good as the geometry of the space
it reads** — the same anisotropy that inverts a diversity metric also erases a
ranking feature.

The per-arm pattern agrees: gains appear only where candidates are uniformly
fresh *and* the similarity is not already the ranking signal (MIND/`popular`
+0.0164, MIND/`bm25_fresh` +0.0033, EB-NeRD/`bm25_fresh` +0.0018,
EB-NeRD/`popular` +0.0013), and are ±0.0006 on the four catalogue arms, where
`retrieval_score` *is* that similarity.

### 5.2 Metrics with and without serving-unavailable features (Q9)

We audited all 24 features for what a live system would actually have and
retrained without the two that fail:

- **`impression_size`** — the length of the impression's own in-view list. On
  EB-NeRD `article_ids_inview` records the articles that came *into view*, so it
  grows with how long the user stays: **Spearman 0.50 with that impression's own
  read time**, an outcome known only afterwards.
- **`has_known_publish_time`** — read off a `published_time` snapshot taken after
  the logs; it flags exactly the articles bulk re-stamped weeks later.

| paired Δ vs the full model | EB-NeRD MRR | MIND MRR |
|---|---|---|
| serving-safe (22 features) | **−0.0218** [−0.0246, −0.0191] | −0.0007 [−0.0019, +0.0006] |

**EB-NeRD's headline shrinks and MIND's does not.** A deployed EB-NeRD model
would score MRR 0.1897, i.e. **+0.0258 [+0.0229, +0.0287] over stage 1 rather
than +0.0476** — the reported gain is roughly half "the user engaged with this
impression". MIND loses nothing, because its gain comes from the
history-similarity features, which are computable at serving time. We report
both numbers; the serving-safe one is what a deployment would get.

`has_known_publish_time` carries **0.0% gain** on both datasets, which confirms
separately that the future-article shortcut is gone under fresh-pool candidates.

### 5.3 Beyond-accuracy and slices (Q5)

| | EB-NeRD | MIND |
|---|---|---|
| diversity (raw) | 0.0374 | 0.9257 |
| diversity vs random baseline | **0.774×** | **0.985×** |
| novelty (bits) | 14.46 | 17.29 |
| coverage (top-10) | 6.79% | 0.32% |

**Raw diversity is not comparable across datasets.** Two *randomly chosen*
articles have cosine 0.951 in EB-NeRD's space and 0.055 in MIND's: the provided
multilingual-BERT vectors are anisotropic and occupy a narrow cone, while MIND's
MiniLM vectors are contrastively trained and spread out. Against each dataset's
own random-pair baseline the ordering **inverts** — EB-NeRD's recommender
concentrates more (0.774×) than MIND's (0.985×). Reporting the raw pair would
have supported the opposite conclusion.

MIND's near-random diversity sits *opposite* its 0.32% coverage: a very small set
of articles that happens to be topically wide. The two metrics are not
substitutes, and quoting either alone misleads.

**Slices, nDCG@5:**

| | cold-start | warm | head | tail |
|---|---|---|---|---|
| EB-NeRD | 0.2818 (n=67) | 0.2023 (n=25,289) | **0.0000** (n=197) | 0.2041 (n=25,159) |
| MIND | 0.2625 (n=12,982) | 0.2616 (n=60,170) | 0.1817 (n=9,150) | 0.2732 (n=64,002) |

MIND's cold-start advantage has essentially closed (0.2625 vs 0.2616, from
0.2580 vs 0.2422): the history-similarity features need click history, so warm
users gained most (+0.0194 nDCG@5 against +0.0045). That is the expected
signature of a history-derived feature, and a useful check that it does what it
claims.

**Cold-start users do slightly better, not worse** — the opposite of the usual
expectation, and of what we first concluded from a weaker stage-1 arm. The
shipped pipeline is not history-driven: stage 1 is a popularity window and the
model leans on freshness and popularity, so a user with no history loses little.

**EB-NeRD's re-ranker is at chance on head articles**: nDCG@5 and nDCG@10 of
exactly 0.0000, AUC 0.5271, the clicked article sitting around rank 139 of 200.
This is a direct consequence of the model's own structure — it puts 53% of its
importance on freshness, and "head" means most-clicked *in training*, i.e. old,
so the model systematically buries precisely those articles. Stage 1 retrieved
the positive for 117 of the 197 affected impressions, so the failure is the
ranker's, not retrieval's. It affects 0.8% of EB-NeRD test impressions against
12.5% of MIND's — EB-NeRD's test clicks land almost entirely on articles that
were not popular in training, which is the news cycle turning over.

---

## 6. Baseline reproduced, then improved (Q3)

### 6.1 NRMS reproduction

NRMS (Wu et al., 2019) reproduced against `ebnerd-benchmark`'s
`hparams_nrms`: multi-head self-attention news encoder, self-attention user
encoder, dot-product scorer, Wu-2019 negative sampling (1 positive +
4 negatives, softmax cross-entropy). Deviations from the benchmark are
documented rather than silent — including that it uses no attention masking, so
pad tokens participate in attention, which we reproduce deliberately.

| | EB-NeRD | MIND |
|---|---|---|
| AUC | 0.5425 | 0.6040 |
| MRR | 0.3394 | 0.2750 |
| nDCG@5 | 0.3770 | 0.2961 |
| nDCG@10 | 0.4570 | 0.3617 |
| epochs (cap) | 15/20 | 8/10 |

Both runs early-stopped inside their caps, so neither comparison below is
confounded by one arm simply training longer.

### 6.2 The improvement: freshness weighting

NRMS reads only title tokens, so it has **no channel at all** for how old an
article is when shown. Given that freshness carries 53% of the re-ranker's gain
on EB-NeRD and that the median clicked article is 3.1 h old, this is the largest
missing signal in the baseline. We add

`score = dot(user, candidate) + g(log1p(age_hours), known)`

with *g* a 2→16→1 MLP — **65 parameters against ~192M** in the embedding matrix,
so no gain can be attributed to added capacity. It is additive on the logit
rather than concatenated into the news vector, which leaves the dot-product
scorer that defines NRMS untouched, so the ablation isolates the age signal
alone.

Three properties make the ablation clean: the head is **zero-initialised** (the
arm starts bit-identical to the baseline, so any gap must have been learned); it
is **constructed only when enabled and under a forked RNG**, so the baseline arm
draws the same data order and reproduces earlier runs exactly; and an
**as-of-time gate** means an article first seen *after* the impression reads
"unknown" rather than "brand new".

### 6.3 Results, with paired bootstrap CIs

Paired over test impressions with both models held fixed (2,000 resamples):

| | EB-NeRD Δ (95% CI) | MIND Δ (95% CI) |
|---|---|---|
| AUC | **+0.0686** [+0.0662, +0.0710] | +0.0002 [−0.0006, +0.0010] |
| MRR | **+0.0567** [+0.0538, +0.0595] | −0.0007 [−0.0014, +0.0001] |
| nDCG@5 | **+0.0657** [+0.0628, +0.0685] | −0.0012 [−0.0021, −0.0003] |
| nDCG@10 | **+0.0525** [+0.0501, +0.0548] | −0.0007 [−0.0015, +0.0001] |

**EB-NeRD: all four CIs clear of zero**, per-impression win/tie/loss 46.2 / 35.1 /
18.6. The gain is visible from epoch 1 (val AUC 0.5795 vs 0.5452), which is what
the zero-initialisation predicts.

**MIND: no effect.** Every delta is a few ten-thousandths; the arms win and lose
about equally per impression. The lone nDCG@5 CI excluding zero is a *loss* of
0.0012 — one marginal exclusion across four correlated metrics, not a result.

The reason is measurable and is **not** the proxy's noisiness, as we first
assumed. A feature can only reorder *within* an impression, so what matters is
within-impression variance:

| | EB-NeRD | MIND |
|---|---|---|
| AUC from the learned age curve alone | **0.677** | 0.514 |
| median age, clicked vs not clicked | 3.1 h vs 4.7 h | 15.0 h vs 16.5 h |
| median within-impression age spread | **1,057 h** | 27.5 h |

On EB-NeRD one impression mixes brand-new articles with ones weeks old, so age
strongly predicts the click. On MIND every candidate is about the same age, so
the age term adds a near-constant to every logit and cannot change the ordering.
MIND's proxy also *caps* the range: first-seen cannot precede the dataset start,
so no article can read older than ~144 h. The defensible claim is therefore
"freshness-as-first-seen does not help on MIND", not "freshness does not help on
MIND".

A caveat we state rather than hide: a paired bootstrap resamples *test
impressions* with both models fixed. It says nothing about **training** variance,
so a gain of the same order as seed noise would need repeated seeds. EB-NeRD's
+0.0686 is far outside that range; MIND's null is not a claim.

---

## 7. Serving and scale (Q4)

Measured single-threaded on CPU, 300 timed requests after 50 warm-up, K=200.
Stage-1 log replay is excluded from request latency and reported separately — a
live system maintains those counters as events arrive, so timing the replay
would measure catch-up, not serving (it measures 0.065 ms/impression regardless).

| | stage 1 | features | scoring | **total p50 / p99** |
|---|---|---|---|---|
| EB-NeRD | 0.12 | 3.42 | 2.58 | **5.97 / 10.50 ms** |
| MIND | 0.56 | 2.72 | 0.14 | **3.47 / 8.00 ms** |

**p99 sits ~10× inside the suggested 100 ms SLA**, at 170 / 253 single-core QPS,
costing **$0.00019–0.00028 per 1000 queries** at $0.17/vCPU-hour.

The distribution is the interesting part: **feature building dominates the
request and stage 1 is negligible** — the intuitive answer, that retrieval or the
GBDT dominates, is wrong on both counts, and latency work should target feature
assembly.

**What the history-similarity features cost.** Adding them roughly doubled
EB-NeRD's request (6.44 → 10.50 ms p99) and raised MIND's by a quarter
(6.99 → 8.00 ms), from two separate causes worth keeping apart: about +1 ms is
the features themselves (one BM25 query over the catalogue plus a dot product per
candidate), and the rest is the model growing, because the extra signal delays
early stopping (EB-NeRD 99 → 174 trees). Set against §5.1, **MIND buys a
significant ranking gain for ~25% more serving cost, while EB-NeRD pays ~2× for a
gain whose CI includes zero** — on EB-NeRD alone these features would not be
worth shipping.

**Index memory**, measured per component in isolated processes:

| component | EB-NeRD | MIND | in the shipped path? |
|---|---|---|---|
| semantic ANN index | 34.5 MiB | 95.6 MiB | **yes** (history-embedding cosine) |
| BM25 inverted index | 4.1 MiB | 36.0 MiB | **yes** (history-title BM25) |
| as-of article statistics | 4.8 MiB | 68.1 MiB | yes |
| LightGBM booster | 2.7 MiB | 3.1 MiB | yes |

This is a finding that reversed during the project, and the reversal is the
interesting part. Replacing similarity retrieval with a recency window removed
both indexes from the serving path, so for most of our work the honest answer to
"measure your ANN index" was **that the shipped pipeline had none**. Adding the
history-similarity features put them back: MIND's resident footprint goes 68.1 →
**199.7 MiB**. The trade is explicit — 132 MiB and ~1 ms per request to make the
MIND re-ranker work at all.

---

## 8. Where it breaks at 10×

Latency is **linear in K** (EB-NeRD 2.44 / 3.21 / 5.08 ms at K = 50 / 100 / 200),
and features are the linear term — so K is the throughput knob, and halving it
costs little recall (93.8% at K=100 against 97.0% at K=200).

Ranked by what fails first:

1. **As-of article-statistics memory — the binding constraint.** It must be
   resident and grows with total events: 68.1 MiB holds 8.9M MIND events today,
   so ~680 MiB at 10× traffic and **~4.4 GB at MINDlarge scale** (~65×). This is
   the first thing to shard or move to a windowed counter store.
2. **CPU for feature assembly.** 165–191 QPS per core means ~10 cores for 10×
   traffic. Embarrassingly parallel across requests, so a cost line rather than a
   wall.
3. **The similarity structures, on a different axis.** The BM25 index and the
   embedding table (132 MiB on MIND) grow with **catalogue size**, not with
   traffic, so 10× traffic does not touch them — but a 10× catalogue does, and the
   embedding table is the one that would then need an ANN service rather than a
   resident array. They are also why a smaller K pays twice: fewer candidates to
   score *and* fewer dot products.
4. **Nothing else.** The fresh pool is bounded by *one hour of traffic*, not by
   catalogue size, so it grows with QPS and not with corpus age — a property that
   came free with the stage-1 redesign. The booster is ~3 MiB.

Two things that would break in a real deployment but are invisible offline: our
pool is rebuilt by replaying a log, where production needs an incrementally
maintained counter store with the same 10-minute click lag; and the model is
retrained in batch, where a one-hour popularity window implies the article
statistics drift within a single day.

---

## 9. What we would do differently

- **Check the geometry of an embedding space before building features on it.**
  The same anisotropy that made raw diversity incomparable across datasets (§5.3)
  also made a history-embedding feature useless on EB-NeRD and decisive on MIND.
  One measurement — the cosine between two random articles — predicts both.
- **Question stage 1 before tuning stage 2.** We spent considerable effort on
  re-ranker losses, features and sampling while recall sat at 2.5%. The
  measurement that changed the project — median clicked-article age of 3.1 h
  against a catalogue reaching to 2000 — took an afternoon and was available from
  the start.
- **Distrust AUC on this task.** It *fell* from 0.98 to 0.92 between arms while
  nDCG rose 10×, because catalogue-wide negatives make discrimination trivially
  easy. MRR and nDCG were the honest headline throughout.
- **Report slices on the shipped configuration only.** Our first cold-start
  conclusion was drawn from a discarded arm and had the sign backwards.
- **MIND's null results are informative, not failures.** MIND lacks publish
  times, dwell times and session IDs, so the two datasets form a natural ablation
  of exactly those three signals — which is more credible evidence than any
  feature-drop experiment we could have designed.

---

## Appendix: reproduction

```bash
python build_pipeline.py                      # download → parse → split → features
python -m src.candidates --method popular      # stage 1
python -m src.reranker --dataset ebnerd --method popular   # stage 2
python run_nrms.py --dataset ebnerd --stage all            # Q3 baseline
python run_nrms.py --dataset ebnerd --stage all --freshness  # Q3 improvement
python scripts/paired_bootstrap.py --dataset ebnerd        # Q3 CIs
python scripts/serving_benchmark.py --dataset ebnerd       # Q4
python scripts/extended_eval.py --dataset ebnerd --method popular  # Q5
python scripts/serving_features_ablation.py --dataset ebnerd       # Q9 + §5.1/§5.2
```

Results land in `results/*.json`. `DesignChoices.md` is the full reasoning log,
including the superseded analyses this note draws its corrections from.
