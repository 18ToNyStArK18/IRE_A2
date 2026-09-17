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
            (22 behavioural features)
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

### 4.2 Features (22)

| group | features |
|---|---|
| click history | click count, recency-weighted engagement, `has_history_timestamps` |
| session | hour, day-of-week, impression size, session position, impressions/clicks earlier in session, avg history dwell time + availability flag |
| article (as-of *t*) | click count, display count, log click count, smoothed CTR, log freshness, `has_known_publish_time` |
| history × candidate | category binary match, category affinity |
| stage-1 output | retrieval rank, position bias, retrieval score |

Two details worth reporting. **CTR is shrunk toward each dataset's own measured
base rate** (MIND 4.1%, EB-NeRD 8.9%) rather than a hand-picked constant; a flat
10% prior would have dominated the feature. **Recency weighting uses real
elapsed time** where per-click timestamps exist (EB-NeRD); an earlier version
weighted by list position, which for a plain count collapses to a function of
history length and carries no recency signal at all.

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
| AUC | 0.8815 → **0.9213** [0.9199, 0.9227] | 0.8655 → 0.8799 [0.8786, 0.8813] |
| MRR | 0.1639 → **0.2102** (+28.2%) | 0.2237 → 0.2239 (+0.1%) |
| nDCG@5 | 0.1471 → **0.2007** (+36.5%) | 0.2451 → 0.2450 (−0.0%) |
| nDCG@10 | 0.1994 → **0.2691** (+34.9%) | 0.2932 → 0.2980 (+1.6%) |
| top features (gain) | freshness 53%, position bias 23% | position bias 57%, retrieval rank 15% |

**EB-NeRD gains substantially; MIND is a wash**, and the feature importances say
why. On MIND, 72% of the model's gain comes from stage-1 rank — it learned to
trust stage 1. MIND has no publish date, and inside an hour-old pool our
"first-seen" proxy is nearly uniform across candidates, so the article features
carry little beyond what popularity already encoded.

For context, the same re-ranker on catalogue-wide candidates scored MRR 0.0182
(EB-NeRD) and 0.0032 (MIND). **The 10× difference is stage-1 recall, not
ranking** — a re-ranker cannot rank what it was never handed.

### 5.1 Beyond-accuracy and slices (Q5)

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
| EB-NeRD | 0.2699 (n=65) | 0.2005 (n=24,526) | **0.0000** (n=117) | 0.2023 (n=24,474) |
| MIND | 0.2580 (n=12,171) | 0.2422 (n=56,511) | 0.1605 (n=8,370) | 0.2571 (n=60,312) |

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
| EB-NeRD | 0.33 | 4.23 | 0.37 | **4.95 / 9.15 ms** |
| MIND | 1.15 | 4.01 | 0.54 | **5.74 / 9.58 ms** |

**p99 sits ~10× inside the suggested 100 ms SLA**, at 191 / 165 single-core QPS,
costing **$0.00025–0.00029 per 1000 queries** at $0.17/vCPU-hour.

The distribution is the interesting part: **feature building is ~85% of the
request and the model is ~7%**. The intuitive answer — that the GBDT dominates —
is wrong, and any latency work should target feature assembly.

**Index memory**, measured per component in isolated processes:

| component | EB-NeRD | MIND | in the shipped path? |
|---|---|---|---|
| semantic ANN index | 34.5 MiB | 95.6 MiB | **no** |
| BM25 inverted index | 4.1 MiB | 36.0 MiB | **no** |
| as-of article statistics | 4.8 MiB | 68.1 MiB | yes |
| LightGBM booster | 2.7 MiB | 3.1 MiB | yes |

The assignment asks us to measure our ANN index; the honest answer is that **the
shipped pipeline has none**. Replacing similarity retrieval with a recency
window deleted the largest serving structure while raising recall 38×.

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
3. **Nothing else.** The fresh pool is bounded by *one hour of traffic*, not by
   catalogue size, so it grows with QPS and not with corpus age — a property that
   came free with the stage-1 redesign. The booster is ~3 MiB.

Two things that would break in a real deployment but are invisible offline: our
pool is rebuilt by replaying a log, where production needs an incrementally
maintained counter store with the same 10-minute click lag; and the model is
retrained in batch, where a one-hour popularity window implies the article
statistics drift within a single day.

---

## 9. What we would do differently

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
```

Results land in `results/*.json`. `DesignChoices.md` is the full reasoning log,
including the superseded analyses this note draws its corrections from.
