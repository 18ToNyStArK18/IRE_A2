# Design Choices — A2 Re-Ranker (Q2)

Working design log for the Q2 re-ranker. Records what we chose, what we
rejected, and *why*, so the Q6 design note can be written from evidence rather
than memory. Decisions marked **PROPOSED** are not yet signed off.

Last updated: 2026-09-11

> **Current state (2026-09-11).** Stage 1 is now fresh-pool candidate generation
> (`src/fresh_pool.py`, the `popular` method): the articles the platform displayed
> in the hour before each impression, ranked by lagged click counts. Test
> recall@200 is **97.0% (EB-NeRD) / 93.9% (MIND)**, up from 2.55% / 2.90%. Stage 2
> is LambdaMART trained on 49 random negatives per group, with Q1 article
> statistics taken as-of each impression over every earlier event. Test MRR after
> re-ranking: **0.2102 (EB-NeRD, +28% over stage 1) / 0.2239 (MIND, flat)**.
> Sections 0–2D document the catalogue-candidate phase that came first; where §2E
> overturned their premises they carry a dated note. Read §2E for the current
> pipeline.

---

## 0. The constraint that drives every decision below

> **Superseded 2026-09-11 (§2E).** Everything in this section is true of A1's
> catalogue-wide candidates, which the re-ranker used until §2E. With fresh-pool
> candidates the click is in the top-200 for 97.0% (EB-NeRD) / 93.9% (MIND) of
> test impressions, so the ceiling described here no longer applies.

Read this first; several conclusions here invert the textbook answer, and this
is the reason.

We took A2 Q2 literally: the re-ranker scores **A1's retrieved top-K**, not the
impression's in-view list. A1's catalogue-wide retrieval recovers very few of
the articles a given impression actually displayed:

| | Recall@200 | source |
|---|---|---|
| EB-NeRD (BM25, recency_weighted, w=10, title) | **0.035** | A1 `ablation_study.csv`, reproduced exactly by `src/candidates.py` |
| MIND (BM25, tfidf_keywords, w=10, title) | **0.045** | A1 `ablation_study.csv` |
| best of all 39 configs A1 tried | **0.046** | A1 `ablation_study.csv` |

So roughly **96% of impressions carry no positive at all** into the re-ranker.
Measured on EB-NeRD val: 82 positive rows out of 464,600 — a **0.018%** positive
rate.

Group shapes, measured:

| dataset | split | impressions | rows at K=200 | mean clicks/impr | exactly 1 click |
|---|---|---|---|---|---|
| MIND | train | 138,578 | 27,715,600 | 1.50 | 72.8% |
| MIND | val | 18,387 | 3,677,400 | 1.53 | 71.0% |
| MIND | test | 73,152 | 14,630,400 | 1.52 | 71.2% |
| EB-NeRD | train | 22,401 | 4,480,200 | 1.00 | 99.7% |
| EB-NeRD | val | 2,323 | 464,600 | 1.00 | 99.7% |
| EB-NeRD | test | 25,356 | 5,071,200 | 1.00 | 99.7% |

Two facts to carry forward:

1. **Every group has exactly 200 candidates**, and at most a handful are
   relevant — usually **exactly one** (EB-NeRD is 99.7% single-click).
2. **Only ~3.5–4.5% of groups contain their positive at all.** Estimated usable
   groups: EB-NeRD train ≈ 780, MIND train ≈ 6,200. (EB-NeRD val's 82 is
   measured; the rest are projected from Recall@200.)

Note the clicks/impression column is measured over the *in-view* list, so every
impression has ≥1 click there. The ~96% loss happens at retrieval.

### 0.1 Does one positive per group make re-ranking pointless? No — measured

> **Superseded 2026-09-11 (§2E).** The ~96%-empty groups and the ≈ 0.035 MRR
> ceiling below belong to catalogue candidates. With fresh-pool candidates the
> final test MRR is 0.2102 (EB-NeRD) / 0.2239 (MIND).

Two separate facts get conflated here, and only one is a problem:

- **"Exactly one positive per group" is normal**, not a pathology. It is the
  standard click-log ranking setup: put the one relevant item as high as
  possible out of 200. MRR is *defined* for exactly this case, and nDCG with a
  single positive reduces cleanly to `1/log2(rank+1)`. Both Codabench
  leaderboards score this same ~1-click-per-impression data, and NRMS trains on
  it.
- **"Zero positives in ~96% of groups" is the actual problem**, and it comes
  from the recall ceiling, not from single-click impressions.

Within the ~4% of groups where retrieval *did* find the click, the headroom is
large. Measured on EB-NeRD val (`bm25`, K=200, the 82 impressions whose click
was retrieved):

| where BM25 ranks the retrieved positive | count | share |
|---|---|---|
| rank 1 | 0 | 0.0% |
| rank 2–5 | 1 | 1.2% |
| rank 6–10 | 4 | 4.9% |
| rank 11–50 | 23 | 28.0% |
| rank 51–100 | 22 | 26.8% |
| rank 101–200 | 32 | 39.0% |

Median rank **80 of 200** — barely better than chance (100), and not a single
positive at rank 1.

| | MRR |
|---|---|
| before re-ranking (full population) | 0.0010 |
| ceiling: every retrieved positive at rank 1 | 0.0353 |
| **unclaimed headroom** | **0.0343 — 97% of the ceiling** |

So the re-ranker's opportunity is real and quantified: a potential **35x
relative MRR improvement**. The reason the gap is so wide is that BM25 was tuned
for *recall* (does the set contain the click) rather than *precision at the top*
(is it ranked first), and it scores on lexical overlap alone — it has no access
to freshness, CTR, category affinity, session position or dwell time, which is
precisely what our 24 features encode.

Two caveats to carry into the report:

1. **Absolute numbers stay small regardless.** Even a perfect re-ranker caps at
   MRR ≈ 0.035 because of §0's recall ceiling. Always report the relative gain
   *and* the ceiling together, or the result reads as a broken model rather than
   a bounded one.
2. **This makes Q3's significance test easier, not harder.** With 97% of the
   headroom unclaimed, a paired bootstrap 95% CI excluding zero should be
   comfortably achievable — beating median-rank-80 is a low bar for any model
   with genuine behavioural features.

---

## 1. Loss function: pointwise vs pairwise vs listwise

### 1.1 The three families

- **Pointwise** — predict an absolute relevance score per (impression,
  candidate) row independently; ranking is a side effect of sorting the scores.
  For binary clicks this is just binary classification (LightGBM
  `objective="binary"`). Knows nothing about groups.
- **Pairwise** — learn from *pairs* within a group: "positive should outrank
  this negative." Optimises the number of inversions.
- **Listwise** — take the whole group as one training example and optimise a
  list-level quantity directly (a softmax over the group, or a smoothed nDCG).

### 1.2 Clarification: RankNet / LambdaRank / LambdaMART are a lineage, not three GBDT options

Worth stating plainly, because listing them as three parallel choices is a
common trap:

- **RankNet** (2005) — pure pairwise cross-entropy on the probability that
  document *i* outranks *j*. Treats an inversion at ranks 1↔2 exactly like one
  at 100↔101, which is wrong for our metrics: MRR and nDCG@5 care almost
  entirely about the top of the list.
- **LambdaRank** (2006) — *not a different loss*, a different **gradient**. It
  takes RankNet's gradient and multiplies it by |ΔnDCG|, the change in nDCG from
  swapping that pair. That makes the gradient position-aware. There is no
  closed-form loss being minimised; the gradient is specified directly.
- **LambdaMART** (2010) — LambdaRank's gradients plugged into MART (gradient
  boosted trees). **This is what LightGBM's `objective="lambdarank"` is.**

So for a GBDT there aren't three pairwise choices. Choosing "pairwise for a
GBDT" *means* LambdaMART. RankNet is its ancestor and LambdaRank is the gradient
trick inside it.

Because LambdaMART weights every pair by a list-level quantity (ΔnDCG), it is
better described as **pairwise gradients with a listwise objective** — it
straddles the two families the question posed.

The genuinely listwise GBDT option is LightGBM's **`rank_xendcg`**
(XE-NDCG, Bruch et al. 2019): a softmax cross-entropy over the group against a
normalised relevance distribution. Reported as more robust than lambdarank when
relevance is binary and positives are sparse — which describes our data.

### 1.3 How our data shapes the choice

> **Note 2026-09-11 (§2E).** The "~96% of groups carry no positive" premise below
> belongs to catalogue candidates; with fresh-pool candidates most groups hold a
> positive. The LambdaMART decision (§1.4) is unaffected — it also rests on
> optimising the metric family we report, and on tooling.

**Pairwise/listwise silently discard groups with no positive.** A group whose
labels are all 0 generates zero pairs (LambdaMART) or a degenerate all-zero
target distribution (listwise softmax). Either way it contributes no gradient.
Given §0, that is **~96% of our groups**.

At first glance that looks disqualifying. It isn't, and this is the key
argument:

> A group with no positive carries **no ranking information**. You cannot learn
> "A should outrank B" from a set where nothing is known to be relevant.
> Pairwise/listwise dropping those groups is not lost signal — it is correctly
> recognising there was none.

And critically, **the metrics we report agree**. AUC, MRR and nDCG are computed
per impression and averaged. For an impression with no positive: AUC is
undefined (needs both classes), MRR is 0, and nDCG returns 0.0 by our
convention (`src/metrics.py`). Those groups contribute a **constant** to the
reported score *no matter how we order them*. They are unimprovable by
definition.

So the training signal that pairwise/listwise keeps is exactly the signal the
evaluation rewards. Pointwise's extra 96% of rows buys calibration of absolute
click probability — which no ranking metric measures.

**The one-positive-per-group fact matters too.** With exactly one relevant item
(EB-NeRD 99.7%, MIND 72.8%), nDCG@k collapses to `1/log2(rank+1)` for that
single positive and MRR to `1/rank`. Pairwise then reduces to "the positive vs
each of the 199 negatives" — which is *also* what a listwise softmax over the
group computes. The two families converge in behaviour here, so the choice
rests on robustness and tooling rather than on a deep modelling difference.

**Methodological bonus:** NRMS already trains with a listwise softmax (Wu-2019
sampling: 1 positive + npratio negatives, categorical cross-entropy). Choosing a
listwise-flavoured objective for the GBDT makes baseline and re-ranker
methodologically parallel, which is a cleaner comparison to write up.

### 1.4 Decision — **FINAL (signed off 2026-09-10)**

> **The method we are building: LambdaMART.**
>
> LightGBM `objective="lambdarank"` — RankNet pairwise gradients, each pair
> weighted by |ΔnDCG| from swapping it, boosted over regression trees (MART).
> Family, stated precisely for the report: **pairwise gradient, weighted by a
> listwise metric (nDCG)**. Not strictly listwise; not plain pairwise.
>
> LambdaMART *is* the GBDT — MART is Multiple Additive Regression Trees — so it
> is LambdaRank gradients **inside** a GBDT, not a GBDT layered on top of
> something else.
>
> No cross-encoder (§2.6). The ablation isolates the contribution of the Q1
> feature groups (§3).

With `rank_xendcg` as a cheap second arm and **pointwise `binary` as the
ablation baseline**.

Reasons:

1. **It optimises the metric family we report.** The ΔnDCG weighting spends
   model capacity on the top of the list, which is precisely what MRR and
   nDCG@5 reward. RankNet-style inversion counting does not.
2. **Pairwise/listwise's group-dropping is aligned with our evaluation**, per
   §1.3 — it is not the handicap it appears to be.
3. **It is the well-trodden LightGBM path**, so tooling risk is low and results
   are comparable to published LTR work.
4. **It hands Q3 a clean ablation for free.** Q3 requires one principled
   improvement plus an ablation with a paired bootstrap 95% CI excluding zero.
   "Pointwise logloss → LambdaMART" is a single-parameter, well-motivated,
   honestly-reportable ablation on identical features and identical splits.
5. `rank_xendcg` is a one-parameter third arm, and the literature suggests it
   may win specifically in our regime (binary, sparse positives).

Rejected: **RankNet** — position-blind, strictly dominated by LambdaMART for
top-heavy metrics, and not a first-class LightGBM objective anyway.

### 1.5 Implementation notes this decision implies

> **Superseded in part 2026-09-11 (§2E).** Pre-filtering to positive-bearing groups
> no longer shrinks anything with fresh-pool candidates. Training now uses 49
> random negatives per group (hard negatives were tried and hurt) and caps MIND at
> 60k impressions; evaluation runs in impression chunks and stays exact. "Before
> re-ranking" now uses stage-1 rank rather than `retrieval_score`, since
> popularity scores are tied integers.

- LightGBM needs a `group` array of contiguous group sizes; rows for one
  impression must be adjacent. Sort by `impression_id` before constructing the
  Dataset.
- **Pre-filter training groups to those containing ≥1 positive.** LightGBM would
  no-op on them anyway, so this is purely a speed/memory win — and a large one:
  it cuts MIND train from 27.7M rows to roughly 1.2M.
- **Do NOT filter at evaluation.** This is the trap already flagged in
  `src/nrms/adapter.py`: `drop_no_click` defines the impression *population*, so
  filtering one side of a before/after comparison makes the delta partly measure
  which impressions were included. Train on the filtered set; evaluate on the
  full population.
- Tie-breaking is already handled: `src/metrics.py::_descending_order` matches
  ebnerd-benchmark's `argsort(y_pred)[::-1]`. This matters more than usual for
  us because the **"before re-ranking" baseline scores integer retrieval ranks**,
  where ties are everywhere.
- Report **"n impressions with ≥1 positive"** next to every metric. Without it
  the numbers are unreadable and look like a broken model rather than a
  documented recall ceiling.
- ~~`lightgbm` is not yet in `requirements.txt`~~ — added (`lightgbm>=4.3`).

---

## 2. Cross-encoder + GBDT hybrid

### 2.1 The premise is correct

The described pattern is real and is what large production stacks do: a
cross-encoder jointly encodes (query, document) and emits a semantic relevance
score, which enters the GBDT as **one feature among many** alongside behavioural
and popularity signals. The GBDT arbitrates between semantic relevance and
everything else. Nothing wrong with the description.

### 2.2 Is it possible here? Yes — but the arithmetic kills it

A cross-encoder must run a transformer forward pass **per (impression,
candidate) pair** — that is the whole point; it cannot be precomputed per
article, because the score depends on the pairing.

Pairs we would need to score at K=200:

| dataset | train | val | test | total |
|---|---|---|---|---|
| MIND | 27.7M | 3.7M | 14.6M | **46.0M** |
| EB-NeRD | 4.5M | 0.5M | 5.1M | **10.0M** |
| | | | | **56.0M pairs** |

At an optimistic **2,000 pairs/sec** on a single GPU (MiniLM-class cross-encoder,
short titles, batched), that is **~7.8 GPU-hours**. At a more realistic
1,000/sec, **~15.6 hours** — and it must be redone from scratch if K, the text
fields, or the split definitions change.

Compare: the entire NRMS baseline is a couple of GPU-hours.

### 2.3 The recall ceiling defeats it anyway

> **Note 2026-09-11 (§2E).** This argument no longer holds: with fresh-pool
> candidates the click is in the set for 97.0% / 93.9% of impressions. The
> cross-encoder rejection still stands on §2.2 (compute) and §2.4 (latency); the
> cascade in §2.6 remains the route if it is ever wanted.

Even granting the compute: a cross-encoder improves the *ordering within* the
candidate set. In ~96% of our groups the clicked article **is not in the set at
all**. No amount of semantic scoring recovers an item that was never retrieved.
The expected metric gain is confined to the ~4% of groups where retrieval
already succeeded — and within those, we are ordering 200 items to surface one.

Spending 8–16 GPU-hours to slightly reorder sets that usually lack the target is
a poor trade against every other use of that time.

### 2.4 It also actively damages the Q4 story

Q4 asks for p99 retrieval latency and a cost/QPS estimate at a target SLA
(the doc suggests p99 < 100ms). A cross-encoder over 200 candidates per request
is on the order of 100ms–1s of pure GPU time *per query*, blowing that SLA by
one to two orders of magnitude. Including it would force us to either report a
failed SLA or explain a cascade we didn't build.

### 2.5 The cheap alternative that captures most of the value

We already have **bi-encoder** article embeddings from A1 (MiniLM for MIND,
provided multilingual BERT for EB-NeRD). A bi-encoder similarity feature —
cosine between the candidate's embedding and the user's pooled history
embedding — is **essentially free**: both vectors are precomputed, so it is one
normalised dot product per row, vectorisable over the whole matrix in seconds.

This is worth doing for an independent reason: it **closes a real gap**. A2 Q1.1
asks for click-history features over "titles, categories, **embeddings**", and
our current 24-feature set covers only the category channel. This has been
flagged as outstanding since the feature module was written. It is the single
highest-value feature still missing.

The trade is the standard one: a bi-encoder cannot model token-level interaction
between query and document the way a cross-encoder can, so it is a weaker
semantic signal — but it costs ~0.001% of the compute and closes a requirement
gap.

### 2.6 Decision — **FINAL (signed off 2026-09-10)**

**GBDT alone is sufficient for this assignment.** Add the **bi-encoder
similarity feature** (cheap, closes the Q1.1 embeddings gap). **Do not build the
cross-encoder.**

Write the cross-encoder up in the design note as *considered and rejected*, with
the §2.2–2.4 numbers. Q6 explicitly asks for "alternatives considered and why you
chose what you did", and a quantified rejection is stronger evidence of
engineering judgement than an unmotivated implementation would be.

If we later want the capability at defensible cost, the production answer is a
**cascade**: GBDT over all 200 → cross-encoder over only the top ~20. That cuts
cross-encoder work 10x and keeps the SLA plausible. Worth one paragraph in the
note as future work; not worth building now.

---

## 2B. First results — and why the headline numbers are not what they look like

EB-NeRD, BM25 candidates, K=200, test split (25,356 impressions). Trained on the
752 train impressions that contain a positive; evaluated over the full
population.

| metric | before (BM25 order) | after (LambdaMART) | change |
|---|---|---|---|
| AUC | 0.5377 | **0.9592** | +0.42 |
| MRR | 0.0010 | **0.0182** | 18x |
| nDCG@5 | 0.0006 | **0.0193** | 32x |
| nDCG@10 | 0.0009 | **0.0195** | 22x |

Within the 647 test impressions whose click was retrieved, MRR goes
0.0392 → 0.7148 — the positive moves from median rank 80 to roughly rank 1–2.
That is **71% of the available headroom** (§0.1) captured.

### Why this is not as good as it looks

**Feature importance (gain):**

| feature | gain |
|---|---|
| freshness_log_hours | **79.9%** |
| display_count_article | 10.2% |
| category_affinity | 1.4% |
| retrieval_score | 1.1% |
| everything else (18 features) | 7.4% |

Roughly 90% of the model is "how fresh is this article" plus "how often has it
been shown" — almost none of it is personalisation.

**Single-feature rankers, no model at all**, same population:

| ranker | AUC | MRR | nDCG@5 |
|---|---|---|---|
| freshness only (fresher first) | **0.8475** | 0.0011 | 0.0000 |
| display_count only | 0.4106 | 0.0007 | 0.0005 |
| retrieval_score only (= the BM25 baseline) | 0.5377 | 0.0010 | 0.0006 |
| full LambdaMART (22 features) | 0.9592 | 0.0182 | 0.0193 |

This splits the result cleanly in two, and the two halves deserve opposite
verdicts:

1. **The AUC is largely an artefact and should not be the headline.** Freshness
   *alone* reaches 0.8475 of the model's 0.9592. The reason is structural: the
   199 negatives are drawn from the **whole catalogue**, including articles long
   out of circulation, while the single positive is one the platform actually
   **displayed** and the user **clicked**. Recency nearly separates those two
   populations by itself. The model is substantially solving "which of these 200
   catalogue articles could plausibly have been on the site at that moment",
   which is a much easier and much less interesting question than "which will
   this user click".
2. **The MRR/nDCG gains are real and are not explained by freshness.** Freshness
   alone scores nDCG@5 = 0.0000 — sorting purely by recency never once puts the
   positive in the top 5. It is a good *coarse* filter (lifting the positive to
   roughly rank 30 of 200) and useless as a final ranker. Getting from rank 30 to
   rank 1–2 is what the other 20% of the gain buys, and that is the part the
   metrics we actually report depend on.

So the model is doing genuine work at the top of the list, and the coarse
separation it leans on is largely free.

### Consequences to carry into the report

> **Note 2026-09-11 (§2E).** The bound by retrieval below applies to catalogue
> candidates. With fresh-pool candidates absolute numbers are no longer capped by
> a 2.5% recall.

- **Report MRR and nDCG as the headline; treat AUC as diagnostic.** Here AUC is
  dominated by an easy, non-personalised discrimination and overstates the
  result badly.
- **Do not put this AUC beside NRMS's.** NRMS discriminates among ~20 articles
  the platform already chose to display; this discriminates a displayed article
  from catalogue randoms. Different task, different negative distribution, not
  comparable. (This is the unresolved Q3 framing issue in `todo.md`.)
- **Absolute numbers stay bounded by retrieval.** MRR 0.0182 against a ceiling
  of 0.0255 on this split — the remaining 96% of impressions are unreachable no
  matter how good the ranker gets.
- **The Q3 feature-group ablation is at risk of a null result.** With freshness
  and display-count carrying ~90% of the gain, ablating the behavioural feature
  group may barely move the metrics. That is a legitimate finding, but plan for
  reporting it rather than being surprised by it — and note it is partly an
  artefact of easy negatives, which better retrieval (see `todo.md`) would fix by
  making the negatives harder.

---

## 2C. NRMS baseline results (Q3.1) — trained by teammate

Both datasets, evaluated on our own temporal test split.

| | EB-NeRD (demo) | MIND (small) |
|---|---|---|
| AUC | 0.5425 | **0.6040** |
| MRR | **0.3394** | 0.2750 |
| nDCG@5 | **0.3770** | 0.2961 |
| nDCG@10 | **0.4570** | 0.3617 |
| impressions | 25,356 | 73,152 |
| epochs (cap) | 15/20 | 8/10 |
| early stopped | yes | yes |
| best val AUC | 0.5779 | 0.6585 |
| peak GPU (train) | 3,869 MiB | 780 MiB |
| peak GPU (eval) | 2,893 MiB | 4,544 MiB |
| wall time | 23 min | 77 min |

**Both runs early-stopped inside their epoch caps**, which is the gate that
matters: `train.py` warns when a run ends by exhausting epochs instead, because
an unconverged baseline makes any later "improvement" partly an artefact of
training longer. These are converged baselines, so a Q3 gain can be attributed
to the change rather than to extra training.

### This settles the comparability question empirically

The impression counts above (25,356 / 73,152) are **exactly our test splits** —
the same rows the re-ranker is evaluated on. So the population is identical and
the *only* thing that differs is the candidate set. That isolates the effect
cleanly:

| EB-NeRD test, 25,356 impressions | AUC |
|---|---|
| NRMS, scoring the **in-view set** (~20 platform-chosen articles) | 0.5425 |
| LambdaMART, scoring **A1's top-200** (catalogue-drawn) | 0.9592 |

The re-ranker is not 77% better than NRMS. It is solving a **much easier
discrimination**: telling an article the platform actually displayed apart from
199 catalogue articles, most of which were stale or out of circulation — which
§2B showed freshness alone nearly achieves (AUC 0.8475 with no model). NRMS is
discriminating among ~20 articles the platform had *already* judged plausible,
where recency carries almost no signal because they are all current.

Consequences, now backed by numbers on both sides:

- **Never put 0.9592 beside 0.5425 without this explanation.** A grader will
  read the raw pair as the re-ranker crushing the baseline, which is false.
- **NRMS's MRR/nDCG are far higher than the re-ranker's** (0.3394 vs 0.0182 on
  EB-NeRD) and that comparison *is* meaningful in the direction that matters:
  the two-stage pipeline is bounded by A1's ~3.5% recall, so it cannot approach
  a single-stage model that never discards the answer. This is the strongest
  available argument for the `todo.md` retrieval-recall work.
- Q3's "beat the baseline" therefore cannot mean "re-ranker beats NRMS on these
  numbers". It has to be either (a) an improvement to NRMS itself, or (b) a
  same-candidate-set before/after, which is the §2B result.

---

## 2D. All four configurations — results and what they reveal

Test split, K=200, LambdaMART. "Headroom" = achieved MRR as a fraction of the
ceiling (§0.1), i.e. how much of what retrieval made *reachable* was claimed.

| config | recall@200 | AUC | MRR | nDCG@5 | nDCG@10 | headroom | iters | train impr |
|---|---|---|---|---|---|---|---|---|
| ebnerd/bm25 | 0.0255 | 0.5377 → **0.9592** | 0.0010 → **0.0182** | 0.0006 → 0.0193 | 0.0009 → 0.0195 | **71.5%** | 17 | 752 |
| ebnerd/semantic | 0.0249 | 0.4774 → **0.9421** | 0.0006 → **0.0180** | 0.0002 → 0.0192 | 0.0003 → 0.0193 | **72.3%** | 13 | 671 |
| mind/bm25 | 0.0290 | 0.6220 → **0.6033** ⚠ | 0.0013 → 0.0032 | 0.0007 → 0.0029 | 0.0014 → 0.0036 | 10.9% | 80 | 23,589 |
| mind/semantic | 0.0387 | 0.5348 → 0.5888 | 0.0017 → **0.0041** | 0.0012 → 0.0037 | 0.0015 → 0.0046 | 10.6% | 11 | 8,528 |

### Finding 1 — the retrieval *method* barely matters; the *dataset* decides everything

Within a dataset the two retrievers land in almost the same place (EB-NeRD
71.5% vs 72.3% headroom, MRR 0.0182 vs 0.0180; MIND 10.9% vs 10.6%). Across
datasets the gap is ~7x. So swapping BM25 for embedding retrieval buys
essentially nothing once a re-ranker sits on top — the re-ranker washes the
difference out, because it re-scores the same 200 items either way and both
retrievers surface a similarly *reachable* set.

What does matter is retrieval **recall**, which is a hard ceiling no re-ranker
can cross. Practical implication for the `todo.md` work: effort spent making
stage 1 recall higher is worth far more than effort spent choosing between BM25
and embeddings.

### Finding 2 — MIND/BM25's recall collapses across the temporal split

Recall@200 by split, the whole reason MIND/bm25 is the only config whose AUC
*degrades*:

| config | train | val | test | train/test |
|---|---|---|---|---|
| mind/bm25 | 0.1702 | 0.0634 | 0.0290 | **5.9x** |
| mind/semantic | 0.0615 | 0.0592 | 0.0387 | 1.6x |
| ebnerd/bm25 | 0.0336 | 0.0353 | 0.0255 | 1.3x |
| ebnerd/semantic | 0.0300 | 0.0456 | 0.0249 | 1.2x |

MIND/bm25 trains on a population where retrieval succeeds **17%** of the time
and is evaluated where it succeeds **2.9%** — a train/eval distribution mismatch
severe enough to make the model actively worse than the ordering it started
from. The others are stable at 1.2–1.6x.

The mechanism is news turnover: BM25 matches the *lexical* content of older
history, and as the news cycle advances the articles actually being shown share
less vocabulary with what the user read days ago. Embeddings degrade far more
gracefully (1.6x), and **MIND/semantic's test recall (3.87%) beats BM25's
(2.90%) despite being 3x lower on train** — semantic retrieval generalises
across time much better here. Worth stating in the report: it inverts the
train-set ranking.

### Finding 3 — feature importance tracks data availability, giving a free ablation

> **Caveat 2026-09-11.** For catalogue candidates this "natural ablation" is
> confounded. EB-NeRD's `has_known_publish_time` flags articles published *after*
> the impression — 0% of clicked articles versus 10.7% of the catalogue — which
> makes it a perfect negative indicator that could not exist at serving time; MIND
> has no such shortcut. So part of EB-NeRD's freshness weight here may be that
> shortcut rather than better freshness data. Under fresh-pool candidates the
> shortcut is gone (0.003% of rows), and the EB-NeRD/MIND gap persists for a
> different reason (§2E).

| config | top features |
|---|---|
| ebnerd/bm25 | freshness **80%**, display_count 10%, category_affinity 1% |
| ebnerd/semantic | freshness **87%**, display_count 7%, click_count_article 1% |
| mind/bm25 | freshness 34%, display_count 23%, click_count_article 22% |
| mind/semantic | display_count **48%**, click_count_article 29%, freshness 12% |

EB-NeRD ships a real `published_time`, and its models put 80–87% of their weight
on freshness. MIND has no publish date — ours is the proxy "first seen as a
candidate in train" — and freshness drops to 34% and then 12%, with popularity
counts taking over.

**The two datasets therefore form a natural ablation of freshness quality**, and
it lines up exactly with performance: the configs with real freshness capture
~72% of headroom, the ones with proxy freshness capture ~11%. That is the
single clearest signal in these results, and it is worth more than an artificial
feature-drop ablation because nothing was held out by hand — the datasets simply
differ in what they record.

### Finding 4 — more training data did not help, at all

MIND/bm25 trains on **23,589** impressions to EB-NeRD/bm25's **752** — 31x more
— and captures a seventh of the headroom. MIND/semantic (8,528) likewise loses
to EB-NeRD/semantic (671). This is not a data-quantity problem, and no amount of
extra MIND impressions would fix it. It is a feature-quality problem (Finding 3)
compounded by a distribution shift (Finding 2).

Corollary for tuning: `best_iteration` says the same thing. EB-NeRD stops at
13–17 rounds on <800 groups, MIND/bm25 grinds to 80 — spending far more capacity
to extract far less.

### Finding 5 — "before" AUC below 0.5

EB-NeRD/semantic starts at AUC **0.4774**: stage 1 orders its own candidates
*worse than random* with respect to what was clicked. Its retrieval score is
anti-correlated with relevance, which is a striking statement about the gap
between "similar to the user's history" and "the thing they clicked". It still
reaches 0.9421 after re-ranking.

### Timings (measured, this machine)

| stage | time |
|---|---|
| EB-NeRD artifact download (361 MB) | ~7 min at 1.2 MB/s (was ~1h40m before a network change) |
| MIND MiniLM embeddings (65,238 articles, CPU) | ~10 min |
| EB-NeRD embeddings (load provided artifact) | seconds |
| MIND bm25 candidates (3 splits) | ~1m40s |
| MIND semantic candidates (3 splits) | ~15 min |
| EB-NeRD bm25 candidates (3 splits) | ~45s |
| EB-NeRD semantic candidates (3 splits) | ~3 min |
| re-ranker mind/bm25 | 7m43s |
| re-ranker mind/semantic | 2m16s |
| re-ranker ebnerd/* | <1 min each |

Note on benchmarking: an initial estimate of 77 min for MIND semantic retrieval
came from timing a sample *while the real job was running*, and overstated it by
5x — the contended sample never saw the BLAS threading the real run had. Do not
benchmark under contention.

---

## 2E. Stage-1 recall fix: fresh-pool candidates (2026-09-11)

§0's recall ceiling turned out not to be inherent. It came from *where* stage 1
searched and *what* it ranked by.

### Diagnosis

Clicks go to what is in circulation right now: EB-NeRD's median clicked-article
age at click time is **3.1 h**, and 92% are clicked within 24 h of publication.
A1 searched a catalogue reaching back to **2000**. Two separate mistakes
compounded:

1. **Wrong search space.** Restricting A1's *unchanged* BM25 to articles the
   platform displayed in the previous hour lifts EB-NeRD recall@200 from 2.55%
   to 84.2%.
2. **Wrong ranking signal.** Ranking that same pool by recent popularity reaches
   97.0% / 93.9%. History similarity adds nothing on top.

### Stage-1 ablation — test recall@K, full populations

| arm | EB-NeRD @50 / @100 / @200 | MIND @50 / @100 / @200 |
|---|---|---|
| `bm25` — A1, whole catalogue | 0.82 / 1.41 / **2.55%** | 1.27 / 1.96 / **2.90%** |
| `bm25_fresh` — A1 scoring, 1 h pool | 21.9 / 42.9 / **84.2%** | 10.0 / 14.2 / **21.0%** |
| `popular` — 1 h pool, lagged clicks | 83.4 / 93.8 / **97.0%** | 81.6 / 88.6 / **93.9%** |

`bm25` reproduces §0's figures exactly, so the arms are like-for-like.

The pool (`src/fresh_pool.py`) is built only from events strictly before the
impression time *t*: displays in [*t* − 1 h, *t*); clicks from impressions in
[*t* − 1 h, *t* − 10 min), because a click lands after its impression; and a 24 h
display backfill when the 1 h pool is shorter than K. On 2,000-impression
samples, 1 h beat 6 h and 24 h (longer windows let stale-but-popular articles
crowd out fresh ones), and the 10-minute lag cost 0 pp on EB-NeRD and ~1 pp on
MIND.

Two findings worth carrying into the report:

- **History similarity is too sparse to rank a fresh pool.** On MIND a title
  query scores a median of only **~20 of ~2,000** pool articles above zero, so
  `bm25_fresh`'s top-200 is ~90% ties, and its recall swung from 10% to 92%
  depending on the tie rule. It now breaks ties with a seeded random draw — the
  only rule under which the arm measures BM25 and nothing else. Insertion order
  smuggles in recency; popularity ties turn the arm into `popular`.
- **The future-article shortcut is gone.** Catalogue retrieval could return
  articles published after the impression (10.7% of the catalogue per EB-NeRD
  test impression), and `has_known_publish_time` flagged them perfectly. In
  `popular` candidates that falls to 0.003% of rows, and all of those come from 4
  articles the platform *displayed* before a bulk re-stamp of their
  `published_time` on 2023-06-29 — a data quirk, not a leak. The guarantee is
  "displayed strictly before *t*", not "published before *t*".

### The re-ranker needed two fixes before it beat stage 1

**Scale first.** With ~94–97% of groups now holding a positive, §1.5's
positives-only filter no longer shrinks anything, and the feature builder's
dicts cost ~1.8 KB/row against ~5 GB of free RAM. Feature rows are now flushed
every 250k; training subsamples negatives; MIND training is capped at 60k
impressions (§2D Finding 4 showed more MIND data did not help); evaluation runs
in impression chunks and still scores every candidate, so metrics stay exact.
"Before re-ranking" is now stage-1 *rank* rather than `retrieval_score`, because
popularity scores are tied integers and sorting on them would scramble the ties.

**Bug 1 — train/serve skew in the Q1 article statistics.** `TrainEventIndex`
froze val/test statistics at the end of train. Train rows got live, accumulating
counts; test rows got a snapshot in which the fresh articles actually clicked
all read zero. The median `click_count_article` of clicked articles was **16 on
train, 0 on val and test**, and 99.2% of clicked test articles read exactly zero.
The model learned "many clicks → clicked" and then buried fresh articles. Fix:
the re-ranker builds the index over every earlier event
(`splits=("train", "val", "test")`) with the same 10-minute click lag. That is
what a live system's counters hold, and it is still strictly before *t*. The CTR
prior stays train-only, and Q1's default behaviour is unchanged.

**Bug 2 — hard-negative sampling.** Keeping each group's 24 best-ranked negatives
over-represents exactly the part of the list where stage-1 rank separates the
positive least well, so the model learned rank was weak and ranked worse than
stage 1 over all 200. Random negatives keep the rank distribution representative.

EB-NeRD, chosen on val, test reported alongside:

| configuration | val MRR | test MRR | test nDCG@5 |
|---|---|---|---|
| stage-1 popularity (before) | 0.1690 | 0.1639 | 0.1471 |
| train-only stats, hard24 + rand25 | — | 0.1291 | 0.0994 |
| all-splits stats, hard24 + rand25 | 0.1447 | 0.1415 | 0.1139 |
| all-splits, hard24 + rand25, candidate-level features only | 0.1020 | 0.1018 | 0.0696 |
| all-splits, full groups (no sampling) | 0.2187 | 0.2085 | 0.1974 |
| **all-splits, 49 random negatives (chosen)** | **0.2249** | **0.2102** | **0.2007** |

Random sampling edges out full groups at a quarter of the rows. Dropping the
impression-level features hurt badly — but that arm ran under the discarded
hard-negative sampling, so it is not the clean feature-group ablation.

### Final results — test, K = 200, before = stage-1 popularity order

| | EB-NeRD | MIND |
|---|---|---|
| impressions / with the click in the top-200 | 25,356 / 97.0% | 73,152 / 93.9% |
| AUC | 0.8815 → **0.9213** | 0.8655 → 0.8799 |
| MRR | 0.1639 → **0.2102** (+28.2%) | 0.2237 → 0.2239 (+0.1%) |
| nDCG@5 | 0.1471 → **0.2007** (+36.5%) | 0.2451 → 0.2450 (−0.0%) |
| nDCG@10 | 0.1994 → **0.2691** (+34.9%) | 0.2932 → 0.2980 (+1.6%) |
| training impressions / rows | 21,897 / 1.09M | 54,222 / 2.73M |
| top features (gain) | freshness 53%, position_bias 23% | position_bias 57%, retrieval_rank 15% |

Against the catalogue pipeline (§2D), MRR goes 0.0182 → **0.2102** on EB-NeRD and
0.0032 → **0.2239** on MIND.

- **EB-NeRD:** the re-ranker adds a lot on top of popularity, mostly through
  freshness, which rests on a real `published_time`.
- **MIND:** a wash. 72% of the model's gain is stage-1 rank; it learned to trust
  stage 1. MIND has no publish date, and within an hour-old pool "first seen" is
  close to uniform across candidates, so the Q1 features carry little beyond
  popularity. Same direction as §2D Finding 3, though no longer explained by the
  future-article shortcut, which is gone.

### Caveats

- **No paired bootstrap CIs yet** (Q3 requires them). EB-NeRD's +28% over 25k
  impressions is very likely significant; MIND's +0.1% is not a gain.
- The pool leans on the platform's own display log. That is legitimate at serving
  time, but it rides the platform's recommender. An EB-NeRD variant with no logs
  at all — newest-published first — reached 94.0% @200 on a sample.
- The 10-minute click lag is an assumption; no click timestamps exist to
  calibrate it.
- **Still not comparable with NRMS.** MRR over 200 candidates is not NRMS's MRR
  over the ~9–23 in-view articles. A fair head-to-head needs identical candidate
  sets.

---

## 3. Open decisions

- [x] ~~Sign-off on §1.4 (LambdaMART primary) and §2.6 (no cross-encoder).~~
      **Done 2026-09-10.** Ablation is over the Q1 feature groups.
- [x] ~~Raising A1's retrieval recall.~~ **Done 2026-09-11** (§2E): fresh-pool
      popularity candidates, recall@200 97.0% (EB-NeRD) / 93.9% (MIND).
- [x] ~~MIND train scale.~~ **Resolved 2026-09-11** (§2E): with ~94% of groups
      holding a positive, pre-filtering no longer shrinks anything. Training uses
      49 random negatives per group and caps MIND at 60k impressions; evaluation is
      chunked and exact.
- [x] ~~Which candidate source feeds the re-ranker.~~ **Resolved 2026-09-11**:
      `popular` (§2E). `bm25` and `bm25_fresh` stay as stage-1 ablation arms.
- [ ] Whether the Q3 "principled improvement" is the loss-function change
      (pointwise → LambdaMART) or the bi-encoder feature. Either works; they
      should not be conflated into one ablation arm.
- [ ] Paired bootstrap 95% CIs for the §2E before/after deltas.
- [ ] The Q1 feature-group ablation under the chosen random-negative sampling
      (§2E's candidate-level-only arm ran under the discarded hard negatives).
- [ ] A fair Q3 head-to-head with NRMS on identical candidate sets.

## 4. Changelog

- **2026-09-10** — Created. Loss-function analysis (§1) and cross-encoder
  analysis (§2) written against measured group statistics and A1's recorded
  recall figures.
- **2026-09-10** — Added §0.1: measured the re-ranking headroom rather than
  assuming it. BM25 ranks its own retrieved positives at median 80/200 with none
  at rank 1, leaving 97% of the achievable MRR unclaimed. Confirms that
  single-positive groups are not the problem and that the re-ranker has a real,
  quantified opportunity.
- **2026-09-11** — Added §2E: the stage-1 recall fix (fresh-pool candidates,
  recall@200 2.55% / 2.90% → 97.0% / 93.9%), the two re-ranker bugs it exposed
  (train/serve skew in the article statistics; hard-negative sampling), and the
  final results. Resolved the recall, MIND-scale and candidate-source decisions;
  added CIs, the clean feature-group ablation and the NRMS head-to-head as open.
- **2026-09-11** — Added the "Current state" block and dated supersession notes to
  §0, §0.1, §1.3, §1.5, §2.3, §2B and §2D Finding 3 where §2E overturned their
  premises; marked the lightgbm requirement done.
