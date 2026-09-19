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
> *(2026-09-19, §2J: with the two features a live system would not have removed,
> EB-NeRD is **0.1915, +17%** over stage 1, CI clear of zero. Quote both.)*
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

- **No paired bootstrap CIs yet** *(done 2026-09-19, §2J: EB-NeRD MRR +0.0463
  [+0.0430, +0.0495] over stage 1, +0.0275 serving-safe)*. EB-NeRD's +28% over 25k
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

## 2F. Q3 improvement: freshness weighting for NRMS (2026-09-11)

Implemented, tested, **not yet trained** — training runs on a separate machine.

### Why freshness, over the two alternatives considered

The proposal on the table was **positional encoding** on the user encoder.
NRMS is permutation-invariant in two places, and the distinction matters:

1. **Word level** — `NewsEncoder` self-attends over 30 title tokens, so
   "dog bites man" ≡ "man bites dog".
2. **History level** — `UserEncoder` self-attends over 20 clicked-article
   vectors, so the most recent click is treated identically to the oldest.

Measured history occupancy, test split, at `HISTORY_SIZE = 20`:

| | MIND | EB-NeRD |
|---|---|---|
| impressions with <20 clicks | **51.1%** | 4.3% |
| mean PAD slots (of 20) | **5.9 → 29.4% of the input** | 0.4 → 1.8% |
| median history length | 19 | 226 |

That rules positional encoding out as a *primary* arm: on MIND, 29.4% of the
history slots are padding occupying positions 0…k, and `layers.py` does no
masking, so PE would be attaching position embeddings to padding. **PE and
masking are complementary, and PE without masking is compromised on exactly the
dataset where padding dominates.**

Freshness was chosen instead on a **category** difference rather than a
magnitude one: the padding problem is something the additive attention can
partly learn around (the pad vector is constant, so attention can down-weight
it), whereas **article age is not in NRMS's input at all** — it cannot be
learned from title tokens at any amount of training. Adding a channel the model
structurally lacks is a surer bet than fixing one it can partly compensate for.

The supporting evidence is all in-project: the Q2 re-ranker put **80–87%** of
its gain on `freshness_log_hours` for EB-NeRD (§2B/§2D), and §2E measured a
median clicked-article age at click time of **3.1 hours**. "Freshness weighting"
is also on the A2 doc's own list of suggested improvements.

Ranked, for the record: freshness > attention masking (zero new parameters, big
on MIND, negligible on EB-NeRD) > positional encoding.

### Design

`score = dot(user_vector, candidate_vector) + g(log1p(age_hours), known)`, with
`g` a 2→16→1 MLP (`FreshnessHead`).

Additive on the **logit**, not concatenated into the news vector: that leaves
the dot-product scorer that *defines* NRMS untouched, so the arm isolates "does
knowing article age help" rather than also changing how the two representations
interact.

Reference times come from `article_stats.freshness_reference_times` —
EB-NeRD's real `published_time`, MIND's earliest recorded sighting — over
`config.ARTICLE_STATS_SPLITS`, the same splits the re-ranker's `TrainEventIndex`
uses. *(Corrected 2026-09-11: as first committed this was true for EB-NeRD
only; MIND used train-only sightings. See "Corrections after review" below.)*

### Three properties that protect the ablation's validity

1. **The head is zero-initialised**, so at step 0 its output is exactly 0 and the
   arm's logits are bit-identical to the baseline's. The arm can only depart from
   the baseline by learning — a measured gain can never be an artefact of a
   different initialisation.
2. **The head is constructed only when the flag is set.** Building it
   unconditionally would draw from the RNG even when disabled, shifting every
   subsequent parameter's initialisation and silently stopping the baseline arm
   from reproducing the §2C runs it is compared against.
3. **~65 parameters** against ~192M in the embedding matrix (<1e-6). No gain can
   be attributed to added capacity.

All three are tested (`tests/test_freshness.py`).

### Leakage (Q9)

The as-of gate is in `ages()`: a reference counts only when it **strictly
precedes** the impression being scored. An article whose first recorded sighting
falls *after* that impression reads `known=0`, never age-zero — clamping it to
zero would tell the model "brand new" about something we simply had not seen
yet, which is the same failure the re-ranker's freshness proxy was fixed for.
`PAD_CODE` has no reference by construction, so padded slots are always unknown.

### CLI

```bash
python run_nrms.py --dataset ebnerd --stage all                   # baseline, path unchanged
python run_nrms.py --dataset ebnerd --stage all --freshness       # -> nrms/freshness/
python run_nrms.py --dataset ebnerd --stage all --freshness --seed 43 --run-tag freshness_s43
```

`--run-tag` defaults to `freshness` when `--freshness` is set and to the
artifact root otherwise, so an arm can never silently overwrite the baseline
while the existing baseline path stays backwards compatible. Shared caches
(codec, token matrix) stay at the artifact root and are reused across arms.

### Verified on real data, EB-NeRD val

- reference time known for **11,777 / 11,777** catalogue articles (100%)
- candidate age: **median 2.2 h, p90 3,344 h** — a ~1,500x spread the baseline
  cannot represent at all
- both arms train and evaluate end to end on CPU; the head's weights move, so
  gradients reach it

### Predictions to check against when the runs land

*(Both runs landed 2026-09-11 — see "Results" below. Prediction 1 was right in
direction and wrong in mechanism; prediction 2 was overtaken by the effect size
on EB-NeRD.)*

- **Expect an asymmetry.** EB-NeRD has a real `published_time`; MIND has only the
  first-seen proxy, so a weaker and noisier result there is the prediction, not a
  bug — and it would mirror §2D's Finding 3 exactly, giving one coherent story
  across both models.
- **One seed per arm will not settle a small gain.** A paired bootstrap resamples
  test *impressions* with both models held fixed; it says nothing about training
  variance. If the effect is small, run seeds 42/43/44 at least on EB-NeRD
  (~23 min each) so the gain can be told apart from seed noise.

### Corrections after review (2026-09-11)

A review of the first commit found two real bugs, both verified independently
before fixing.

**1. MIND dated almost none of its test candidates.** `FreshnessLookup.build`
defaulted to train-only sightings, while the re-ranker already used every split.
On MIND — whose reference is "first time shown" — any article first appearing
after the train period had no age at all. Measured on 5,000-impression samples:

| split | reference from | candidates with an age | clicks with an age |
|---|---|---|---|
| train | train only | 99.6% | 99.6% |
| val | train only | 90.4% | 88.8% |
| test | train only | **25.1%** | **17.4%** |
| test | all splits | 99.9% | 99.9% |

Three reasons this mattered more than a coverage gap:

- It is the §2E bug 1 pattern again — train/serve skew. The head would learn age
  on 99.6% dated candidates and be tested where 75% are undated, with the
  (fresh) clicked articles hit hardest.
- **The equivalence test was vacuous.** It compared the two functions' train-only
  *defaults*, so it passed while the property it guarded — "same definition as
  the re-ranker" — was false for MIND.
- **The §2F prediction would have come true for the wrong reason.** "MIND gains
  less because first-seen is a noisy proxy" is exactly what this bug produces on
  its own, and it would have been misattributed. And because val is only mildly
  affected (90%), early stopping would have looked healthy — nothing during
  training would have flagged it.

Fix: the splits tuple moved from `reranker.py` to `config.ARTICLE_STATS_SPLITS`,
and both the re-ranker and `FreshnessLookup.build` (as its default) read it. The
root cause was that the tuple lived where the NRMS path could not see it; with one
constant the "same definition" claim is structurally true rather than something a
test has to keep catching. The equivalence test now reads that constant on both
sides. Leak-free: `ages()` only uses a reference strictly before the impression.
EB-NeRD is unaffected (its reference is `published_time`).

**2. The two arms trained on different data orders.** Validity property 1 above
claimed the arm "can only depart from the baseline by learning". That was
overstated: conditional, last-position construction kept every other parameter's
init identical, but the head's own init still drew from torch's global generator,
and the next readers of that generator are the DataLoader's shuffle seed and CPU
dropout. With the same `torch.manual_seed(42)` the shuffle seeds differed
(`7773910853395796049` vs `5950622100989377681`), so a gain would have been mixed
with seed noise — decisive with one seed per arm and a small effect. Negative
sampling was unaffected (it uses its own seeded numpy generator).

Fix: the head is built inside `torch.random.fork_rng(devices=[])`, so it consumes
nothing from the global generator; both arms now draw identical shuffle seeds.
CPU-only forking is sufficient because construction runs on CPU and GPU dropout
uses the CUDA generator. Two tests assert the generator state and shuffle seed
match across arms.

Also corrected: the head is **65** parameters, not the "~50" the code comments
said.

**Run order after these fixes:** both arms are valid on both datasets. Before
them, only EB-NeRD's age signal was sound, and neither dataset had matched data
order.

### Results (2026-09-11) — trained on the RTX 5050 machine

Both arms, both datasets, same seed (42), same epoch caps and same test splits as
the §2C baselines. Every run early-stopped inside its cap, so no comparison is
confounded by one arm simply training longer. W&B: `rn5oq50v` (EB-NeRD),
`050iyvjo` (MIND).

| test split | EB-NeRD base | EB-NeRD +fresh | Δ (95% CI) | MIND base | MIND +fresh | Δ (95% CI) |
|---|---|---|---|---|---|---|
| AUC | 0.5425 | **0.6112** | **+0.0686** [+0.0662, +0.0710] | 0.6040 | 0.6042 | +0.0002 [−0.0006, +0.0010] |
| MRR | 0.3394 | **0.3962** | +0.0567 [+0.0538, +0.0595] | 0.2750 | 0.2743 | −0.0007 [−0.0014, +0.0001] |
| nDCG@5 | 0.3770 | **0.4427** | +0.0657 [+0.0628, +0.0685] | 0.2961 | 0.2949 | −0.0012 [−0.0021, −0.0003] |
| nDCG@10 | 0.4570 | **0.5095** | +0.0525 [+0.0501, +0.0548] | 0.3617 | 0.3610 | −0.0007 [−0.0015, +0.0001] |
| best val AUC | 0.5779 | 0.6407 | +0.0629 | 0.6585 | 0.6612 | +0.0027 |
| epochs (cap) | 15/20 | 15/20 | — | 8/10 | 8/10 | — |
| wall time | 23 min | 18 min | — | 77 min | 52 min | — |
| peak GPU train / eval | 3,869 / 2,893 MiB | 3,872 / 2,894 MiB | — | 780 / 4,544 MiB | 785 / 4,546 MiB | — |

CIs are a paired bootstrap over test **impressions** (2,000 resamples, seed 0,
both models held fixed), scoring the identical impression population both sides —
so they bound sampling noise in the test set, **not** training variance.
Reproduce with `python scripts/paired_bootstrap.py --dataset ebnerd`; the
diagnostics in the next section come from `python scripts/age_signal.py`. Both
reuse `src/metrics.py`, so they cannot drift from `metrics_test.json`.

**EB-NeRD: a large, unambiguous gain.** +0.069 AUC, with the CI nowhere near
zero and per-impression win/tie/loss of 46.2% / 35.1% / 18.6%. It is visible from
epoch 1 (val AUC 0.5795 vs 0.5452), which is what the zero-init property
predicts: the arms start bit-identical, so the gap can only be learned.

**MIND: no effect.** Every delta is within a few ten-thousandths and the AUC CI
spans zero; per impression the arms win and lose about equally (32.0% / 38.3% /
29.7%). Only nDCG@5's CI excludes zero, as a *loss* of 0.0012 — one marginal
exclusion across four correlated metrics, and not a result worth a story.

### Why the MIND arm gained nothing — and how far that generalises

The obvious suspect is the bug fixed above, but coverage is now fine: **99.9%**
of MIND test candidates and 99.9% of test clicks carry an age. (The run log's
"34.9% of catalogue articles" is a different and harmless number — most catalogue
articles never appear in a behaviour log at all; the ones actually shown are
dated.) What differs is the signal itself, measured on each test split with
`scripts/age_signal.py`:

| test split | EB-NeRD | MIND |
|---|---|---|
| AUC of the learned head used *alone* as a ranker | **0.6765** | **0.5142** |
| median age, clicked vs not-clicked | 3.1 h vs 4.7 h | 15.0 h vs 16.5 h |
| median within-impression age spread (std) | **1,057 h** | **27.5 h** |
| candidates / clicks carrying an age | 100% / 100% | 99.9% / 99.9% |

EB-NeRD impressions mix minutes-old articles with weeks-old ones; MIND
impressions are internally near-uniform in age, and an additive term that is
nearly constant across a row cannot reorder it. The learned curves match:
EB-NeRD's peaks at ~6 h and falls away on both sides, while MIND's is monotone
and shallow.

**Worth knowing for the report: "newest first" is not the rule.** Ranking
EB-NeRD's candidates by recency alone scores AUC **0.4950** — chance. The signal
is a *band* around ~6 h, which only a learned curve can express. Any claim that
"recency helps" should say age-band, not recency.

### The claim this licenses: freshness-**as-first-seen** fails on MIND

MIND has no publish dates, so its age is the earliest-sighting proxy, and that
proxy is right-censored: an article already in circulation when logging began
cannot be dated earlier than the log window. So the defensible claim is
**"freshness-as-first-seen doesn't help on MIND"**, not "freshness doesn't help
on MIND". With real publish dates the within-impression spread could be far
larger and the conclusion could change. The report must not overreach here.

EB-NeRD is the only dataset that can price the proxy, having both definitions.
Feeding the *same* trained curve first-seen ages instead of real ones
(`scripts/age_signal.py --dataset ebnerd --reference first-seen`):

| EB-NeRD test, one fitted curve, two age definitions | `published_time` | first-seen proxy |
|---|---|---|
| head-alone AUC | 0.6765 | **0.6631** |
| median within-impression spread | 1,057 h | **52.7 h** |
| candidates dated | 100% | 99.6% |
| median understatement vs truth | — | 0.1 h (13.6% understated by >24 h) |

So the proxy compresses the spread ~20x yet costs only **0.013 AUC**: it stays
accurate in the 0–12 h band where the discrimination actually happens, and only
the old tail is clamped. On the one dataset where this is checkable, the proxy is
not what kills the signal — which makes "MIND's in-view sets are genuinely
age-homogeneous" the better-supported explanation of the null.

That evidence narrows the uncertainty; it does not remove it. It is EB-NeRD
evidence about an EB-NeRD-shaped age distribution, and MIND's censored share
cannot be measured without the publish dates it does not ship. **State the scoped
claim, cite the transfer test as the reason a proxy artefact is the less likely
explanation, and leave it there.**

### The honest caveat for the report

On EB-NeRD the head **alone** ranks at AUC 0.6765, above the full freshness NRMS
at 0.6112. Age is simply a stronger signal on this split than NRMS's title
encoder, whose §2C baseline sits at 0.5425 and whose val loss rises from epoch 6
while train loss keeps falling; mixing the two dilutes the stronger one.

**"Freshness weighting improves NRMS by +0.069 AUC on EB-NeRD" is true and
correctly measured**, and the ablation is clean by construction (identical init,
identical data order, 65 parameters). But it must be reported next to the
age-only number, or it overstates what the *model* contributes. It is the same
lesson as §2B, where freshness alone reached AUC 0.8475 with no model at all.

### Seed count — resolved

§2F flagged that one seed per arm cannot separate a small gain from seed noise.
EB-NeRD's +0.069 is an order of magnitude larger than plausible seed-to-seed
variation, so extra seeds are not needed to support the claim. MIND's null is
likewise not a seed question: the diagnostic above shows there is no signal to
find *under the first-seen definition*, and repeating it would only re-measure
zero. Seeds 43/44 stay available
(`--freshness --seed 43 --run-tag freshness_s43`) if a reviewer asks.

---

## 2G. Q4 — serving and scale (2026-09-15)

Measured with `scripts/serving_benchmark.py`, which writes
`results/serving_<dataset>.json`. 300 timed requests after 50 warm-up, K=200,
CPU only, single-threaded, on this laptop.

### What counts as a request, and what does not

Stage 1 is a 1-hour window over the display log. `FreshPool.advance_to(t)`
replays the log forward, but a live system never does that inside a request — it
maintains those counters as impressions arrive. Timing it would measure log
catch-up, not serving, so it is excluded from request latency and reported
separately (measured: **0.065 ms per impression**, so it would not have mattered
either way). Session counters are treated the same way: a live system holds them
in a session store, so sessionizing the split is startup here. This is a
judgement call and is stated rather than buried.

What *is* timed is what a request does against state already in memory: rank the
current pool, build features for the top-K, score them.

### Q4.2 Latency — p99 is ~10x inside the SLA

| | stage 1 | stage 2a features | stage 2b scoring | **total** |
|---|---|---|---|---|
| EB-NeRD p50 / p99 | 0.33 / 0.56 | 4.23 / 8.12 | 0.37 / 0.60 | **4.95 / 9.15 ms** |
| MIND p50 / p99 | 1.15 / 1.83 | 4.01 / 7.32 | 0.54 / 0.80 | **5.74 / 9.58 ms** |

**Feature building is ~85% of the request; the model is ~7%.** That is the
result worth carrying into the report, because the intuitive answer is the
opposite. The GBDT is not the bottleneck — assembling its inputs is. Any latency
work should target `build_candidate_features`, not the model.

### Q4.3 Throughput and cost

| | single-core QPS | p99 vs 100 ms SLA | USD / 1000 queries |
|---|---|---|---|
| EB-NeRD | 190.9 | 9.15 ms — **pass** | $0.000247 |
| MIND | 165.1 | 9.58 ms — **pass** | $0.000286 |

Priced at $0.17 per vCPU-hour with the core saturated; the rate is recorded in
each JSON's `config` block so the figure can be re-derived. Real serving
provisions for peak, so divide by target utilisation to size a fleet.

### Q4.1 Index memory

Each component built in its own process, because sequential in-process RSS
deltas are contaminated by pandas freeing buffers — the first version of this
script reported the article index at *minus* 126 MiB. `explicit_mib` is the
exact array bytes; `rss_mib` includes Python object overhead and build
transients.

| component | EB-NeRD explicit | MIND explicit | in the shipped path? |
|---|---|---|---|
| `semantic_ann_index` | 34.5 MiB | **95.6 MiB** | no — `semantic` arm only |
| `bm25_inverted_index` | 4.1 MiB | 36.0 MiB | no — `bm25` arm only |
| `article_stats_index` | 4.8 MiB | 68.1 MiB | **yes** |
| `fresh_pool_log` | 7.5 MiB | 26.5 MiB | offline replay only (see below) |
| `behaviors_split` | 17.0 MiB | 35.4 MiB | offline only |
| `lightgbm_booster` | 2.7 MiB rss | 3.1 MiB rss | **yes** |

**The Q4 question "measure your ANN index" has an interesting answer: the shipped
path has no ANN index.** Replacing similarity retrieval with a recency window
(§2E) deleted the largest serving structure — 95.6 MiB on MIND — while taking
recall from 2.90% to 93.9%. The serving footprint is now the fresh-pool window
plus the article-statistics index.

### Q4.4 Scaling to 10x — what breaks first

Two measured curves, rather than assertion.

**Latency is linear in K**, and features are the linear term:

| K | EB-NeRD total | MIND total |
|---|---|---|
| 50 | 2.44 ms | 2.35 ms |
| 100 | 3.21 ms | 3.41 ms |
| 200 | 5.08 ms | 5.92 ms |

So K is the latency knob: dropping to K=100 cuts request time ~37% and, per
§2E's recall figures, costs little recall.

**The article index grows with event volume**, which is the axis traffic moves
along. MIND: 40.7 MiB over train, 68.1 MiB over all splits — at 8 bytes per
event that is 8.9M events resident.

Ranked by what gives out first:

1. **`article_stats_index` memory — the binding constraint.** It must be
   resident, and it grows with total events. 10x MIND-small traffic is ~680 MiB;
   MINDlarge (~15M impressions against MIND-small's 230k, ~65x) is ~4.4 GB. The
   module docstring already flagged that it "does not survive MINDlarge", and
   this measurement puts a number on it. First thing to shard or to move to a
   time-windowed counter store.
2. **CPU for feature building.** 165–191 QPS per core, so 10x traffic is ~10
   cores. Embarrassingly parallel across requests, so this is a cost line, not a
   wall.
3. **Nothing else.** The fresh-pool window is bounded by *one hour of traffic*,
   not by catalogue size, so it grows with QPS and not with corpus age — a good
   property that came free with §2E. The booster is ~3 MiB and scoring is 7% of
   the request.

Note `fresh_pool_log` (439 MiB RSS on MIND) is an artifact of offline replay: it
holds every split's log so the benchmark can seek through time. A live system
holds only the 1-hour window. It is not a serving cost and must not be reported
as one.

### Caveats

- **The timed model is the `bm25` arm, not the shipped `popular` one**, which was
  trained on the other machine. Tree count drives scoring latency (EB-NeRD 17
  trees, MIND 80), but scoring is only ~7% of the request, so even a 5x larger
  model moves p99 by ~2 ms and no conclusion here turns on it. Re-run where the
  shipped model lives to make the number exact; each JSON records which file it
  timed.
- Single machine, CPU only, single-threaded. p99 from 300 requests is noisy at
  the ~1 ms level (two EB-NeRD runs gave 12.1 and 9.2 ms); the SLA verdict has a
  10x margin, so the noise does not threaten it.
- NRMS is not in these numbers. It is a different serving profile (GPU, ~774 MiB
  checkpoint) and is not part of the two-stage pipeline.

---

## 2H. Q5 — extended evaluation (2026-09-15)

`scripts/extended_eval.py` → `results/extended_eval_<dataset>_<method>.json`.
A1's `beyond_accuracy.py`, `slicing.py` and `bootstrap.py` are now ported into
`src/` (the §1.4 decision to run one harness on A2's metrics API), so every
figure reuses `src/metrics.py` and cannot drift from `report_<method>_<split>.json`.

Run on the shipped **`popular`** arm for both datasets. The first version of
this section reported the `bm25` arm, which was the only model on the machine it
was written on; those numbers are kept below as a contrast because the gap
between them is itself the §2E result.

### Results, test split, top-10 for beyond-accuracy

| | EB-NeRD/popular | MIND/popular | EB-NeRD/bm25 | MIND/bm25 |
|---|---|---|---|---|
| AUC | 0.9213 [0.9199, 0.9227] | 0.8799 [0.8786, 0.8813] | 0.9787 | 0.9698 |
| MRR | 0.2102 [0.2070, 0.2136] | 0.2239 [0.2219, 0.2260] | 0.0194 | 0.0189 |
| nDCG@5 | **0.2007** [0.1969, 0.2049] | **0.2450** [0.2427, 0.2475] | 0.0202 | 0.0199 |
| nDCG@10 | 0.2691 [0.2655, 0.2728] | 0.2980 [0.2958, 0.3004] | 0.0204 | 0.0206 |
| diversity | 0.0374 [0.0363, 0.0386] | 0.9257 [0.9246, 0.9268] | 0.0372 | 0.8566 |
| novelty | 14.46 [14.456, 14.457] | 17.29 [17.269, 17.311] | 13.05 | 13.57 |
| coverage | 0.0679 [0.0672, 0.0684] | 0.0032 [0.0031, 0.0033] | 0.1736 | 0.0639 |
| impressions scored / total | **24,591 / 25,356** | **68,682 / 73,152** | 647 / 25,356 | 2,120 / 73,152 |

Accuracy reconciles exactly with `report_popular_test.json`, which is the point
of reusing `src/metrics.py` on both paths. Averaged over the **full** population:
only positive-bearing impressions are scored, which is exact, not a shortcut
(§2E, tested). AUC is the inflated catalogue-negatives figure §2B explains, not a
like-for-like number against NRMS — and note it *falls* from bm25's 0.98 to 0.92
while nDCG rises 10x, which is that section's warning in one line.

**The 10x nDCG gap is recall, not ranking.** `bm25` retrieves the clicked article
for 2.6% of EB-NeRD impressions; `popular` does for 97%. A re-ranker cannot rank
what stage 1 never handed it, so the earlier figures measured the discarded arm's
recall ceiling. Anything quoted from this section must name the arm.

### Slices (nDCG@5)

| | cold-start | warm | head | tail |
|---|---|---|---|---|
| EB-NeRD | 0.2699 (n=65 ⚠) | 0.2005 (n=24,526) | **0.0000** (n=117) | 0.2023 (n=24,474) |
| MIND | **0.2580** (n=12,171) | **0.2422** (n=56,511) | **0.1605** (n=8,370) | 0.2571 (n=60,312) |

Slice denominators are exact, taken from the whole population rather than scaled
from the scored share; both slices are computable without the model.

**Correction — cold-start is not worse, it is slightly better.** The `bm25`
version of this section reported MIND cold-start users scoring ~46% below warm
(0.0117 vs 0.0217) and read it as the expected weakness of a history-driven
re-ranker. On the shipped arm the sign reverses: **cold-start 0.2580 vs warm
0.2422**, ~6.5% better, on 12,171 scored impressions. EB-NeRD points the same way
(0.2699 vs 0.2005), though its cold-start population is only 67 impressions and
is marked ⚠. The explanation is that the shipped pipeline is not history-driven:
stage 1 is a one-hour popularity window and the model leans on freshness and
popularity, so a user with no history loses little. The earlier reading was an
artefact of the arm, and it would have gone into the report as a finding.

**EB-NeRD's re-ranker is at chance on head articles.** The head slice scores
nDCG@5 and nDCG@10 of exactly 0.0000, AUC 0.5271 and MRR 0.0072 — the clicked
article sits around rank 139 of 200. That is not a bug: the EB-NeRD model puts
80–87% of its importance on `freshness_log_hours` (§2B/§2D), and "head" means
most-clicked *in train*, i.e. old. The model systematically buries exactly those
articles. MIND shows the same direction with a working model (head 0.1605 vs tail
0.2571, AUC 0.8334 vs 0.8864).

Two supporting facts worth reporting beside it: head-clicked impressions are only
197 of 25,356 on EB-NeRD (0.8%) against 9,150 of 73,152 on MIND (12.5%) — EB-NeRD's
test clicks land almost entirely on articles that were not popular in train, which
is the news cycle turning over — and stage 1 retrieves the positive for 117/197 of
them, so the failure is the ranker's, not retrieval's.

### Diversity is not comparable across datasets — and the raw numbers invert

Raw diversity reads EB-NeRD 0.037 against MIND 0.926, which looks like EB-NeRD
recommending near-identical articles. It is an artefact of the embedding space.
Measured over random article pairs:

| | random-pair diversity | recommended (popular) | vs random | (bm25) | vs random |
|---|---|---|---|---|---|
| EB-NeRD (provided multilingual BERT, 768-d) | 0.0484 | 0.0374 | **0.774x** | 0.0372 | 0.769x |
| MIND (MiniLM, 384-d) | 0.9402 | 0.9257 | **0.985x** | 0.8566 | 0.911x |

Both random-pair baselines reproduced to four decimals on a second machine
(0.0484 / 0.9402), which is the check that the two arms are being measured
against identical embedding spaces: EB-NeRD's are the provided RecSys24 vectors
and MIND's are recomputed by `src/embeddings_index.py`, so agreement was not
guaranteed.

Two *randomly chosen* EB-NeRD articles already have cosine 0.951: raw BERT
vectors are anisotropic and occupy a narrow cone, while MiniLM is contrastively
trained and spreads its space. So the 23x raw gap is the embedding model, and
against its own yardstick **EB-NeRD's recommender concentrates more than MIND's**
— the opposite of the raw reading. The JSON carries `diversity_random_baseline`
and `diversity_vs_random`; report the ratio, never the raw value across datasets.

On the shipped arm MIND sits at **0.985x random**: its one-hour fresh pool spans
enough topics that a top-10 list is barely less diverse than ten articles drawn
at random from the whole catalogue. Note this runs *opposite* to coverage, which
collapsed to 0.32% — the pipeline recommends from a very small set of articles
that happens to be topically wide. The two metrics are not substitutes, and a
report that quotes only one of them will mislead in whichever direction it
picked.

### Three bugs the first run surfaced

Worth recording, because all three would have shipped silently:

1. **Novelty was `inf`.** `popularity.parquet` lists every article ever
   *displayed* in train (display_count was added for the CTR feature), so most
   rows have click_count 0 and `-log2(0)` poisoned the mean and its whole CI.
   Zero-count articles now fall through to the unseen-article prior.
2. **The coverage CI excluded its own point estimate** — `[0.1535, 0.1588]`
   beside a value of 0.1736. A bootstrap resample repeats ~1/e of the
   impressions and so unions fewer distinct articles; the bias is inherent to
   union statistics. A first-order bias correction re-centres the interval, with
   the raw resample mean and the shift both reported so it stays auditable.
3. **Slice denominators were estimated** by scaling the scored count by the
   overall scored share — which assumes retrieval succeeds equally often in
   every slice, exactly what a cold-start slice exists to test. Now exact.

### Caveats

- Beyond-accuracy is computed on a 2,000-impression sample, and **coverage is
  conditional on that sample size** — scoring more impressions can only cover
  more catalogue. The count is reported beside every such figure; do not compare
  coverage across runs with different `--beyond-sample`.
- **Novelty rose on `popular` rather than falling** (13.05 → 14.46 EB-NeRD,
  13.57 → 17.29 MIND), against the prediction written here for the `bm25` run.
  The reason is definitional: novelty is `-log2(train click probability)`, and
  fresh-pool articles are recent, so most have no train clicks at all and score
  as maximally novel by construction. On this pipeline the metric is measuring
  recency, not editorial novelty. Report it with that caveat or not at all.
- **Coverage is the real cost of the recall fix.** EB-NeRD 0.1736 → 0.0679 and
  MIND 0.0639 → **0.0032**: the shipped pipeline recommends from a one-hour
  window, so it touches 0.32% of MIND's catalogue. That is the honest trade for
  taking recall@200 from 2.90% to 93.9% (§2E), and it belongs in the report as a
  trade rather than being quietly dropped.
- Accuracy is scored a chunk of impressions at a time
  (`--chunk-impressions`, default 5,000, mirroring `reranker.evaluate_over_population`).
  MIND's 73k impressions x 200 candidates is ~14.6M feature rows, which OOM'd this
  laptop as a single frame; chunked it is exact and peaks around one chunk.
- **Codabench submission is still outstanding** — Q5's fourth bullet.

---

## 2I. Q5/Q7.3 — Codabench submissions (2026-09-15)

`run_submit.py` → `data/submissions/<dataset>_nrms_freshness.zip` (gitignored,
which Q8 requires). Both files generated, format-verified, and ready to upload.

| | inner file | lines | size | wall time |
|---|---|---|---|---|
| MIND (`MINDlarge_test`) | `prediction.txt` | 2,370,727 | 108.0 MB | 2.9 min |
| EB-NeRD (`ebnerd_testset`) | `predictions.txt` | 13,536,710 | 229.8 MB | 4.2 min |

Line counts match the official impression counts exactly. Both boards were open
at the time of writing with **10 submissions/day** (checked against the Codabench
API): MIND's "Official Test" phase and EB-NeRD's "official testset" phase are
current with no end date. A1's "one upload per day" note referred to the closed
RecSys '24 phase.

### Why NRMS scores these and the re-ranker does not

The boards score a ranking of each impression's **given in-view list**. The
two-stage pipeline ranks a fresh pool it retrieves itself, and its click-driven
features are empty on an unlabelled test set, so it has nothing to say about a
list it did not choose. NRMS scores supplied candidates natively — that is the
task it was trained on — so the freshness arm (§2F) produces both files.

### The per-impression scorer could not have done this

`evaluate.py` re-encodes every candidate and every history slot of every
impression: measured at 282 (EB-NeRD) and 111 (MIND) impressions/s, the two test
sets would have taken **~13.5 h and ~6 h**. `src/nrms/serve.py` encodes each
catalogue once and gathers article vectors by row index, which is *exactly* the
same computation — `NewsEncoder` has no context, no user and no position — and
runs at 70,011 and 16,339 impressions/s, i.e. **4.2 and 2.9 minutes**.

That claim is the only thing standing between a valid submission and 16M rows of
quietly wrong ranking, so it is checked numerically rather than argued.
`scripts/verify_fast_scorer.py` re-scores our own labelled test splits through
the fast path and requires the committed metrics back:

| | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| deviation from `metrics_test.json`, both datasets | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

This is also the honest answer to Q4's serving question for NRMS: serving it
means a precomputed article-vector table plus per-request user encoding, not
re-encoding the catalogue per request.

### Reused from A1 rather than rewritten

`src/submission.py` is a port of A1's `src/newsrec/predict.py`, which produced
files the MIND leaderboard scored (bm25 0.5934, minilm 0.6425). Kept as-is: the
streaming chunk iterators, the vectorised id→row resolution, `SubmissionWriter`
with per-chunk resume, and `verify_submission`. **The formats came from the
competition APIs in A1, not from guesswork** — different inner filenames per
board, and integer ranks `1..n` with rank 1 highest rather than scores.

Three things changed for A2:

- `Chunk` carries each impression's own time, which A1's similarity scorer never
  needed and the freshness arm cannot work without.
- Catalogue text is concatenated per `TEXT_COLUMNS` (EB-NeRD title+subtitle+body,
  MIND title+abstract) and tokenised with the same `padding="max_length"` and
  `title_size` as training, so the news encoder sees text in the shape it learnt.
- `n_history=0` now means "no history" instead of "all history". `[-0:]` slices
  the *whole* list in Python, which crashed the first-seen pre-pass; the case is
  branched on explicitly in both places.

### Freshness references on the test sets

Same definitions as `article_stats.freshness_reference_times`, sourced from the
period being scored: EB-NeRD's real `published_time` (**125,541/125,541 articles,
100%**), MIND's earliest sighting from one extra streaming pass over the test log
(**30,043/120,961, 24.8%** — the rest never appear as candidates, so they are
never scored). `FreshnessLookup.ages` still gates on the reference strictly
preceding the impression, so a later sighting cannot date an earlier row.

MIND's early impressions therefore read `known=0` while the window is cold. That
is inherent to the proxy and nothing rides on it: MIND's freshness arm measured
null offline (§2H).

### Verification before upload

- Line counts equal the official impression counts; `verify_submission`
  stream-checks that every sampled line's ranks are a permutation of `1..n`.
- Output was diffed against the source files directly — impression ids in file
  order, and each line's rank count equal to that impression's candidate count
  (50,000 MIND lines against `behaviors.tsv`, 5,000 EB-NeRD against
  `behaviors.parquet`).
- `tests/test_submission.py` pins the format rules and the fast scorer's padding
  row.

### Outstanding

- **Upload both files and capture the leaderboard screenshots** (Q5 bullet 4,
  Q7.3). Nothing else in this section needs a rerun.
- A1's `mind_minilm.zip` remains on disk as an alternative. Its offline MIND AUC
  (0.6296) looked higher than NRMS's (0.6040), though on A1's own split, so the
  two are not strictly comparable. Leaderboard rank is not graded; submitting A2's
  own pipeline is the point.

---

## 2J. Q9 — metrics with and without features unavailable at serving time (2026-09-19)

`scripts/serving_features_ablation.py` → `results/serving_features_ablation_<dataset>.json`.
Q9 asks for metrics with and without any feature a live system would not have. All
22 re-ranker features were audited; three were in question.

### Which features a live system would not have

| feature | verdict | why |
|---|---|---|
| `impression_size` | **unavailable** | `len()` of the impression's own in-view list. On EB-NeRD `article_ids_inview` is the articles that came *into view*, which grows with how long the user stays on the page: **Spearman 0.50 with that impression's own `read_time`** (EB-NeRD demo train, 24,724 impressions), an outcome known only afterwards. On MIND it is the platform's list length for the impression being ranked, which a two-stage system choosing its own 200 candidates would not have either. |
| `has_known_publish_time` | **unavailable** | Read off the dataset's `published_time` snapshot, taken after the logs; it flags exactly the articles bulk re-stamped on 2023-06-29 (§2E). |
| `session_clicks_before` | borderline | Clicks on earlier impressions in the same session. A live session store holds most of them, but unlike the article statistics this feature applies no click-reporting lag. Dropped only in a stricter third arm. |

The other 19 are available: history-derived, stage-1-derived, or as-of article
statistics that already apply the 10-minute click lag.

### Design

Three arms — `all` (22 features), `serving_safe` (drops the two unavailable
features), `serving_safe_strict` (also drops `session_clicks_before`) — trained
exactly as `reranker.run` trains the shipped model: same train impressions (MIND
60k cap), same 49 random negatives, same seed. Only the feature list differs. The
`all` arm therefore has to reproduce `report_popular_test.json`, and does (max
deviation 1e-14 EB-NeRD, 6e-14 MIND), which validates the training path. Every
test impression is scored by all arms in one chunked pass; deltas are **paired**
over impressions (2,000 resamples, seed 0), with MRR/nDCG averaged over the full
population exactly as the reports are.

### Results — test, `popular` candidates

| | EB-NeRD MRR | EB-NeRD nDCG@5 | EB-NeRD nDCG@10 | MIND MRR | MIND nDCG@10 | MIND AUC |
|---|---|---|---|---|---|---|
| stage 1 (before) | 0.1639 | 0.1471 | 0.1994 | 0.2237 | 0.2932 | 0.8655 |
| `all` (as reported) | 0.2102 | 0.2007 | 0.2691 | 0.2239 | 0.2980 | 0.8799 |
| **`serving_safe`** | **0.1915** | **0.1773** | **0.2407** | 0.2234 | 0.2951 | 0.8756 |
| `serving_safe_strict` | 0.1913 | 0.1777 | 0.2401 | 0.2236 | 0.2954 | 0.8756 |

| paired Δ, 95% CI | EB-NeRD MRR | EB-NeRD nDCG@10 | MIND MRR | MIND nDCG@10 |
|---|---|---|---|---|
| `serving_safe` − `all` | **−0.0187** [−0.0210, −0.0163] | −0.0284 [−0.0310, −0.0258] | −0.0005 [−0.0018, +0.0007] | −0.0029 [−0.0041, −0.0017] |
| `all` − stage 1 | +0.0463 [+0.0430, +0.0495] | +0.0697 [+0.0660, +0.0732] | +0.0003 [−0.0012, +0.0017] | +0.0048 [+0.0034, +0.0061] |
| **`serving_safe` − stage 1** | **+0.0275** [+0.0248, +0.0305] | **+0.0413** [+0.0380, +0.0446] | −0.0003 [−0.0014, +0.0008] | +0.0018 [+0.0009, +0.0027] |

Gain share of the dropped features in the `all` model: `impression_size` 4.9%
(EB-NeRD) / 12.4% (MIND), `session_clicks_before` 1.1% / 0.0%,
`has_known_publish_time` **0.0%** on both.

### What it changes

- **The EB-NeRD headline was partly `impression_size`.** Without it the re-ranker
  still beats stage 1 by a clear margin — MRR +0.0275 [+0.0248, +0.0305], **+17%
  rather than +28%** — but ~40% of the reported MRR gain leaned on a feature that
  partly encodes the user's own engagement with the impression being scored. Its
  4.9% gain share understates that because gain measures split usage, not the
  ranking change it buys at the top. **Report both numbers**; the serving-safe one
  is what a deployed system would get.
- **MIND is unchanged in substance.** There was no MRR gain to lose (§2E); the
  12.4% gain share shows up only as −0.0043 AUC and −0.0029 nDCG@10.
- **The §2E future-article shortcut is confirmed gone.** `has_known_publish_time`
  carries exactly zero gain for fresh-pool candidates on both datasets — the
  measurement §2E's 0.003%-of-rows argument predicted.
- **`session_clicks_before` is harmless**: the strict arm is indistinguishable from
  `serving_safe` on every metric.
- **This also supplies §2E's missing paired CIs** (§3), for both the reported and
  the serving-safe re-ranker.
- **NRMS is unaffected.** It uses none of these features. It ranks the given
  in-view list, which is exactly the task both leaderboards score, so that list is
  its input by construction rather than a leak.

## 2K. Q1.1 — history-content features in the re-ranker (2026-09-19)

**Implemented; results pending** the rerun (`scripts/rerun_history_content.sh`,
run on a separate machine). Numbers and the dated notes in §2E/§2G/§2H/§2J will
be filled in from its output.

### Why

Q1.1 asks for click-history features over the user's recent clicked articles'
"titles, categories, **embeddings**". Titles and embeddings already drive the
`bm25`/`bm25_fresh` (history-title BM25) and `semantic` (pooled history
embeddings) stage-1 arms -- where they reach the re-ranker as `retrieval_score` --
and NRMS. But the **shipped `popular` re-ranker** saw history content only through
category: its `retrieval_score` is a click count. This closes that gap, and the
§2.6 decision's "bi-encoder similarity feature" with it.

### Design

`src/history_content.py`, two features for every stage-1 method, built entirely
from A1 code (`config.HISTORY_CONTENT_FEATURES`, default on):

| feature | definition |
|---|---|
| `history_title_bm25` | A1's history-title query (`BM25_CANDIDATE_CONFIG`: titles of the last 10 clicks; MIND `tfidf_keywords`, EB-NeRD `recency_weighted`) scored against the candidate with A1's BM25 index over title+abstract |
| `history_embedding_cosine` | cosine between the candidate and A1's pooled user embedding (`SEMANTIC_CANDIDATE_CONFIG`: MIND mean, EB-NeRD max over the full history) |

NaN means no usable history, never "dissimilar". Embedding coverage is 100% of
both catalogues and of every `popular` test candidate. One scorer call per
impression; the scorer is cached per process so chunked evaluation builds the
BM25 index once. Switching it off reproduces the 22-feature models exactly: on
EB-NeRD val every original column is bit-identical with the switch on or off.

**Leakage (Q9):** only the impression's own `history` -- clicks strictly before
it -- and static catalogue text and vectors are read, so both features are
available at serving time and stay in §2J's serving-safe arms.
`scripts/serving_features_ablation.py` gains a `no_history_content` arm, which
gives their contribution as a paired CI on the same harness.

**Serving:** `scripts/serving_benchmark.py` now computes both per request inside
the timed feature-building stage. This changes a §2G finding: the shipped path
now holds the BM25 index and the embedding table, so "the shipped path has no ANN
index" no longer holds.

**Expectation, stated before the numbers:** on EB-NeRD val the raw means barely
separate clicked from non-clicked candidates (title BM25 42.8 vs 43.0, cosine
0.598 vs 0.598) -- consistent with §2E's finding that history similarity is a weak
signal within a fresh pool. A small gain, or none, would not be a surprise.

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
- [x] ~~Whether the Q3 "principled improvement" is the loss-function change or
      the bi-encoder feature.~~ **Resolved 2026-09-11** (§2F): neither — the Q3
      improvement is **freshness weighting on NRMS**, which improves the actual
      baseline rather than the re-ranker, so "reproduce then beat it" compares
      like with like.
- [x] ~~Run the §2F arms and decide on seed count.~~ **Done 2026-09-11** (§2F
      Results): EB-NeRD +0.069 AUC (CI [+0.066, +0.071]), MIND null. Extra seeds
      judged unnecessary — the EB-NeRD effect dwarfs seed noise and MIND's null is
      explained by age-homogeneous candidate sets, not variance.
- [x] ~~Generate the Codabench submission files.~~ **Done 2026-09-15** (§2I):
      both verified and waiting to be uploaded; screenshots still needed.
- [x] ~~Paired bootstrap 95% CIs for the §2E before/after deltas.~~ **Done
      2026-09-19** (§2J): EB-NeRD MRR +0.0463 [+0.0430, +0.0495], serving-safe
      +0.0275 [+0.0248, +0.0305]; MIND spans zero.
- [x] ~~Q9: metrics with and without serving-unavailable features.~~ **Done
      2026-09-19** (§2J).
- [ ] The Q1 feature-group ablation under the chosen random-negative sampling
      (§2E's candidate-level-only arm ran under the discarded hard negatives).
- [ ] A fair Q3 head-to-head with NRMS on identical candidate sets.
- [ ] Fill in §2K's results once `scripts/rerun_history_content.sh` has run.

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
- **2026-09-11** — Added §2F: freshness weighting implemented as the Q3
  improvement, behind `--freshness`. Chose it over the proposed positional
  encoding after measuring that 29.4% of MIND's history slots are padding, which
  makes PE unsound there without masking. Records the three properties that keep
  the ablation valid (zero-init head, conditional construction to preserve the
  baseline's RNG stream, ~65 parameters) and the as-of gate that keeps the age
  signal leak-free. Tested, not yet trained. Q3-improvement decision closed.
- **2026-09-11** — §2F "Corrections after review": fixed two verified bugs in the
  freshness arm. (1) MIND dated only 25% of test candidates (17% of clicks)
  because the reference used train-only sightings; the splits tuple moved to
  `config.ARTICLE_STATS_SPLITS`, shared by the re-ranker and `FreshnessLookup`,
  and the vacuous equivalence test now checks the splits actually in use.
  (2) The arms trained on different data orders because the head's init consumed
  the global RNG; it is now built under `fork_rng`. Head parameter count
  corrected to 65.
- **2026-09-15** — Added §2G (Q4 serving and scale): `scripts/serving_benchmark.py`
  plus `results/serving_<dataset>.json`. p99 9.2 ms (EB-NeRD) / 9.6 ms (MIND)
  against a 100 ms SLA; feature building is ~85% of the request and the model ~7%.
  The shipped path carries no ANN index at all, and `article_stats_index` memory
  is what breaks first at 10x.
- **2026-09-15** — Added §2H (Q5 extended evaluation): ported A1's
  `beyond_accuracy`/`slicing`/`bootstrap` into `src/`, added
  `scripts/extended_eval.py` and `results/extended_eval_<dataset>_<method>.json`.
  All seven metrics with bootstrap CIs plus both slices. MIND cold-start users
  score 46% below warm. Fixed three bugs the first run exposed: novelty `inf`
  from zero-click articles, a coverage CI that excluded its own point estimate,
  and estimated rather than exact slice denominators. Also established that raw
  diversity is not comparable across datasets (embedding anisotropy) and added a
  random-pair baseline.
- **2026-09-11** — §2F "Results": both freshness arms trained and evaluated.
  EB-NeRD +0.0686 AUC / +0.0567 MRR / +0.0525 nDCG@10 over the §2C baseline
  (paired-bootstrap 95% CIs all clear of zero); MIND null on every metric. Traced
  MIND's null to age-homogeneous in-view sets (learned head alone ranks at AUC
  0.514 on MIND vs 0.677 on EB-NeRD; within-impression age spread 27.5 h vs
  1,057 h), **not** to the corrected coverage bug — 99.9% of MIND test candidates
  are dated. Recorded the caveat that EB-NeRD's age-only ranker (0.677) beats the
  full freshness NRMS (0.611). Seed-count decision closed.
- **2026-09-15** — §2F: scoped the MIND conclusion to "freshness-**as-first-seen**
  doesn't help on MIND" — the proxy is right-censored, so the null cannot be
  claimed for freshness in general. Added the EB-NeRD transfer test that prices
  the proxy where both definitions exist (one fitted curve, real vs proxy ages:
  AUC 0.6765 → 0.6631, within-impression spread 1,057 h → 52.7 h), which makes
  age-homogeneous in-view sets the better-supported explanation of the null
  without settling it. Also recorded that EB-NeRD's signal is an age *band*
  (~6 h), not recency: newest-first ranks at AUC 0.4950, chance. Moved the
  analysis into `scripts/paired_bootstrap.py` and `scripts/age_signal.py` so
  every figure in §2F Results is reproducible.
- **2026-09-15** — §2H re-run on the shipped `popular` arm for both datasets; the
  first version reported `bm25`, the discarded stage-1 arm, whose 2.6%/2.9%
  retrieval recall put nDCG@5 10x low (0.0202 → 0.2007 EB-NeRD, 0.0199 → 0.2450
  MIND) and left EB-NeRD's slices at n=2 and n=9. Two findings changed with it:
  cold-start users are ~6.5% **better** than warm, not 46% worse (the shipped
  pipeline is freshness/popularity-driven, not history-driven), and EB-NeRD's
  re-ranker is at chance on head articles (nDCG 0.0000, AUC 0.5271) because it
  is essentially a freshness model and "head" means popular in train, i.e. old.
  Coverage fell to 0.32% of MIND's catalogue — the trade for the §2E recall fix.
  `extended_eval.py` now chunks the accuracy pass (MIND OOM'd at ~14.6M feature
  rows) and degrades diversity gracefully where the semantic feature store is
  absent.
- **2026-09-15** — §2H diversity filled in for the shipped arm: the EB-NeRD
  provided-embedding artifact and MIND's MiniLM feature store were both built
  locally, so all seven Q5 metrics now come from `popular` on one machine.
  EB-NeRD 0.0374 (0.774x random), MIND 0.9257 (**0.985x** random). Both
  random-pair baselines reproduced the other machine's to four decimals.
  MIND's near-random diversity sits opposite its 0.32% coverage — a small
  article set that is topically wide — so the two must be reported together.
- **2026-09-15** — Added §2I (Q5/Q7.3 Codabench submissions): both files built and
  verified — MIND 2,370,727 lines, EB-NeRD 13,536,710, matching the official
  counts. Ported A1's leaderboard-validated submission plumbing into
  `src/submission.py` and added `src/nrms/serve.py`, a cached-article-vector
  scorer that is exactly equivalent to the per-impression path (zero deviation on
  all four metrics for both arms, `scripts/verify_fast_scorer.py`) and turns
  ~19.5 h of scoring into 7 min. Confirmed via the Codabench API that both boards
  are open at 10 submissions/day. Upload and screenshots remain.
- **2026-09-19** — Added §2J (Q9): re-ranker metrics with and without features a
  live system would not have, via `scripts/serving_features_ablation.py`.
  `impression_size` is outcome-contaminated on EB-NeRD (Spearman 0.50 with the
  impression's own read time); dropping it with `has_known_publish_time` takes
  EB-NeRD MRR 0.2102 → 0.1915 (−0.0187 [−0.0210, −0.0163]), still +0.0275
  [+0.0248, +0.0305] over stage 1 (+17% rather than +28%). MIND MRR unchanged.
  `has_known_publish_time` carries 0% gain on both, confirming §2E. Also supplies
  §2E's missing paired CIs. Flagged the +28% headline in "Current state" and §2E.
- **2026-09-19** — Added §2K: history-title (BM25) and history-embedding (cosine)
  similarity features for the re-ranker, default on, built from A1's query
  construction, BM25 index and pooled user embeddings (`src/history_content.py`).
  Closes the Q1.1 titles/embeddings gap for the shipped `popular` arm. Serving
  benchmark and the Q9 ablation (new `no_history_content` arm) updated. Results
  pending the rerun on a separate machine.
