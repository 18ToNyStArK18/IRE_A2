# Design Choices — A2 Re-Ranker (Q2)

Working design log for the Q2 re-ranker. Records what we chose, what we
rejected, and *why*, so the Q6 design note can be written from evidence rather
than memory. Decisions marked **PROPOSED** are not yet signed off.

Last updated: 2026-09-10

---

## 0. The constraint that drives every decision below

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
- `lightgbm` is **not yet in `requirements.txt`** — needs adding.

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

## 3. Open decisions

- [x] ~~Sign-off on §1.4 (LambdaMART primary) and §2.6 (no cross-encoder).~~
      **Done 2026-09-10.** Ablation is over the Q1 feature groups.
- [ ] Raising A1's retrieval recall is tracked in `todo.md`, deliberately
      deferred: the re-ranker consumes the candidate contract regardless of
      candidate quality, so better retrieval changes the numbers, not the code.
- [ ] **MIND train scale.** 27.7M rows before filtering. Group pre-filtering
      (§1.5) cuts it to ~1.2M, which is comfortable — confirm that is the chosen
      route rather than subsampling impressions.
- [ ] Which candidate source feeds the re-ranker: BM25, semantic, or both as
      separate runs. Both are generated at K=200; only `ebnerd/bm25/val` exists
      on disk so far.
- [ ] Whether the Q3 "principled improvement" is the loss-function change
      (pointwise → LambdaMART) or the bi-encoder feature. Either works; they
      should not be conflated into one ablation arm.

## 4. Changelog

- **2026-09-10** — Created. Loss-function analysis (§1) and cross-encoder
  analysis (§2) written against measured group statistics and A1's recorded
  recall figures.
- **2026-09-10** — Added §0.1: measured the re-ranking headroom rather than
  assuming it. BM25 ranks its own retrieved positives at median 80/200 with none
  at rank 1, leaving 97% of the achievable MRR unclaimed. Confirms that
  single-positive groups are not the problem and that the re-ranker has a real,
  quantified opportunity.
