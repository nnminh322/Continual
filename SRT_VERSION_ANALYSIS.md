# SRT Version Analysis

This note compares the three SRT variants that matter for the recent LLaMA continual-routing investigation:

1. the lower-scoring runtime version seen in `new_llama_gainlora/hmm_log.txt`, which belongs to the `llama` line,
2. the `routing_class_v2` experimental offline proxy,
3. the near-perfect runtime version now carried by `new_debug` / `run_debug`.

The goal is not to list every edit, but to answer one specific question precisely: what actually changed, and why could that change improve routing?

## Short answer

The jump to near-perfect routing did **not** come from the low-VRAM changes in `run_debug`.

The main algorithmic jump came from the router patch in commit `ad60437`, which changed pooled covariance from an old **union-style pooled covariance** to a **pooled within-class covariance** and made runtime/offline router state explicit via `cov_dof` and `covariance_mode`.

By the time branch `llama` reached ref `048e693`, the runtime had already been aligned on the important prompt/extractor side:

- source-only routing input,
- `instruction.format(sentence)` prompt,
- `add_special_tokens=False`,
- frozen LLaMA extractor,
- last non-padding token pooling.

So the big remaining difference from `llama` to near-perfect `run_debug` was mostly **router math**, not prompt plumbing.

## Version map

| Variant | Concrete ref / artifact | What it is | Main distinguishing property |
| --- | --- | --- | --- |
| Low SRT | branch `llama`, ref `048e693`; empirical log `new_llama_gainlora/hmm_log.txt` | deployed runtime SRT before the within-class covariance patch | runtime path already aligned to source-only routing, but router still used old pooled covariance semantics |
| Experimental `routing_class_v2` | offline evaluator in `routing_analysis/routing_class_v2.py`; historical artifact `routing_analysis/ablation_truely.txt` | frozen-embedding proxy for continual routing | offline, not full generation runtime; historical artifact is still pre-within-class and therefore stale relative to current code |
| Near-perfect runtime | branch `new_debug`, head `9f023a2`; routing-critical patch is `ad60437` | deployed runtime after router-math fix | same runtime pipeline as `llama`, but covariance estimator changed to pooled within-class form and router state became explicit |

## 1. Low SRT: what the `hmm_log.txt` version actually is

### Identity

This is the deployed LLaMA continual runtime on the `llama` line. The branch pointer is currently `048e693`.

Important correction: the `70%` line in `hmm_log.txt` is **not routing accuracy**. It is only the evaluation progress bar.

The actual routing summary in that log is:

- task 1 overall: `100.0% (20/20)`
- task 2 overall: `98.6% (138/140)`
- task 3 overall: `97.2% (350/360)`
- task 4 overall: `97.8% (609/623)`

So the low version was already strong, but it was not fully saturated.

### What this version already had

By `048e693`, the runtime already had the important prompt/extraction alignment pieces:

- routing uses `input_ids_wo_label`, i.e. source-only prompt tokens,
- the source prompt is `instruction.format(sentence)`,
- tokenization uses `add_special_tokens=False`,
- the frozen extractor pools the final hidden state at the last non-padding token,
- the support-signature path and inference path are already using the same source-prompt profile.

In other words, this version is **not** the old obviously-misaligned runtime anymore.

### What this version still did wrong

Its router still used the older pooled-covariance update.

In that older form, the code path looked like this conceptually:

```text
welford_pooled_update(mu_old, cov_old, n_old, mu_new, cov_new, n_new)
-> returns pooled mean + pooled covariance + pooled sample count
-> shrinkage uses n_pool
```

The statistical issue is that this pooled covariance behaves like the covariance of the **union of all seen samples**, not the average **within-task** covariance.

That means the pooled covariance is contaminated by between-task scatter.

Conceptually:

```text
Sigma_union ~= Sigma_within + Sigma_between
```

For routing, that is the wrong object if the classifier is supposed to use shared within-task noise around task centroids.

### Why that matters

If task means are far apart, the old pooled covariance treats part of that separation as if it were ordinary shared variance.

That weakens Mahalanobis discrimination exactly in the directions that help tell tasks apart.

So this low-SRT runtime was already well-aligned on prompt/profile, but its shared covariance still blurred class separation a bit.

## 2. Experimental `routing_class_v2`: what it really is

### Identity

`routing_analysis/routing_class_v2.py` is an **offline continual-routing evaluator on frozen `.npz` embeddings**.

It is not the deployed runtime sitting inside `model.generate(...)`.

Its role is hypothesis validation and deployment proxy evaluation.

The old artifact in the repo, `routing_analysis/ablation_truely.txt`, is from the **pre-within-class** era. It has not been regenerated after the new router math.

### What it measures

This pipeline evaluates routing directly on saved frozen embeddings. That removes several runtime layers:

- no DeepSpeed/generation interaction,
- no decoder-time routing inside the full inference loop,
- no continual prediction loop bookkeeping,
- direct frozen-embedding classification.

It also advertises `PooledMahalanobis_RIDGE` as the deployment proxy for SuperNI.

### Why it can look a bit better than runtime

This version can look slightly cleaner than runtime for structural reasons:

- it operates directly on frozen embeddings,
- it avoids runtime generation plumbing,
- it often uses macro task-averaging, while runtime logs often show micro accuracy over the seen-task evaluation pool.

That last point matters a lot: the numbers are **not apples-to-apples** unless the metric definition and sample pool are matched exactly.

### What the historical experimental version still shared with low runtime

In the historical form represented by `ablation_truely.txt`, `routing_class_v2.py` still used the old pooled covariance semantics.

The old file shape was effectively:

```text
welford_pooled_update(mu_old_t, cov_old_t, n_old, mu_new_t, cov_new_t, n_new)
Sigma_shrunk = shrink_fn(Sigma_pool_t, n_pool, d)
```

There was no `cov_dof`, no `covariance_mode`, and no explicit separation between pooled mean and pooled within-class covariance degrees of freedom.

So the historical offline experiment was cleaner as an evaluation surface, but it still carried the same covariance-model weakness as the old runtime.

## 3. Near-perfect `run_debug`: what changed for real

### Identity

The near-perfect runtime is what you now associate with the `run_debug` / `new_debug` line.

However, the branch name is a little misleading.

The actual routing-critical patch is commit `ad60437`.

The later commit `692ed0a` on top of it only changes low-VRAM launcher behavior and does **not** touch the router files.

### What changed from `llama` to the near-perfect runtime

Between `048e693` and `ad60437`, the routing-related diff is concentrated in only three files:

- `new_llama_gainlora/src/srt_router_v2.py`
- `new_gainlora/src/srt_router.py`
- `routing_analysis/routing_class_v2.py`

That is strong evidence that the routing jump is coming from router math, not from unrelated runtime plumbing.

### The exact router change

The new version introduces:

- `COVARIANCE_MODE = "within_class"`
- `_cov_dof` / `cov_dof`
- a new `welford_pooled_update(..., dof_old, ..., n_new)` signature
- shrinkage using `max(cov_dof, 1)` instead of `n_pool`
- persisted router metadata: `cov_dof` and `covariance_mode`
- hard rejection of legacy `union_legacy` router artifacts on load

The new covariance target is:

```text
Sigma_within = sum_t ((n_t - 1) * Sigma_t) / sum_t (n_t - 1)
```

while the pooled mean is still averaged over all samples:

```text
mu_pool = sum_t (n_t * mu_t) / sum_t n_t
```

This is the statistically correct shared-covariance object for a pooled-covariance Mahalanobis classifier over task centroids.

### What did not change here

The near-perfect runtime did **not** need another big rewrite of:

- source-only prompt routing,
- `input_ids_wo_label`,
- frozen extractor pooling,
- `build_superni_source_prompt`,
- runtime routing summary.

Those were already mostly in place by the `llama` ref you asked about.

So if you ask what explains the jump specifically from the low branch to the near-perfect branch, the answer is: **the covariance estimator changed**.

## 4. Side-by-side difference table

| Dimension | Low SRT (`llama` / `hmm_log`) | Experimental `routing_class_v2` | Near-perfect (`run_debug`) |
| --- | --- | --- | --- |
| Evaluation surface | deployed runtime | offline frozen-embedding proxy | deployed runtime |
| Prompt for routing | source-only | frozen embeddings already extracted | source-only |
| Tokenization / pooling profile | already runtime-aligned | depends on embedding extraction metadata; historical artifact was old | runtime-aligned |
| Shared covariance target | old pooled union-style covariance | old pooled union-style covariance in historical artifact | pooled within-class covariance |
| Shrinkage sample count | `n_pool` | `n_pool` in historical artifact | `cov_dof = sum(n_t - 1)` |
| Router state metadata | legacy / weaker | not a runtime artifact | explicit `cov_dof` + `covariance_mode` |
| Direct comparability of numbers | runtime micro summary | often macro over frozen embeddings | runtime micro summary |

## 5. Why the new version can improve so much

### 5.1. It fixes the classifier's covariance model

This is the main reason.

For a pooled-covariance Mahalanobis router over task centroids, the correct shared covariance is the pooled **within-task** covariance, not the covariance of the union of all tasks.

If you use the union covariance, then directions that separate tasks inflate the shared covariance and become partially normalized away.

That directly hurts discrimination.

### 5.2. The inverse covariance becomes better calibrated

The shrinkage step in the new version uses `cov_dof` instead of `n_pool`.

That means ridge shrinkage is calibrated by the number of covariance degrees of freedom that actually estimate within-task noise.

This is cleaner than pretending every pooled sample equally contributes to within-class covariance estimation.

### 5.3. Runtime and offline now use the same statistical object

The new runtime router, legacy GainLoRA router, and offline evaluator were all patched together.

That removes a whole class of silent mismatches where the experiment and the deployment are no longer evaluating the same estimator.

### 5.4. The new router state refuses old semantics

The load path now rejects legacy `union_legacy` states.

That matters because otherwise a new runtime could accidentally load an old router artifact and still report numbers as if nothing changed.

## 6. What did **not** explain the improvement

### Not the `70%` line in `hmm_log.txt`

That line was only a progress bar.

### Not the low-VRAM patch

The low-VRAM commit on top of `ad60437` changes launcher/runtime defaults such as eval batch size, gradient checkpointing defaults, and the DeepSpeed config choice.

It does not modify the router files.

So it is not the reason routing became near-perfect.

### Not a looser summary metric

The runtime summary path is not more permissive now.

In fact, the current summary explicitly tracks `slot_correct` and `label_correct` separately. That is stricter and more informative than the older one-bit `✓` style.

## 7. Important caveat about `routing_class_v2`

The current source file `routing_analysis/routing_class_v2.py` has already been patched to the new within-class covariance semantics.

But the artifact `routing_analysis/ablation_truely.txt` is still an old run from before that patch.

So this pair is currently mismatched:

- current source: new math,
- checked-in text artifact: old math.

That is why `ablation_truely.txt` should be treated as a **historical baseline artifact**, not as the current offline truth.

## 8. Final conclusion

If you strip away branch names and focus only on what changed mathematically, the story is simple:

1. the lower runtime version was already mostly aligned on prompt/extractor behavior,
2. the historical `routing_class_v2` experiment was a cleaner offline proxy but still old in covariance semantics,
3. the near-perfect runtime became near-perfect mainly because the shared covariance changed from a union-style pooled covariance to a pooled within-class covariance.

That is the one change that directly makes Mahalanobis routing sharper rather than blurrier.

If you want a strict apples-to-apples follow-up, the next step is to regenerate the offline `routing_class_v2` results with the patched source and compare them to the current runtime on the exact same round and task pool.