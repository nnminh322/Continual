# Continual Learning with Statistical Routing & Information-Geometric LoRA

> Dự án nghiên cứu: tối đa single-task performance trên frozen-backbone LoRA, zero-rehearsal.

---

## Abstract

### Contribution 1 — Statistical Routing Theory (SRT)
**Vấn đề:** Backbone frozen → mỗi task tạo một phân phối $\mathcal{P}_t$ trên embedding space. Router cần xác định task ID từ input mà không có training data cũ.

**Phát hiện cốt lõi:**
- Embedding space là anisotropic cone, không phải isotropic sphere
- Task centroid separation + covariance shape quyết định routing difficulty
- Whitening (ZCA) biến anisotropic → isotropic, đơn giản hóa Mahalanobis → L2
- LLaMA routing ceiling ~95% vì PaR ≈ 9 (effective dims rất thấp)

**Giải pháp:** SRT framework — chọn metric tối ưu (L2 / Mahalanobis / PPCA) theo geometry của task signatures $\{\mu_t, \Sigma_t\}$.

**Tại sao routing accuracy quan trọng?** Router sai → sai adapter → catastrophic forgetting. GainLoRA baseline dùng learned MLP router → drift. SRT non-parametric → zero drift, zero forgetting on router.

---

### Contribution 2 — Information-Geometric LoRA (IGL)
**Vấn đề:** FGCL/GALA fail trên frozen backbone vì tất cả regularization đều thuộc paradigm "isolate" — nhưng backbone frozen đã cô lập sẵn. Isolation thêm không giúp gì.

**Root cause:** Frozen backbone tạo information bottleneck. LoRA rank $r \ll d$. Mỗi task chỉ encode được $r$ directions trong không gian $d$. Khi $T$ tăng → rank exhaustion.

**Giải pháp:** Thay vì isolate, ta **tối ưu hóa information gain** trong constraint. Objective:

$$\max_{\Delta W_t} \; I(Y_t; h + \Delta W_t h) \quad \text{s.t.} \quad I(Y_t; h + \Delta W_s h) \leq \epsilon, \; \forall s < t$$

Key innovation: gradient direction được compute từ **stored covariances** $\{\Sigma_s\}$, không phải từ gradients trên frozen features (những gradients đó quá nhỏ để tạo signal).

**Hệ quả:**
- FSR/GPM fail vì Fisher computed trên static space
- KF-FNG fail vì diagonal approximation quá coarse
- IGL succeed vì dùng **full covariance** trong generalized eigenproblem

---

### Integration

```
Task t arrives:
  C1 (SRT): Route via d_PSR → identify task
  C2 (IGL): Update LoRA via information-theoretic gradient using {Σ_s}_{s≤t}
  Store: {μ_t, Σ_t} + B_t
  Inference: C1 routes → C2's B_t fires
```

---

## Settings

- **Zero-rehearsal**: Không lưu raw samples. Statistical signatures $\{\mu_t, \Sigma_t\}$ được phép (cùng loại GPM).
- **Task-agnostic inference**: Không có task ID tại inference.
- **Frozen backbone**: Chỉ LoRA adapters được train.

---

## File Structure

```
routing_analysis/
├── contribution_UNIFIED.md     # Toàn bộ lý thuyết (C1 + C2 + proofs)
├── exp_fgcl.py                 # Experiment runner cho FGCL methods
├── analyze_geometry.py          # Phase A: embedding geometry EDA
├── compare_routing.py           # Phase B+C: routing metric/algorithm comparison
├── ablation_psr.py             # Phase D: PSR component ablation
├── validate_theory.py          # Phase E: theory validation
├── simulate_gpm_routing.py     # Phase F: GPM vs RLS learned routing simulation
├── extract_embeddings_*.py     # Embedding extraction scripts
├── CL_Benchmark/               # Dữ liệu benchmark
├── embeddings/                 # Pre-extracted embeddings (output)
└── results/                    # Kết quả experiments
```

---

## Running

### C1 (SRT) — Routing Analysis
```bash
# Phase A: Embedding geometry
python analyze_geometry.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence

# Phase B+C: Routing comparison
python compare_routing.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence

# Phase D: PSR ablation
python ablation_psr.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence

# Phase E: Theory validation
python validate_theory.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence

# Phase F: Learned routing simulation
python simulate_gpm_routing.py --emb_dir embeddings/T5EncoderModel --benchmark Long_Sequence
```

### C2 (IGL) — Continual LoRA Training
```bash
# Full experiment (all phases × all methods × all benchmarks)
python exp_fgcl.py

# Just T4 (full comparison), one benchmark
python exp_fgcl.py --phase T4 --benchmark Long_Sequence

# One method, eval every task (for analysis)
python exp_fgcl.py --phase T4 --benchmark Long_Sequence --eval_every 1
```

---

## Experiment Plan for Contribution 2

### Priority 1 — Theory Validation

**E-C2-1: IGL vs Standard LoRA (single task)**

Protocol: Train 1 task với IGL vs standard AdamW. Measure convergence speed và final accuracy.

- Dataset: Long_Sequence (T5-XL)
- Metrics: final accuracy, training loss curve
- Baseline comparison: standard_lora vs IGL (IGL_Train from §9.3.2)

Expected: IGL ≈ standard (same task, no interference). Validate that IGL doesn't hurt single-task performance.

**E-C2-2: Information Gradient Magnitude**

Protocol: Compute $\|\nabla_{\text{info}}\|$ vs $\|\nabla_{\text{loss}}\|$ on frozen features.

- Measure: ratio $\frac{\|\text{tr}(A\Sigma_t A^\top) - \lambda\sum_s \text{tr}(A\Sigma_s A^\top)\|}{\|\nabla_{\text{CE}}\|}$
- Dataset: T5-XL, 6 tasks
- Expected: $\|\nabla_{\text{info}}\| \gg \|\nabla_{\text{loss}}\|$ — explains why loss-gradient methods fail

**E-C2-3: Generalized Eigenvalue Spectrum**

Protocol: Compute eigenvalues of $S_{\text{signal}} - \lambda S_{\text{interference}}$ per task.

- Dataset: T5-XL, Long_Sequence, tasks 1..15
- Metrics: top eigenvalue magnitude, eigenvalue gap, condition number
- Expected: $\lambda_{\max}$ decreases as tasks accumulate (shared signal decreases)

### Priority 2 — CL Comparison

**E-C2-4: IGL vs Standard LoRA vs InfLoRA (CL setting)**

Protocol: Full CL pipeline, 6-15 tasks, measure AP / FT / BWT.

| Method | Description |
|--------|-------------|
| standard_lora | Baseline: no CL mechanism |
| inflora | GPM null-space projection |
| **igl** | Information-theoretic gradient direction |
| **igl_full** | IGL + pooled shrinkage (Theorem 5 from C1) |

- Dataset: Long_Sequence, SuperNI
- Backbone: T5-large, T5-xl
- Metrics: AP, FT, BWT, AP per task progression
- **Key metric**: BWT improvement over standard_lora

**Expected**: IGL shows measurably better BWT than FSR/GPM methods.

**E-C2-5: λ Sensitivity Analysis**

Protocol: Sweep $\lambda \in \{0.0, 0.1, 0.5, 1.0\}$ for IGL.

- Dataset: Long_Sequence, 8 tasks
- Expected: U-shaped curve — too low $\lambda$ → interference; too high → no signal. Optimal around empirical estimate.

### Priority 3 — Capacity & Scaling

**E-C2-6: Rank Exhaustion Experiment**

Protocol: Increase number of tasks from 6 → 15 → 30 (pseudo-tasks from splitting).

- Measure: AP vs T curve for IGL vs standard_lora
- Expected: Gap widens as T increases (IGL handles interference better)

**E-C2-7: Per-Layer Analysis**

Protocol: Compute IGL update direction per layer, compare with GPM basis overlap.

- Measure: $\delta_l = \text{overlap}(v_{\text{IGL}}^{(l)}, \text{GPM\_basis}^{(l)})$
- Expected: IGL directions ≠ GPM null-space directions (explains why GPM doesn't help)

---

## Benchmark

| Benchmark | Tasks | Domain spread | Notes |
|-----------|-------|-------------|-------|
| Long_Sequence | 15 | Sentiment + NLI + QA | Same-domain pairs (yelp/amazon/imdb) |
| SuperNI | 15 | Diverse | Cross-domain tasks |

---

## Key Results (Expected)

| Experiment | Metric | Expected |
|------------|--------|----------|
| E-C2-1 | Single-task accuracy | IGL ≈ standard |
| E-C2-2 | Gradient ratio | $\|\nabla_{\text{info}}\| > \|\nabla_{\text{loss}}\|$ by 10-100× |
| E-C2-4 | BWT | IGL > FSR > GPM > standard |
| E-C2-5 | $\lambda$ sensitivity | Optimal around empirical estimate |
| E-C2-6 | AP vs T | IGL gap widens vs standard |
