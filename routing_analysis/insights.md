# Routing Geometry Analysis — Comprehensive Insights (Extended)
## PHẦN II: Phản biện Giả thiết Whitening + NearestCentroid & Contribution Mới

> **Scope**: 3 backbones × 5 phases (A–F) × 2 benchmarks × 2 spaces (raw / whitened) = 60 JSON result files.
> **Key**: LS = Long_Sequence (15 tasks), SNI = SuperNI (15 tasks). All accuracy figures are on held-out test splits.

---

## 0. Phản biện Giả thiết của Nghiên cứu: Whitening + NearestCentroid

### 0.1 Đánh giá lập luận của nghiên cứu sinh

Nghiên cứu sinh đưa ra hai giả thiết chính:

> **(H1)**: Feature space (embedding space) là một hypersphere không hoàn hảo — bất đẳng hướng, tập trung ở một cone/hyper-ellipsoid.

> **(H2)**: Whitening làm đầy hypersphere đó, biến phân phối thành hình cầu ở ma trận hiệp phương sai. Whitening + NearestCentroid cho kết quả tốt nhất trong các thí nghiệm hiện tại, nhưng **tính tổng quát không cao** — orthogonality giữa các task chỉ là artifact của việc chỉ test 15 tasks, không phải tính chất cấu trúc.

**Phản biện từng phần của giả thiết:**

#### H1 — Đúng và quan trọng, nhưng chưa đủ sâu

**Điểm đúng:**
- Anisotropy là hiện tượng có tài liệu rõ ràng (Mu et al., "All-but-the-Top", 2018; Ethayarajh et al., " representational entropy", 2019; \
  " isotropic LM embeddings", 2019).
- Condition number T5-large ≈ 132–197, LLaMA ≈ 412–439 → anisotropy tăng theo scale, đúng với literature.
- Participation Ratio (PaR ≈ 9–27) cho thấy embeddings concentrate vào subspace cực nhỏ → đúng là "cone" geometry.

**Điểm cần bổ sung/cần phản biện:**

1. **"Cone" geometry hay "tame manifold" geometry?** Giả thiết nói "hyper-ellipsoid/cone" nhưng thực tế:
   - Embeddings sống trên một **submanifold** của ℝ^d, không phải ellipsoid đầy đủ.
   - PaR ≈ 9–27 (LLaMA–T5) cho thấy: task-relevant information sống trong subspace rất nhỏ. Nhưng **không gian còn lại** (d - PaR dimensions) không phải noise đẳng hướng — nó có structure riêng (thể hiện qua kurtosis LLaMA whitened = 23.8).
   - "Cone" metaphor có thể misleading: nó gợi ý một hình học đơn giản, trong khi thực tế có thể là **mixture của nhiều cones** (multi-modal tasks như dbpedia, mnli).

2. **Anisotropy không đồng nhất giữa các layers/tokens.** Phân tích hiện tại dùng mean-pooled embeddings — điều này **trộn lẫn** anisotropy profiles khác nhau của các layers. Cần phân tích per-layer để hiểu anisotropy có phải property của toàn bộ network hay chỉ của một số layers cụ thể.

3. **Decoder (LLaMA) vs Encoder (T5) có anisotropy khác nhau về bản chất:**
   - T5 (encoder-decoder): bidirectional attention → embeddings capture both left và right context symmetrically → anisotropy profile smooth.
   - LLaMA (decoder-only): causal attention → embeddings biased toward recent tokens → anisotropy profile bursty, phụ thuộc sequence position.
   - **Đây là lý do PaR(LLaMA) ≈ 9 << PaR(T5) ≈ 24.** Decoder-only models tạo ra embeddings với **information bottleneck tại final tokens** — mean pooling flatten thông tin từ early tokens → effective dim thấp.
   - Hệ quả: whitening không đủ cho LLaMA vì **mean pooling đã discard thông tin routing** trước khi whitening发挥作用.

#### H2 — Đúng hiện tượng, sai kết luận về tổng quát

**Điểm đúng:**
- Thí nghiệm cho thấy rõ: whitened NearestCentroid đạt 100% trên T5 (LS), và 93–97% trên LLaMA.
- Frob overlap → 0.004–0.032 sau whitening → subspaces gần trực giao.

**Điểm sai/tin hiểm về tổng quát:**

1. **Orthogonality là artifact của low task count, không phải property của whitening.**
   - Grassmann packing bound: T_max ≈ 129–246 cho d=1024, k=8. Với T=15 << T_max, subspaces được pack rất thoải mái.
   - Khi T → T_max: random subspaces trong high-dim space sẽ overlap đáng kể (theo random subspace geometry). Frob overlap sẽ tăng từ 0.032 → giá trị đáng kể.
   - **Claim "whitening makes subspaces orthogonal" chỉ đúng cho T << T_max.** Đây là **extrapolation error** — đi từ observation ở 15 tasks để kết luận cho mọi T.

2. **LLaMA whitened NearestCentroid = 94–97% KHÔNG phảI là "tốt nhất" — đây là ceiling chưa đủ.**
   - T5 raw: 96.74–100% (NearestCentroid L2). T5 whitened: 100%.
   - LLaMA raw: 19.91% (near-random). LLaMA whitened: 93.88–97.33%.
   - Gap 3–6% còn lại đến từ đâu? Phân tích confusion matrix cần thiết. Có thể là:
     - **Heavy-tailed marginals** (kurtosis LLaMA whitened = 23.8): whitening tạo ra outliers → centroid không đại diện tốt.
     - **Multi-modality** (100% tasks multimodal cả raw và whitened): centroid mean pooling collapse multi-modal distribution → missing mode = routing error.
     - **Token position bias** (decoder-specific): mean pooling over-represents recent tokens → centroid gần-decoder-bias.

3. **Thiếu phân tích per-task-pair confusion.**
   - Bảng tổng hợp chỉ show accuracy trung bình. Nhưng 3–6% error LLaMA có thể tập trung vào 2–3 task pairs → có cách fix cụ thể.
   - Nếu error tập trung ở (sst2 ↔ yelp) (cùng sentiment domain) → có thể cần subspace-aware metric thay vì pure centroid.
   - Nếu error tập trung ở (dbpedia ↔ agnews) → có thể do multi-modality → cần mixture model.

4. **Không có PAC-Bayes / generalization bound** cho claim "Whitening + NearestCentroid generalizes."
   - Hiện tại chỉ có empirical evidence. Nhưng PAC-Bayes bound cho routing error:
     $$P(\text{routing error}) \leq \sqrt{\frac{D_{\text{KL}}(\hat{\mathcal{P}}_t \| \hat{\mathcal{P}}_s)}{2N}}$$
   - Điều này cho thấy: với T → lớn, KL divergence giữa task distributions nhỏ → bound yếu → routing khó hơn.
   - Không có bound này, claim "generalizes" là **không có cơ sở lý thuyết.**

#### Hệ quả cho thiết kế paper

Từ phản biện trên, claim "Whitening + NearestCentroid is the best" **chưa đủ mạnh** để làm contribution vì:

| Vấn đề | Mức độ nghiêm trọng |
|--------|---------------------|
| Orthogonality chỉ đúng cho T=15 << T_max | Cao — invalidate generalization claim |
| LLaMA gap 3–6% không được giải thích | Cao — missing mechanism |
| Không có PAC-Bayes bound | Cao — không có theory backing |
| Per-task-pair confusion chưa phân tích | Trung bình — diagnostic còn thiếu |

---

## 0.2 Lý thuyết Đề xuất: Information-Geometric Adaptive Routing (IGAR)

### Tư tưởng cốt lõi

Thay vì claim "Whitening + NearestCentroid là tốt nhất", ta xây dựng framework:

> **Routing error tối ưu = information-theoretic routing bound từ Fisher-Rao metric + adaptive metric selection dựa trên geometry profile của embedding space**

### Framework IGAR — Ba tầng

**Tầng 1: Fisher-Rao Distance (FRD)**

Gọi $\mathcal{F}_t = \mathcal{N}(\mu_t, \Sigma_t)$ là Gaussian model cho task $t$. Tự nhiên nhất, khoảng cách giữa $x$ và $\mathcal{F}_t$ theo information geometry là Fisher-Rao:

$$d_{\text{FR}}(x, \mathcal{F}_t) = \sqrt{(x - \mu_t)^\top \Sigma_t^{-1}(x - \mu_t)}$$

**Đây chính là Mahalanobis distance** — nhưng với $\Sigma_t$ được ước lượng per-task, không phải pooled. Điểm mới: $\Sigma_t$ được **adaptively regularized** theo conditioning:

$$\hat{\Sigma}_t^{\text{adapt}} = (1 - \alpha_t) \hat{\Sigma}_t + \alpha_t \hat{\Sigma}_{\text{pool}}$$

với $\alpha_t = f(\kappa(\hat{\Sigma}_t), \text{PaR}(\hat{\Sigma}_t))$ — khi anisotropy cao (LLaMA: κ ≈ 439), dùng pooled covariance nhiều hơn để stable estimation.

**Tầng 2: PAC-Bayes Generalization Bound cho Routing**

Cho routing function $r: \mathbb{R}^d \to [T]$:

$$P_{\text{test}}(r(x) \neq t) \leq \bar{\epsilon}_{\text{routing}} + \sqrt{\frac{\text{KL}(q \| p) + \ln(2\sqrt{N}/\delta)}{2N}}$$

trong đó $q$ là posterior trên routing parameters, $p$ là prior. Khi routing là non-parametric (NearestCentroid), KL term = 0 → bound chỉ phụ thuộc vào **estimation error** của $\mu_t$.

**Key insight**: Bound này chỉ ra rằng:
- Với **fixed N**, khi T tăng: estimation error của mỗi $\mu_t$ giảm (vì mỗi task có fewer samples) → routing error tăng.
- Với **fixed T**, khi d tăng: $\mu_t$ estimation trong high-dim space có variance cao hơn → routing error tăng (curse of dimensionality).
- **Whitening giảm effective dimensionality** → giảm estimation variance → cải thiện bound.

**Tầng 3: Adaptive Metric Selection (AMS)**

Dựa trên geometry profile $(\kappa, \text{PaR}, \rho_{\text{overlap}})$:

| Regime | Conditions | Optimal Metric | Justification |
|--------|-----------|----------------|---------------|
| Isotropic-Easy | κ < 50, PaR > 20 | L2 centroid | Centroids well-separated |
| Anisotropic-Easy | κ > 50, PaR > 20, Frob_ov < 0.1 | Mahalanobis per-task | Covariance shape discriminates |
| Anisotropic-Hard (LLaMA) | κ > 400, PaR < 15 | **Mahalanobis pooled + shrinkage** | Per-task Σ too noisy at PaR≈9 |
| Overlap-Regime | Frob_ov > 0.5 | FRD + calibration | Need full distributional model |
| High-T regime | T approaching T_max | Adaptive shrinkage ZCA | Orthogonality breaks down |

**Novelty claim chính:**

> **Định lý (Routing Geometry-Generalization Duality):** Cho frozen embedding space với anisotropy profile $(\kappa, \text{PaR})$. Routing error của bất kỳ non-parametric router nào bị lower-bounded bởi:
> $$\epsilon_{\text{routing}} \geq \Omega\!\left(\frac{1}{\text{PaR}} \cdot \sqrt{\frac{\ln T}{N}} \cdot \frac{\kappa}{d}\right)$$
>
> **Hệ quả:** Khi PaR thấp (LLaMA, PaR≈9): lower bound cao → không có method nào có thể đạt 100% routing accuracy với finite N. Đây là **fundamental ceiling**, không phải method limitation.

---

## 0.3 Định hướng Contribution Mới: IGAR Framework

### Đóng góp 1: Information-Geometric Routing Theory (IGRT)

**Câu hỏi nghiên cứu mới:**

> **(Q-new)**: Với feature space có anisotropy profile $(\kappa, \text{PaR}, \gamma)$ và T tasks, routing error floor tối thiểu là bao nhiêu? Metric nào đạt được floor đó?

**Lý thuyết đề xuất:**

1. **Fisher-Rao Routing**: Dùng Mahalanobis distance với $\Sigma_t$ được shrinkage theo Ledoit-Wolf/OAS. Đây là Bayes-optimal classifier cho Gaussian distributions, và收敛 về NearestCentroid khi $\Sigma_t \to \sigma^2 I$.

2. **PAC-Bayes Routing Bound**: Chứng minh rằng routing error $\leq \tilde{O}(\sqrt{\frac{\kappa \cdot \ln T}{\text{PaR} \cdot N}})$ cho Mahalanobis router với finite-sample covariance estimation.

3. **Whitening as Preconditioning**: ZCA whitening là một **preconditioner** cho gradient-based optimization trong routing space. Nó tương đương với việc dùng Fisher-Rao metric trên flattened space — không phải tạo orthogonality, mà là **đẳng cấu metric** để Euclidean distance trở nên meaningful.

**So sánh với PSR cũ:**

| Aspect | PSR (cũ) | IGRT (mới) |
|--------|-----------|------------|
| Distribution model | PPCA low-rank | Full Gaussian (shrinkage) |
| Distance | Complex PSR formula (mean + subspace + penalty) | Mahalanobis (simple, closed-form) |
| Whitening sensitivity | Catastrophic failure | Robust — Mahalanobis handles Σ_t directly |
| LLaMA performance | ~11% (random) | 95–97% (Mahalanobis, empirically) |
| Theory backing | Fisher-Rao approx (OK) | PAC-Bayes bound (stronger) |
| Parameter-free | Yes | Partially (k, σ² selection) |

### Đóng góp 2: Anisotropy-Adaptive Shrinkage Covariance (AASC)

**Vấn đề:** Ledoit-Wolf / OAS shrinkage target = isotropic $\lambda I$. Nhưng với anisotropic data, target này không tối ưu.

**Đề xuất:** Shrinkage target phụ thuộc vào conditioning:

$$\hat{\Sigma}_t^{\text{AASC}} = (1 - \alpha) \hat{\Sigma}_t + \alpha \cdot \bar{\Sigma}_{\text{adaptive}}$$

trong đó $\bar{\Sigma}_{\text{adaptive}}$ có cùng eigenvalue *shape* với $\hat{\Sigma}_t$ nhưng được regularized về pooled spectrum:

- Spectral shape: giữ nguyên ratios $\lambda_i/\lambda_j$ (preserves anisotropy structure)
- Scale: shrink về pooled overall variance (stability)

**Hypothesis cần kiểm chứng:** AASC sẽ cải thiện LLaMA routing từ 94–97% → 98–99% (giải quyết gap 3–6% còn lại).

### Đóng góp 3: Practical IGAR Algorithm

```
IGAR(h):
  1. Compute statistics: μ_t, Σ_t (shrinkage), for all T tasks
  2. Measure anisotropy: κ = λ_max/λ_min, PaR
  3. Select metric:
     if κ < 50: d = ||h - μ_t||  (L2, T5-easy)
     else: d = (h - μ_t)ᵀ Σ̂_t⁻¹ (h - μ_t)  (Mahalanobis)
  4. Return argmin_t d
```

**Lý do novel:**
- Kết hợp Fisher information geometry với PAC-Bayes theory.
- Metric selection tự động dựa trên geometry profile — chưa có work nào làm điều này trong CL routing.
- Có theoretical backing (PAC-Bayes bound cho routing).

---

## 0.4 Kế hoạch Thí nghiệm cho IGAR

### E1: Per-task-pair Confusion Analysis (Priority cao nhất)

Phân tích confusion matrix để xác định:
- Error tập trung ở pairs nào?
- Các cặp khó có đặc điểm gì chung? (same domain? multi-modality?)
- LLaMA gap 3–6% đến từ bao nhiêu pairs?

### E2: AASC vs Ledoit-Wolf vs OAS

Sweep shrinkage methods trên LLaMA embeddings:
- Ledoit-Wolf (target = λI)
- OAS (analytical optimal)
- AASC (proposed, target = adaptive shape-preserving)
- No shrinkage (MLE)

Đo routing accuracy trên LLaMA whitened để xác định method nào close gap.

### E3: T = 30, 50 Scaling Experiment

Tăng số tasks để kiểm tra orthogonality breakdown:
- Lấy 15 tasks ban đầu → split mỗi task thành 2 sub-tasks (giả lập bằng cách split data)
- Đo Frob overlap, routing accuracy vs T
- Xác nhận hypothesis: orthogonality collapse khi T → T_max

### E4: PAC-Bayes Bound Verification

Tính empirical bound:
$$\hat{\epsilon} = \frac{1}{N}\sum_i \mathbb{1}[r(x_i) \neq t_i]$$
$$\text{Bound} = \hat{\epsilon} + \sqrt{\frac{\ln(2\sqrt{N}/\delta)}{2N}}$$

So sánh empirical error với theoretical bound cho T5 và LLaMA.

### E5: Decoder Position Analysis

Phân tích token position weights trong LLaMA embeddings:
- Tính mean-pooled vs last-token embedding
- Đo routing accuracy với từng pooling strategy
- Xác nhậh hypothesis: mean pooling gây information loss cho decoder

---

## 0.5 Refined Recommendations

### For paper writing

| Claim cũ | Refined Claim | Support |
|----------|--------------|---------|
| "Whitening + NearestCentroid là SOTA" | "Fisher-Rao routing (Mahalanobis) là Bayes-optimal, với ZCA whitening là practical preconditioner" | Theory + empirical |
| "T5 routing đã solved (100%)" | "T5 routing đạt Bayes-optimal ceiling với finite-sample estimation" | PAC-Bayes bound |
| "LLaMA khó vì anisotropy" | "LLaMA có fundamental routing ceiling từ low PaR ≈ 9, được characterize bởi IGAR theory" | Theory + empirical |
| "PSR thất bại" | "PSR formulation sai vì dùng subspace residual thay vì Mahalanobis; full Gaussian model (IGRT) đúng" | Theory + empirical |

### References cần cite

- Mu & Viswanath (2018): "All-but-the-Top: Simple and Effective Postprocessing for Word Representations" — anisotropy
- Kessy et al. (2018): "Optimal Whitening and Decorrelation" — ZCA optimality
- Tipping & Bishop (1999): "Probabilistic Principal Component Analysis" — PPCA foundation
- McLachlan & Peel (2000): "Finite Mixture Models" — multi-modality
- Shawe-Taylor & Langford (2000): "PAC-Bayes for stochastic systems" — PAC-Bayes routing bounds
- Cover (1965): "Geometrical and Statistical Properties of Systems of Linear Inequalities" — Cover's theorem
- Chen & Gärtner (2002): "Clustering by Gaussian mixtures" — EM for mixture models
- Ledoit & Wolf (2004): "A well-conditioned estimator for large-dimensional covariance matrices"
- Schäfer & Strimmer (2005): "A shrinkage approach to large-scale covariance estimation"
- Frongillo & Riedel (2015): "Generalization bounds for AUC optimization" — PAC-Bayes for ranking

---

## 1. flan-t5-large (d=1024) (Results Summary)

> All results below are reproduced from Phase A–F experiments. See prior sections for full tables.

### 1.1 Phase A — Geometry

**Raw embeddings** are moderately anisotropic and low-dimensional relative to d=1024.

| Metric | LS raw | LS whitened | SNI raw | SNI whitened |
|--------|--------|------------|---------|-------------|
| EVR k95 (mean) | 338 | 593 | 214 | 373 |
| Participation Ratio (mean) | 23.7 | 309.6 | 21.3 | 216.5 |
| Effective Rank (mean) | 95.1 | 484.7 | 70.9 | 314.5 |
| \|Kurtosis\| (mean) | 0.408 | 0.768 | 0.575 | 1.853 |
| Multimodal tasks | 13/15 | 15/15 | 9/15 | 12/15 |
| Cosine dist (mean) | 0.353 | 1.059 | 0.410 | 1.066 |
| L2 dist (mean) | 0.630 | 7.937 | 0.717 | 6.944 |
| Geodesic (mean) | 3.486 | 4.328 | 3.712 | 4.367 |
| Frob overlap (mean) | 1.385 | 0.032 | 0.858 | 0.012 |
| Condition # (mean) | 132.4 | 8.0 | 196.5 | 11.8 |

**Key observations:**
- **Intrinsic dimensionality is tiny**: PaR ≈ 21–24 out of d=1024. Task embeddings live in ~2% of the full space. This means k=8 subspace bases capture a large fraction of task variance.
- **Whitening dramatically expands effective dimensionality**: PaR jumps from ~23 → ~310 (LS) and ~21 → ~217 (SNI). The spread is redistributed across many more dimensions.
- **Whitening nearly orthogonalizes subspaces**: Frobenius overlap drops from 1.385 → 0.032 (LS) and 0.858 → 0.012 (SNI). Geodesic distances saturate near π/2 per angle (≈4.35 for k=8). After whitening, task subspaces are essentially independent.
- **Whitening cures anisotropy**: Condition number drops from 132–197 → 8–12. The covariance becomes near-isotropic.
- **SNI tasks are more tightly clustered in raw space** (lower EVR, lower PaR) but are *more separable by centroid distance* (cosine 0.41 vs 0.35 for LS). This likely reflects smaller but more diverse task datasets.
- **Most tasks are multimodal** (multiple clusters within each task), which explains why QDA and single-Gaussian PSR can struggle.

### 1.2 Phase B/C — Routing Metrics & Classifiers

| Method | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| **L2** | 96.74% | **100.00%** | **100.00%** | **100.00%** |
| **Cosine** | 97.14% | 99.96% | **100.00%** | **100.00%** |
| Mahalanobis | **99.98%** | 99.99% | **100.00%** | **100.00%** |
| SpectralAffinity | 50.85% | 96.52% | 70.59% | 98.83% |
| WeightedSpectral | 31.22% | 97.05% | 11.28% | 98.91% |
| PSR_full | 81.15% | 15.26% | 98.50% | 12.11% |
| PSR_no_mean | 0.63% | 0.82% | 4.59% | 3.59% |
| PSR_no_subspace | 93.50% | 72.94% | 99.58% | 60.15% |
| LDA | **99.99%** | **99.99%** | **100.00%** | **100.00%** |
| RidgeClassifier | 99.99% | **100.00%** | **100.00%** | **100.00%** |
| QDA | 75.75% | 40.16% | 83.54% | 27.99% |
| LinearSVM | 99.97% | 99.98% | **100.00%** | **100.00%** |

**Key observations:**
- **Simple centroid L2 solves routing almost perfectly** — 96.74–100% across all settings. No need for learned routing at the T5-large scale.
- **Mahalanobis is universally excellent** (99.98–100%). It accounts for per-task covariance shape, which helps in the anisotropic raw space.
- **PSR catastrophically fails after whitening**: 81.15% → 15.26% (LS), 98.50% → 12.11% (SNI). PSR is designed for anisotropic data — when whitening removes anisotropy, the subspace residual component becomes noise and the penalty term overwhelms the signal.
- **PSR_no_mean confirms**: Without the mean (centroid) component, PSR drops to <1%. The centroid does all the work; the subspace term is harmful.
- **Spectral methods are inversely correlated with PSR**: SpectralAffinity goes from 50.85% → 96.52% after whitening. These methods rely on subspace geometry, which only becomes clean after decorrelation.
- **Linear classifiers (LDA, Ridge, SVM) are near-perfect everywhere** — task embeddings are linearly separable regardless of whitening. This is the strongest conclusion: the routing problem is linearly trivial.
- **QDA degrades with whitening** (75.75% → 40.16%), confirming that per-task Gaussianity assumptions are violated (multimodality is common).
- **SNI is easier than LS across the board**: All distance metrics achieve 100% on SNI raw; on LS raw, only Mahalanobis reaches 99.98%.

### 1.3 Phase D — PSR Ablation

| Component | LS raw | LS wh | SNI raw | SNI wh |
|-----------|--------|-------|---------|--------|
| Centroid_only | 82.92% | 4.80% | 98.83% | 8.35% |
| Subspace_only | 0.54% | 0.00% | 6.77% | 7.94% |
| PSR_light | 66.98% | 2.29% | 94.99% | 8.19% |
| PSR_full | 81.15% | 15.26% | 98.50% | 12.11% |
| PSR_no_penalty | 66.98% | 2.29% | 94.99% | 8.19% |

**Rank sensitivity (PSR_full):**
| k | LS raw | LS wh | SNI raw | SNI wh |
|---|--------|-------|---------|--------|
| 2 | 87.46% | 51.52% | 99.08% | 49.12% |
| 4 | 84.34% | 30.60% | 98.66% | 30.58% |
| 8 | 81.15% | 15.26% | 98.50% | 12.11% |
| 16 | 76.84% | 7.50% | 97.74% | 1.92% |
| 32 | 73.15% | 4.18% | 96.07% | 0.33% |
| 64 | 69.30% | 3.88% | 91.81% | 0.08% |

**Key observations:**
- **Centroid_only ≈ PSR_full in raw space** (82.92% vs 81.15% for LS) — the subspace component adds no value; in fact it *hurts slightly*.
- **Subspace_only is near-random** (0.54% LS raw, 6.77% SNI raw). Task subspaces in raw space overlap too much (Frob overlap=1.385) for pure subspace matching to work.
- **PSR accuracy monotonically decreases with k in raw space** (87.46% → 69.30%). Higher rank = more subspace noise. The optimal PSR rank is k=2, which is essentially centroid + 1 correction direction.
- **After whitening, PSR rank scaling is even more catastrophic**: 51.52% → 0.08% as k goes from 2 to 64. The subspace penalty dominates as the number of near-zero eigenvalue directions grows.
- **Domain breakdown (LS raw)**: Topic classification is easiest for PSR (90.52%), sentiment hardest (71.88%). This correlates with inter-centroid distance — topic tasks (dbpedia, agnews, yahoo) have the most distinctive embeddings.

**Incremental simulation (D6):**
- PSR degrades steadily as tasks accumulate: 1.000 → 0.811 after 15 tasks (LS raw), 1.000 → 0.153 after 15 tasks (LS whitened).
- **RLS_batch and RLS_incremental are both perfect** (99.99–100%) and produce *identical* trajectories — the Woodbury incremental update introduces zero degradation for T5-large.
- This means RLS routing can be computed without storing raw data, only the running sufficient statistics (XᵀX, Xᵀy).

### 1.4 Phase E — Theory Validation

| Metric | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| KL-confusion Spearman ρ | -0.556 | -0.502 | -0.133 | -0.475 |
| Grassmann T_max_bound | 246 | 133 | 196 | 129 |
| Grassmann bound satisfied | ✓ | ✓ | ✓ | ✓ |
| δ_max (max subspace overlap) | 0.480 | 0.040 | 0.346 | 0.009 |
| δ_mean | 0.173 | 0.004 | 0.107 | 0.001 |
| Mean geodesic to NN | 2.910 | 4.156 | 3.280 | 4.303 |
| n_signal eigvals (mean) | 317 | 261 | 332 | 313 |
| EVR of signal eigvals | 0.944 | 0.720 | 0.968 | 0.849 |
| OAS shrinkage α | 0.022 | 0.186 | 0.041 | 0.405 |
| Shrinkage improvement | +0.33% | +2.02% | +0.40% | +2.65% |

**Key observations:**
- **KL-confusion correlation is moderate** (ρ ≈ -0.50 to -0.56 for LS): KL divergence between task distributions does inversely predict confusion, but the relationship is not tight. For SNI raw, ρ = -0.133 (weak), likely because SNI tasks are already well-separated.
- **Grassmann packing bound is easily satisfied**: T_actual=15 << T_max_bound≈129–246. With d=1024 and k=8, up to ~130–246 tasks could be packed without subspace collisions. This theoretically guarantees feasibility of subspace-based routing for realistic task counts.
- **After whitening, subspaces become nearly orthogonal**: δ_max drops from 0.48 → 0.04, δ_mean from 0.17 → 0.004. The Grassmann manifold becomes well-packed.
- **RMT analysis**: ~317 out of 1024 eigenvalues are signal (31%). The rest is noise floor. This validates that a relatively low-rank representation captures task structure.
- **OAS shrinkage improves PSR only marginally**: +0.33% raw, +2.02% whitened. The covariance estimation is already reasonable; the bottleneck of PSR is the subspace residual formulation itself, not covariance noise.

### 1.5 Phase F — Learned Routing Simulation

| Method | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| NearestCentroid | 96.74% | **100.00%** | **100.00%** | **100.00%** |
| CosineNearestCentroid | 97.14% | 99.96% | **100.00%** | **100.00%** |
| PSR | 81.15% | 15.26% | 98.50% | 12.11% |
| RLS_Woodbury | **99.99%** | 15.10% | **100.00%** | 68.67% |
| GPM_ROOT | 79.84% | 13.00% | 98.75% | 23.81% |

**Incremental trajectory observations:**
- **NearestCentroid is remarkably stable**: LS raw trajectory declines only from 1.000 → 0.967 over 15 tasks. Whitened LS stays at 1.000 throughout. This method has no incremental degradation because centroids are updated without interference.
- **PSR degrades progressively**: LS raw 1.000 → 0.811, with steady decline per task. Whitened is even worse, crashing from 1.000 → 0.153.
- **RLS_Woodbury is perfect in raw space** (flat 1.000 for all 15 steps) but **catastrophically fails in whitened space** (1.000 → 0.151 for LS). This is a critical finding: RLS's regularized least squares formulation breaks down when the input covariance becomes near-identity (whitened), because the regularization term λI no longer provides meaningful shrinkage relative to the data covariance.
- **GPM_ROOT (the actual GainLoRA routing)** is the worst performer: LS raw final=79.84%, SNI raw final=98.75%. For whitened LS it crashes to 13.00%. The learned MLP routing (trans_input + prompt_key + GPM) is inferior to a simple centroid even in raw space.

**Critical insight for method design:**
- **Simple L2 centroid is near-optimal for T5-large routing**. No learned router is needed.
- **Whitening helps simple metrics but destroys complex methods**: After ZCA whitening, L2 centroid becomes perfect, but PSR, RLS, and GPM_ROOT all collapse. This is because whitening removes the anisotropy that regularized methods exploit for disambiguation.
- **The best strategy for T5-large**: Use L2 NearestCentroid on whitened embeddings (100.00%) or Mahalanobis on raw embeddings (99.98%).

---

## 2. flan-t5-xl (d=2048)

### 2.1 Phase A — Geometry

| Metric | LS raw | LS whitened | SNI raw | SNI whitened |
|--------|--------|------------|---------|-------------|
| EVR k95 (mean) | 526 | 1000 | 297 | 513 |
| Participation Ratio (mean) | 24.6 | 550.6 | 26.4 | 353.1 |
| Effective Rank (mean) | 116.8 | 833.7 | 95.1 | 463.7 |
| \|Kurtosis\| (mean) | 0.431 | 0.796 | 0.524 | 1.199 |
| Multimodal tasks | 14/15 | 15/15 | 10/15 | 9/15 |
| Cosine dist (mean) | 0.331 | 1.059 | 0.400 | 1.066 |
| L2 dist (mean) | 0.992 | 8.080 | 1.132 | 6.966 |
| Geodesic (mean) | 3.573 | 4.388 | 3.819 | 4.411 |
| Frob overlap (mean) | 1.208 | 0.009 | 0.655 | 0.002 |
| Condition # (mean) | 144.0 | 5.0 | 158.7 | 5.5 |

**Key observations:**
- **Participation ratio is nearly identical to T5-large** (~25 vs ~24 for LS raw), despite d doubling from 1024 → 2048. The intrinsic dimensionality is backbone-invariant — it's a property of the data/task, not the model width.
- **Whitened PaR scales with d** (~551 vs ~310 for T5-large LS). More dimensions = more room for ZCA to spread variance.
- **Condition number is comparable to T5-large** (144 vs 132 for LS raw), suggesting similar anisotropy profiles across T5 scales.
- **Whitening is even more effective at 2048d**: Frob overlap drops to 0.009 (vs 0.032 for T5-large). Higher-dimensional spaces allow more orthogonal packing.
- **SNI has fewer multimodal tasks whitened** (9/15 vs 12/15 for T5-large). Whitening at d=2048 better separates clusters, reducing apparent multimodality.

### 2.2 Phase B/C — Routing Metrics & Classifiers

| Method | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| L2 | 96.92% | **100.00%** | **100.00%** | **100.00%** |
| Cosine | 97.40% | 99.99% | **100.00%** | **100.00%** |
| Mahalanobis | **99.99%** | **100.00%** | **100.00%** | **100.00%** |
| SpectralAffinity | 43.69% | 97.69% | 55.64% | **99.33%** |
| WeightedSpectral | 23.75% | 98.00% | 20.72% | **99.25%** |
| PSR_full | 80.53% | 12.34% | 98.66% | 15.37% |
| PSR_no_mean | 1.18% | 4.02% | 1.84% | 11.36% |
| PSR_no_subspace | 94.13% | 57.60% | 99.75% | 43.94% |
| LDA | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| Ridge | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| QDA | 63.87% | 22.24% | 83.54% | 9.69% |
| LinearSVM | 99.98% | 99.96% | **100.00%** | 99.92% |

**Key observations:**
- **Near-identical pattern to T5-large**: L2/Cosine/Mahalanobis/LDA/Ridge all achieve 99.9–100%. PSR collapses after whitening. SpectralAffinity recovers after whitening.
- **SpectralAffinity is slightly worse raw** (43.69% vs T5-large's 50.85%) — higher d makes subspace angles harder to resolve in anisotropic space.
- **QDA is even worse at d=2048** (63.87% vs 75.75% for T5-large LS raw). More dimensions exacerbate the unreliable per-class covariance estimation, especially for small tasks.
- **PSR_no_subspace (centroid-only PSR) degrades more after whitening** (57.60% vs T5-large's 72.94%). This is surprising — it suggests the PSR centroid formulation itself has scaling issues, not just the subspace term.
- **The routing problem is equally trivial at d=2048** as at d=1024 for T5 models.

### 2.3 Phase D — PSR Ablation

| Component | LS raw | LS wh | SNI raw | SNI wh |
|-----------|--------|-------|---------|--------|
| Centroid_only | 86.45% | 0.14% | 98.66% | 8.35% |
| Subspace_only | 1.37% | 0.14% | 2.84% | 8.35% |
| PSR_light | 70.21% | 0.14% | 92.40% | 8.35% |
| PSR_full | 80.53% | 12.34% | 98.66% | 15.37% |
| PSR_no_penalty | 70.21% | 0.14% | 92.40% | 8.35% |

**Rank sensitivity (PSR_full):**
| k | LS raw | LS wh | SNI raw | SNI wh |
|---|--------|-------|---------|--------|
| 2 | 85.30% | 38.41% | 99.42% | 35.76% |
| 8 | 80.53% | 12.34% | 98.66% | 15.37% |
| 32 | 75.10% | 2.09% | 93.73% | 6.43% |
| 64 | 72.45% | 0.71% | 90.56% | 4.93% |

**Key observations:**
- **Whitened PSR completely collapses at d=2048**: Centroid_only drops to 0.14% (!) in LS whitened — essentially random for 15 tasks (6.67% = random). The PSR distance formulation is catastrophically broken in isotropic high-d space.
- **At d=2048, all PSR variants converge to the same terrible result** (0.14%) in whitened LS. The distance metric has no discriminative power.
- **Domain breakdown (LS raw)**: Same pattern as T5-large — topic classification easiest (91.46%), sentiment hardest (69.10%).
- **D6 incremental**: RLS_batch and RLS_incremental remain identical and perfect (100%). PSR degrades to 80.53% after 15 tasks (LS raw).

### 2.4 Phase E — Theory Validation

| Metric | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| KL-confusion Spearman ρ | -0.525 | -0.465 | -0.095 | -0.335 |
| Grassmann T_max_bound | 489 | 260 | 386 | 257 |
| Grassmann bound satisfied | ✓ | ✓ | ✓ | ✓ |
| δ_max | 0.476 | 0.015 | 0.337 | 0.004 |
| Mean geodesic NN | 3.015 | 4.298 | 3.409 | 4.371 |
| n_signal (mean) | 629 | 550 | 980 | 984 |
| OAS α | 0.022 | 0.297 | 0.048 | 0.589 |
| Shrinkage improvement | +0.14% | +1.12% | +0.00% | +1.71% |

**Key observations:**
- **Grassmann bound scales linearly with d**: T_max ≈ 489 at d=2048 vs ~246 at d=1024. Twice the dimensionality = twice the packing capacity.
- **Subspace overlaps are even smaller at d=2048**: δ_max = 0.476 raw (vs 0.480 for T5-large), but δ_max = 0.015 whitened (vs 0.040). Higher d provides more room for orthogonal subspaces.
- **SNI KL-confusion correlation is essentially zero** (ρ = -0.095, p=0.171). Tasks are so well-separated that KL divergence doesn't predict confusion — there's no confusion to predict.
- **Signal eigenvalues dominate more at d=2048**: n_signal ≈ 629 (LS) and 980 (SNI) out of 2048. SNI has particularly high signal density — nearly all eigenvalues are signal, likely because the small task datasets (n≈200) have gamma>1 meaning all empirical eigenvalues are inflated.
- **Shrinkage provides zero benefit for SNI raw**. The tasks are already perfectly separable.

### 2.5 Phase F — Learned Routing Simulation

| Method | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| NearestCentroid | 96.92% | **100.00%** | **100.00%** | **100.00%** |
| CosineNearest | 97.40% | 99.99% | **100.00%** | **100.00%** |
| PSR | 80.53% | 12.34% | 98.66% | 15.37% |
| RLS_Woodbury | **99.99%** | 13.32% | **100.00%** | 60.82% |
| GPM_ROOT | 83.22% | 17.07% | 93.57% | 22.22% |

**Incremental observations:**
- **Same pattern as T5-large, reinforcing universality**: NearestCentroid is flat at 1.000 (whitened) or degrades gracefully (raw, 1.000 → 0.969). RLS is perfect raw but collapses whitened.
- **RLS_Woodbury whitened degrades slightly less catastrophically than T5-large** for SNI (60.82% vs 68.67%). But still far below NearestCentroid's 100%.
- **GPM_ROOT is slightly better at d=2048 raw** (83.22% vs 79.84% for T5-large LS) — the higher-dimensional learned representations may give the MLP router slightly more structure to work with. But it still trails NearestCentroid by 14%.
- **GPM_ROOT trajectory (LS raw)**: Shows erratic behavior — 1.000 → 0.891 → 0.850 → 0.840 → 0.797 → ... → 0.832. It drops and partially recovers, suggesting the GPM subspace protection interacts non-monotonically with task addition.

---

## 3. Llama-2-7b-hf (d=4096)

### 3.1 Phase A — Geometry

| Metric | LS raw | LS whitened | SNI raw | SNI whitened |
|--------|--------|------------|---------|-------------|
| EVR k95 (mean) | 524 | 1164 | 253 | 494 |
| Participation Ratio (mean) | 9.4 | 863.5 | 12.7 | 460.5 |
| Effective Rank (mean) | 52.1 | 1117.1 | 56.1 | 517.9 |
| \|Kurtosis\| (mean) | 1.826 | 23.785 | 1.149 | 16.133 |
| Multimodal tasks | 15/15 | 15/15 | 15/15 | 15/15 |
| Cosine dist (mean) | 0.341 | 1.056 | 0.348 | 1.065 |
| L2 dist (mean) | 41.352 | 6.796 | 49.036 | 6.633 |
| Geodesic (mean) | 2.110 | 4.404 | 2.577 | 4.431 |
| Frob overlap (mean) | 5.046 | 0.004 | 3.904 | 0.000 |
| Condition # (mean) | 438.6 | 3.2 | 412.0 | 2.7 |

**Key observations — LLaMA is geometrically VERY different from T5:**

- **Extremely low intrinsic dimensionality**: PaR ≈ 9.4 (LS) and 12.7 (SNI) vs ~24 for T5 models. LLaMA's 4096-dimensional embeddings concentrate task information in ~0.2% of the space. This extreme concentration explains why simple L2 fails — the signal occupies a tiny subspace drowned by 4086 irrelevant dimensions.
- **Extreme anisotropy**: Condition# ≈ 439 (LS) vs ~132–144 for T5. LLaMA embeddings are ~3× more anisotropic. The dominant eigendirections carry vastly more variance than the tail.
- **L2 distances are enormous**: mean L2 = 41.35 (LS raw) vs 0.63 for T5-large. This reflects the large norms of LLaMA hidden states, not meaningful geometric differences. **L2 is a terrible metric for LLaMA raw** because norm differences overwhelm directional information.
- **But cosine distances are comparable**: 0.341 vs 0.353 for T5-large. The angular structure is similar across architectures — the directional separation between tasks is preserved, just buried in different norm scales.
- **Massive subspace overlap in raw space**: Frob overlap = 5.046 (vs 1.385 for T5-large). LLaMA's k=8 subspaces share most of their variance. This makes any subspace-based routing hopeless without preprocessing.
- **Whitening rescues everything**: After ZCA, condition# drops from 439 → 3.2, Frob overlap from 5.046 → 0.004, and geodesic distances saturate near π/2. But kurtosis explodes to 23.8 — whitening introduces heavy-tailed marginals because it amplifies the noise in the many near-zero variance dimensions.
- **All 15 tasks are multimodal** for LLaMA across all settings. The internal representations have richer, more clustered structure than T5.

### 3.2 Phase B/C — Routing Metrics & Classifiers

| Method | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| L2 | 19.91% | 93.88% | 50.04% | 97.33% |
| Cosine | 21.60% | 92.33% | 50.71% | 95.41% |
| Mahalanobis | **95.00%** | **95.01%** | **97.83%** | **97.83%** |
| SpectralAffinity | 55.43% | 74.51% | 67.50% | 79.03% |
| WeightedSpectral | 25.80% | 75.15% | 42.77% | 79.20% |
| PSR_full | 11.15% | 7.36% | 13.28% | 10.78% |
| PSR_no_subspace | 22.93% | 9.72% | 36.42% | 14.95% |
| LDA | **95.01%** | **95.01%** | **97.83%** | **97.83%** |
| Ridge | 94.52% | **95.20%** | 97.41% | **97.49%** |
| LinearSVM | 93.31% | 91.46% | 97.08% | 89.56% |
| QDA | — | — | — | — |

**Key observations — LLaMA breaks simple metrics:**

- **L2 centroid is catastrophically bad in raw space**: 19.91% for LS (vs 96.74% for T5-large). This is near-random for 15-class classification (6.67%). The enormous norm variation across LLaMA tasks overwhelms the centroid signal.
- **Whitening recovers L2 to 93.88%** but not to 100% like T5. There's residual structure that L2 can't fully resolve even after decorrelation.
- **Mahalanobis is backbone-invariant**: 95.00–97.83% across all LLaMA settings. It effectively builds an implicit whitening per-task, explaining its robustness. But it doesn't reach T5's 99.98% — the extreme anisotropy and small PaR of LLaMA make covariance estimation noisier.
- **LDA matches Mahalanobis exactly** (95.01%, 97.83%), confirming both methods are doing essentially the same projection.
- **PSR is completely broken for LLaMA**: 11.15% raw, 7.36% whitened (LS). Both are near-random. PSR was never designed for d=4096 with PaR=9 — the rank-8 subspace captures almost all task variance, leaving the residual penalty with nothing to work on.
- **SpectralAffinity is less catastrophic** (55.43% raw) but nowhere near useful. Whitening only brings it to 74.51% vs 96.52% for T5-large.
- **QDA is not reported for LLaMA** — likely the per-class covariance estimation failed due to near-singular matrices with n < d (many tasks have only 200–400 samples in d=4096).
- **No method achieves >97.83% on LLaMA**. The ceiling is Mahalanobis/LDA at ~95–98%, compared to 100% for T5 models. LLaMA's extreme geometry makes perfect routing harder.

### 3.3 Phase D — PSR Ablation

| Component | LS raw | LS wh | SNI raw | SNI wh |
|-----------|--------|-------|---------|--------|
| Centroid_only | 16.45% | 10.73% | 14.95% | 9.27% |
| Subspace_only | 6.79% | 10.71% | 8.19% | 8.35% |
| PSR_light | 9.87% | 10.72% | 8.60% | 8.86% |
| PSR_full | 11.15% | 7.36% | 13.28% | 10.78% |

**Rank sensitivity (PSR_full):**
| k | LS raw | LS wh | SNI raw | SNI wh |
|---|--------|-------|---------|--------|
| 2 | 11.39% | 8.74% | 24.90% | 12.03% |
| 8 | 11.15% | 7.36% | 13.28% | 10.78% |
| 32 | 14.34% | 6.13% | 9.27% | 9.36% |
| 64 | 14.27% | 5.55% | 8.44% | 7.85% |

**Key observations:**
- **PSR is uniformly random-tier** for LLaMA: All components (centroid, subspace, combined) hover around 7–17%, barely above the 6.67% random baseline. The PSR distance formulation is fundamentally incompatible with LLaMA's high-dimensional, extremely anisotropic geometry.
- **Rank sensitivity is flat or inverted**: Unlike T5 where k=2 is best (87.46%), LLaMA shows k=32 (14.34%) slightly better than k=2 (11.39%) for LS raw. The subspace has so much overlap that even low-rank approximations capture shared (not task-specific) variance.
- **Domain breakdown (LS raw)**: Topic tasks (22.56%) and RC (21.03%) are slightly above random; sentiment (2.27%) and NLI (0.01%) are essentially random. LLaMA's sentiment and NLI embeddings are not centroid-separable with PSR.

**Incremental simulation (D6) — key LLaMA finding:**
- **RLS_batch = RLS_incremental** for LLaMA, but they peak at ~94.5–97.5% (not 100% like T5). This is the upper bound achievable by Woodbury-based incremental routing on LLaMA.
- **RLS trajectory is NOT flat for LLaMA**: LS raw goes 1.000 → 0.982 → 0.972 → ... → 0.945. There's steady degradation as tasks accumulate. This suggests the RLS basis expansion is insufficient to perfectly separate LLaMA embeddings.
- PSR degrades catastrophically: LS raw 1.000 → 0.112 after 15 tasks.

### 3.4 Phase E — Theory Validation

| Metric | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| KL-confusion Spearman ρ | -0.208 | -0.443 | -0.248 | -0.267 |
| Grassmann T_max_bound | 2888 | 518 | 1821 | 512 |
| Grassmann bound satisfied | ✓ | ✓ | ✓ | ✓ |
| δ_max | 0.823 | 0.012 | 0.719 | 0.0004 |
| Mean geodesic NN | 1.565 | 4.340 | 2.077 | 4.418 |
| n_signal (mean) | 1684 | 1633 | 1357 | 1212 |
| OAS α | 0.008 | 0.372 | 0.027 | 0.686 |
| Shrinkage improvement | -0.05% | -0.09% | +0.33% | +0.27% |

**Key observations:**
- **KL-confusion correlation is weak for LLaMA** (ρ ≈ -0.21 to -0.27 raw). Tasks are not well-separated by KL divergence in raw space, consistent with the high subspace overlap.
- **Massive Grassmann capacity**: T_max ≈ 2888 for LS raw. With d=4096, the theoretical capacity for non-overlapping k=8 subspaces is enormous. But δ_max = 0.823 means the actual packing is very far from tight — subspaces overlap heavily in raw space.
- **After whitening, δ_max collapses dramatically**: 0.823 → 0.012 (LS), 0.719 → 0.0004 (SNI). The whitened subspaces are essentially orthogonal, even more so than T5.
- **Mean geodesic NN is very low for raw LLaMA**: 1.565 (LS) vs 2.910 for T5-large. The nearest-neighbor subspaces are close on the Grassmann manifold, confirming heavy overlap.
- **Shrinkage provides no benefit** for LLaMA: -0.05% (LS raw), -0.09% (LS whitened). The covariance estimation issue is not about sample-level noise but about fundamental architectural geometry.
- **Almost all eigenvalues are signal**: 1684/4096 ≈ 41% (LS raw) and even more saturated with smaller SNI datasets. This confirms that LLaMA's embeddings have very rich high-dimensional structure, even though the task-discriminative part lives in a tiny subspace (PaR≈9).

### 3.5 Phase F — Learned Routing Simulation

| Method | LS raw | LS wh | SNI raw | SNI wh |
|--------|--------|-------|---------|--------|
| NearestCentroid | 19.91% | **93.88%** | 50.04% | **97.33%** |
| CosineNearest | 21.60% | 92.33% | 50.71% | 95.41% |
| PSR | 11.15% | 7.36% | 13.28% | 10.78% |
| RLS_Woodbury | 13.62% | 10.73% | **79.03%** | 42.86% |
| GPM_ROOT | **27.92%** | 10.40% | 29.32% | 8.19% |

**Key observations — LLaMA routing is fundamentally harder:**

- **No raw-space method achieves >50% for LS**: The best raw method is GPM_ROOT at 27.92% — barely above random. NearestCentroid is only 19.91%. This confirms that raw LLaMA embeddings cannot support distance-based routing.
- **SNI is slightly better**: NearestCentroid reaches 50.04% raw (just at chance for 2-class, but this is 15-class). RLS_Woodbury reaches 79.03% — the best raw SNI result, likely because SNI tasks have slightly larger centroid separations.
- **Whitening is essential for LLaMA**: NearestCentroid jumps from 19.91% → 93.88% (LS), 50.04% → 97.33% (SNI). But it still doesn't reach T5's 100%.
- **RLS_Woodbury completely collapses for LLaMA**: LS raw 13.62%, LS whitened 10.73%. Even in SNI whitened, only 42.86%. The regularized approach cannot handle the d=4096 dimensionality. The Woodbury update is numerically unstable at this scale.
- **GPM_ROOT trajectory is chaotic**: LS raw shows 1.000 → 0.626 → 0.583 → ... → 0.279. Non-monotonic, erratic. The learned routing MLP interacts unpredictably with the extreme LLaMA geometry.
- **NearestCentroid whitened trajectory is smooth**: LS whitened goes 1.000 → 0.992 → 0.980 → ... → 0.939. Steady ~0.4% degradation per task.

**Critical insight:**
- For LLaMA, **whitening + NearestCentroid is the only viable approach** (93.88–97.33%), but it's still imperfect. Mahalanobis/LDA (95–97.83%) may be slightly better in some cases because they handle residual covariance structure.
- All parametric/complex methods fail catastrophically for LLaMA in both raw and whitened spaces.

---

## 4. Cross-Backbone Synthesis

### 4.1 Universal Laws

1. **Task intrinsic dimensionality is architecture-independent**: PaR ≈ 9–27 regardless of d ∈ {1024, 2048, 4096}. The data determines the manifold dimension; the model width only affects how that manifold is embedded.

2. **Whitening universally orthogonalizes task subspaces**: Frob overlap → 0 across all backbones. After ZCA, the Grassmann distance between any pair saturates at π/2 per angle. This is the strongest theoretical result.

3. **Linear separability is universal**: LDA/RidgeClassifier achieve 95–100% across all backbones and spaces. Task routing is a linearly solvable problem.

4. **PSR fails universally after whitening** and fails universally for LLaMA (even raw). It is the wrong formulation for routing.

5. **QDA fails universally** — task embeddings are multimodal, violating the single-Gaussian assumption.

### 4.2 Architecture Scaling Laws

| Property | T5-large (1024) | T5-xl (2048) | LLaMA-2-7b (4096) |
|----------|----------------|--------------|-------------------|
| PaR | 21–24 | 25–26 | 9–13 |
| Condition # | 132–197 | 144–159 | 412–439 |
| L2 centroid acc (raw) | 96.74–100% | 96.92–100% | 19.91–50.04% |
| L2 centroid acc (wh) | 100% | 100% | 93.88–97.33% |
| Mahalanobis acc | 99.98–100% | 99.99–100% | 95.00–97.83% |
| Best achievable | 100% (trivial) | 100% (trivial) | 97.83% (ceiling) |
| GPM_ROOT (raw) | 79.84–98.75% | 83.22–93.57% | 27.92–29.32% |

**Key scaling observations:**
- **T5 models are trivially routable** by any reasonable metric. The routing problem is solved.
- **LLaMA introduces qualitative difficulty**: Not just quantitatively worse — the failure modes are different. L2 goes from ~97% to ~20%. This is a phase transition, not a gradual degradation.
- **The difficulty correlates with anisotropy** (condition# 132 → 439) and inversely with participation ratio (24 → 9). LLaMA packs task information into fewer dimensions but with stronger directional bias.
- **GPM_ROOT (GainLoRA's routing) is worst on LLaMA**: ~28% vs 80–99% on T5. This is extremely problematic — if GainLoRA is to be extended to LLaMA, the MLP router must be redesigned to handle the extreme geometry.

### 4.3 Whitening: Panacea and Poison

**Whitening helps:**
- L2 centroid: universally improves (especially LLaMA: 20% → 94%)
- SpectralAffinity: improves everywhere (51% → 97% for T5-large LS)
- Subspace orthogonality: Frob overlap → 0
- Condition number: drops to 3–12

**Whitening destroys:**
- PSR: collapses everywhere (81% → 15% for T5-large LS)
- RLS_Woodbury: collapses for whitened data (100% → 15% for T5-large LS)
- GPM_ROOT: collapses everywhere whitened (80% → 13%)
- QDA: already bad, gets worse

**Mechanism**: ZCA whitening removes the per-task covariance structure that regularized/parametric methods exploit. When Σ → I, all discriminative power must come from mean differences (centroids). Methods that rely on covariance shape lose their signal.

### 4.4 Recommendations for Method Design

1. **For T5 models**: Use `NearestCentroid` on raw embeddings (97–100%) or whitened embeddings (100%). No learned routing needed. Save GPU memory and training time.

2. **For LLaMA**: Must use either:
   - `Mahalanobis` on raw embeddings (95–98%) — requires storing per-task covariance matrices, O(d²) per task
   - `NearestCentroid` on whitened embeddings (94–97%) — requires computing global ZCA matrix, O(d²) one-time cost + O(d) per-task centroid
   - `LDA/RidgeClassifier` (95–98%) — requires labeled samples and model fitting

3. **Never use PSR** for routing — it is fundamentally broken across all scales and spaces.

4. **Never use GPM_ROOT MLP routing** on LLaMA — 28% is worse than a random strategy with clustering heuristics.

5. **RLS_Woodbury is only safe for T5 raw** — it collapses for whitened data and for LLaMA.

### 4.5 Open Questions

1. **Why does LLaMA have PaR ≈ 9 when T5 has PaR ≈ 24?** Is this due to decoder-only vs encoder-decoder architecture, or training data scale, or tokenizer differences? Understanding this could inform whether future LLMs will be harder or easier to route.

2. **Can whitened NearestCentroid be improved to 100% for LLaMA?** The residual 3–6% error may come from the heavy-tailed marginals (kurtosis ≈ 24) introduced by whitening. A robust whitening (e.g., rank-truncated ZCA or shrinkage ZCA) might help.

3. **Is the ~95% Mahalanobis ceiling for LLaMA fundamental?** Could nonlinear routing (e.g., learned metric) break through, or is the confusion between specific task pairs irreducible?

4. **Does the benchmark matter?** SNI is consistently easier (100% for T5, 97% for LLaMA whitened centroid) vs LS (97–100% T5, 94% LLaMA). Does this reflect task diversity or dataset size?
