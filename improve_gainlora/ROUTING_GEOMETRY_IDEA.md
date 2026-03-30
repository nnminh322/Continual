# Routing Geometry Analysis — Contribution Framework

> **Tài liệu nghiên cứu Contribution 1 & 2** — Bám sát settings (zero-replay, task-agnostic inference, statistical signatures allowed). Theory-first, toán học trước, implement sau.

---

# PHẦN I — ĐỘNG LỰC VÀ KHUNG LÝ THUYẾT

---

## 1. Bài toán Routing dưới Góc nhìn Hình học

### 1.1 Bản chất Routing

Tại inference, ta nhận input $x$ **không có task ID**. Cần xác định:

$$t^* = \arg\max_{t \in [T]} P(\text{task}=t \mid h(x))$$

trong đó $h(x) = \text{Pool}(\text{Embed}(x)) \in \mathbb{R}^d$ là embedding trung bình từ backbone **đóng băng** (T5 encoder hoặc LLaMA embedding layer).

**Quan sát cốt lõi**: Vì backbone đóng băng, $h(x)$ là hàm **bất biến** qua toàn bộ quá trình CL. Mỗi task $t$ sinh ra một phân phối $\mathcal{P}_t$ trên $\mathbb{R}^d$. **Routing = phân loại phân phối (distribution discrimination)**.

### 1.2 Tại sao Hình học Feature Space Quyết định Routing

Hiệu quả routing phụ thuộc vào ba yếu tố hình học:

| Yếu tố | Ý nghĩa | Ảnh hưởng |
|---------|---------|-----------|
| **Separation** (tách biệt) | $\mu_t$ và $\mu_s$ cách xa nhau | Cross-domain dễ route |
| **Compactness** (tập trung) | Phân phối $\mathcal{P}_t$ có variance thấp | Ít nhầm lẫn |
| **Anisotropy** (bất đẳng hướng) | Eigenvectors chính của $\Sigma_t$ phân biệt | Same-domain có thể phân biệt qua subspace |

**Vấn đề**: Các phương pháp hiện tại chọn distance metric / routing algorithm **mà không biết** hình học thực tế của feature space. Ví dụ:
- Spectral affinity (SpecRoute) giả định subspace orthogonality → thất bại khi same-domain tasks có subspace chồng lấn.
- Cosine similarity (GainLoRA) giả định tín hiệu nằm ở hướng mean → bỏ qua cấu trúc covariance.
- RLS (V11) giả định linear separability trong expanded space → chi phí lưu trữ $O(E^2)$.

### 1.3 Câu hỏi Nghiên cứu

**Q1** (Geometry): Bản chất hình học của $\{\mathcal{P}_t\}_{t=1}^T$ trong embedding space là gì? Có phải Gaussian? Sub-Gaussian? Nằm trên đa tạp con (submanifold)?

**Q2** (Metric): Distance metric nào phù hợp nhất với hình học đó? Cosine, $\ell_2$, Mahalanobis, Grassmann geodesic, hay Wasserstein?

**Q3** (Algorithm): Routing algorithm nào khai thác tối ưu cấu trúc đó? Centroid-based, subspace-based, distribution-based, hay discriminative?

**Q4** (Few-shot): Bao nhiêu mẫu đủ để xây dựng task signature đáng tin cậy? Có vấn đề curse of dimensionality không?

**Q5** (Backbone): T5 (encoder, bidirectional) và LLaMA (decoder, autoregressive) có hình học khác nhau không? Cần phương pháp khác nhau?

---

## 2. Framework Lý thuyết: Probabilistic Subspace Routing (PSR)

### 2.1 Từ Hình học đến Routing — Ba Cấp độ Trừu tượng

**Level 0 — Point-based (Nearest Centroid):**
$$t^* = \arg\min_t \|h - \mu_t\|^2$$

Chỉ dùng first moment. Thất bại khi $\mu_{\text{yelp}} \approx \mu_{\text{amazon}}$ (same-domain sentiment).

**Level 1 — Subspace-based (Spectral Affinity):**
$$t^* = \arg\max_t \frac{\|V_t^\top h\|^2}{\|h\|^2}$$

Chỉ dùng principal subspace. Thất bại khi subspaces chồng lấn (same-domain).

**Level 2 — Distribution-based (Full Model):**
$$t^* = \arg\max_t \log p(h \mid \text{task}=t) = \arg\min_t \; d_{\text{PSR}}(h, t)$$

Dùng **toàn bộ** mô hình phân phối. Đây là Bayes-optimal routing.

### 2.2 Mô hình Phân phối: Probabilistic PCA (PPCA)

Mỗi task $t$ được mô hình hóa bởi low-rank Gaussian (Tipping & Bishop, 1999):

$$h \mid \text{task}=t \;\sim\; \mathcal{N}(\mu_t,\; \Sigma_t), \qquad \Sigma_t = V_t \Lambda_t V_t^\top + \sigma_t^2 I_d$$

trong đó:
- $\mu_t \in \mathbb{R}^d$: task centroid (first moment)
- $V_t \in \mathbb{R}^{d \times k}$: principal subspace ($k$ eigenvectors chính), $V_t^\top V_t = I_k$
- $\Lambda_t = \text{diag}(\lambda_{t,1}, \ldots, \lambda_{t,k})$: eigenvalue excess, $\lambda_{t,i} > 0$
- $\sigma_t^2$: isotropic residual variance

**Tại sao PPCA?**
1. **Low-rank**: Nén $d \times d$ covariance thành $O(dk)$ tham số → phù hợp lưu trữ.
2. **Có lý thuyết**: Maximum likelihood solution của factor analysis model.
3. **Kết nối SVD/GPM**: $V_t$ chính xác là top-$k$ eigenvectors từ sample covariance — cùng loại thống kê mà GPM đã lưu.
4. **Sub-Gaussian nếu data sub-Gaussian**: Concentration inequalities vẫn áp dụng.

### 2.3 Bayes-Optimal Routing dưới PPCA — Công thức Chính

Sử dụng nghịch đảo Woodbury cho $\Sigma_t^{-1}$:

$$\Sigma_t^{-1} = \frac{1}{\sigma_t^2}\left(I - V_t \left(\Lambda_t + \sigma_t^2 I\right)^{-1} \Lambda_t V_t^\top\right)$$

Log-likelihood (bỏ hằng số chung):

$$\boxed{d_{\text{PSR}}(h, t) = \sum_{i=1}^{k} \frac{\lambda_{t,i}}{\sigma_t^2(\lambda_{t,i} + \sigma_t^2)} \left(v_{t,i}^\top (h - \mu_t)\right)^2 + \frac{\|h - \mu_t\|^2}{\sigma_t^2} + \sum_{i=1}^{k} \ln(\lambda_{t,i} + \sigma_t^2) + (d-k)\ln \sigma_t^2}$$

Routing: $t^* = \arg\min_t \; d_{\text{PSR}}(h, t)$


**Phân tích các thành phần:**

| Thành phần | Ý nghĩa hình học | Khi nào chi phối |
|------------|-------------------|------------------|
| $\sum_i \frac{\lambda_{t,i}}{\sigma_t^2(\lambda_{t,i}+\sigma_t^2)} (v_{t,i}^\top \delta)^2$ | **In-subspace Mahalanobis**: khoảng cách theo hướng principal, co giãn theo eigenvalue | Same-domain (mean gần, subspace khác) |
| $\frac{\|\delta\|^2}{\sigma_t^2}$ | **Isotropic residual**: khoảng cách Euclidean chuẩn hóa | Cross-domain (mean xa) |
| $\sum \ln(\lambda_{t,i}+\sigma_t^2) + (d-k)\ln\sigma_t^2$ | **Complexity penalty**: phạt task có variance lớn | Cân bằng giữa tasks có kích cỡ khác nhau |

### 2.4 Các special case — PSR Subsumes Existing Methods

**Case 1: $k \to 0$ (không dùng subspace)**
$$d_{\text{PSR}} \to \frac{\|h - \mu_t\|^2}{\sigma_t^2} + d\ln\sigma_t^2 \quad \Rightarrow \quad \textbf{Nearest centroid (normalized)}$$

**Case 2: $\mu_t = 0,\; \sigma_t^2 \to 0$ (pure subspace, centered)**
Likelihood $\to -\infty$ khi $h \notin \text{span}(V_t)$, tức chỉ $V_t^\top h$ xác định routing → **Spectral affinity**.

**Case 3: $k = d$ (full covariance)**
$$d_{\text{PSR}} \to (h-\mu_t)^\top \Sigma_t^{-1} (h-\mu_t) + \ln|\Sigma_t| \quad \Rightarrow \quad \textbf{QDA (Quadratic Discriminant Analysis)}$$

**Case 4: $\Sigma_t = \Sigma$ (shared covariance)**
$$d_{\text{PSR}} \propto (h-\mu_t)^\top \Sigma^{-1} (h-\mu_t) \quad \Rightarrow \quad \textbf{LDA (Linear Discriminant Analysis)}$$

**Bảng tổng hợp:**

| Phương pháp hiện tại | PSR specialization | Mất gì |
|-----------------------|-------------------|--------|
| Nearest Centroid (L2) | $k=0, \sigma_t=\sigma$ | Subspace info + per-task variance |
| Nearest Centroid (cosine) | $k=0$, chuẩn hóa $h$ | Subspace info + scale info |
| Spectral Affinity (SpecRoute) | $\mu_t=0$ | Mean displacement |
| QDA | $k=d$ | Low-rank regularization → overfit khi $n \ll d$ |
| LDA | $\Sigma_t=\Sigma$ | Heteroscedasticity (task-specific spread) |
| **PSR (proposed)** | Tunable $k$ | **None — unified** |

### 2.5 Kết nối với Các Lĩnh vực Khác (Cross-Domain Insights)

#### 2.5.1 Hình học Thông tin (Information Geometry)

Không gian các mô hình PPCA $\{(\mu_t, V_t, \Lambda_t, \sigma_t^2)\}$ hình thành một **đa tạp thống kê** (statistical manifold). Khoảng cách tự nhiên trên manifold này là Fisher-Rao metric:

$$ds^2_{\text{FR}} = \text{tr}(\Sigma^{-1} d\Sigma \cdot \Sigma^{-1} d\Sigma) + 2\, d\mu^\top \Sigma^{-1} d\mu$$

**Insight**: $d_{\text{PSR}}$ chính là **xấp xỉ bậc hai** của Fisher-Rao geodesic distance tại điểm $\mu_t$. Routing tối ưu theo PSR ≈ routing theo geodesic trên statistical manifold.

#### 2.5.2 Optimal Transport — Bures-Wasserstein (BW) Distance

Wasserstein-2 distance giữa hai Gaussians:

$$W_2^2(\mathcal{P}_s, \mathcal{P}_t) = \|\mu_s - \mu_t\|^2 + \mathcal{B}^2(\Sigma_s, \Sigma_t)$$

trong đó $\mathcal{B}^2(\Sigma_s, \Sigma_t) = \text{tr}(\Sigma_s) + \text{tr}(\Sigma_t) - 2\text{tr}\!\left(\left(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2}\right)^{1/2}\right)$ là Bures metric.

**Kết nối**: Khoảng cách BW giữa hai task captures **cả** mean shift **và** covariance mismatch. Dưới mô hình PPCA, BW distance có closed form qua eigenvalues và principal angles.

**Ứng dụng cho routing**: Thay vì so sánh $h$ (point) vs $\mathcal{P}_t$ (distribution), ta có thể so sánh $\mathcal{P}_{\text{test-batch}}$ vs $\mathcal{P}_t$ khi batch size > 1 → robust hơn.

#### 2.5.3 Grassmannian Geometry

Subspace $\text{span}(V_t) \in \text{Gr}(k, d)$ là điểm trên Grassmann manifold. Khoảng cách geodesic:

$$d_G(V_s, V_t) = \left(\sum_{i=1}^k \theta_i^2\right)^{1/2}$$

trong đó $\theta_i$ là principal angles giữa hai subspaces.

**Insight cho CL**: Grassmannian packing bound $T_{\max} \leq d/(k(1-\varepsilon))$ (đã có trong SpecRoute) cho biết **dung lượng tối đa** của hệ thống. PSR enriches Grassmannian routing bằng cách thêm mean displacement và spectral weighting.

#### 2.5.4 Random Matrix Theory (RMT) — Few-shot Regime

Khi $n_t$ (số mẫu per task) không lớn hơn nhiều so với $d$, sample covariance $\hat{\Sigma}_t$ bị nhiễu theo luật Marchenko-Pastur:
- Eigenvalues bị "phình" (inflation) khi $d/n_t \to \gamma > 0$.
- Số eigenvalues "signal" vs "noise" cần sửa bằng shrinkage (Ledoit-Wolf) hoặc Tracy-Widom test.

**Ý nghĩa cho routing**: Nếu $n_t$ nhỏ (few-shot, một số tasks chỉ có $< 200$ mẫu), $\hat{V}_t$ không đáng tin cậy → cần regularization. PSR tự động regularize qua $\sigma_t^2$ (noise floor).

#### 2.5.5 Few-shot Learning (Prototypical Networks)

Snell et al. (2017): nearest centroid in learned embedding space ≈ optimal Bayes classifier under isotropic Gaussian. PSR mở rộng insight này cho **anisotropic** case:
- Isotropic: prototypical networks (centroid only)
- Anisotropic: PSR (centroid + subspace + spectrum)

#### 2.5.6 Topological Data Analysis (TDA)

Persistent homology captures **hình dạng toàn cục** (connected components, loops, voids) bất biến dưới deformation liên tục. Có thể tạo **topological signature** per task:
- $\beta_0$ (connected components): 1 cluster? nhiều subclusters?
- $\beta_1$ (1-dimensional holes): có cấu trúc vòng?
- Persistence diagram → vectorize (landscape, image) → routing feature bổ sung

**Ứng dụng tiềm năng**: Khi PPCA không đủ (e.g., task distributions multi-modal hoặc phi tuyến), TDA features bổ sung tín hiệu routing.

---

# PHẦN II — HAI ĐÓNG GÓP

---

## Contribution 1: Probabilistic Subspace Routing (PSR) — *Routing optimal dựa trên hình học phân phối*

### Tuyên bố Chính

> Mỗi task $t$ sinh ra phân phối $\mathcal{P}_t$ trên frozen embedding space. Ta mô hình hóa $\mathcal{P}_t$ bằng low-rank Gaussian (PPCA), lưu trữ signature $(μ_t, V_t, Λ_t, σ_t^2)$, và routing bằng Bayes-optimal criterion $d_{\text{PSR}}$. Framework này:
> 1. **Subsumes** nearest centroid, spectral affinity, LDA, QDA — mỗi phương pháp cũ là một special case.
> 2. **Giải thích** khi nào mỗi phương pháp thất bại qua KL decomposition.
> 3. **Không có tham số học** (zero drift, zero replay), incrementally updatable.
> 4. **Budget** per task: $O(dk + k + 1) \approx 4.6$KB cho $d=512, k=8$.

### Novelty Claims

**N1 — Unified Decomposition**: Chưa có work nào phân rã routing error trong CL thành mean-displacement + subspace-angle + spectral-ratio terms dưới cùng một framework. Work hiện tại hoặc dùng mean-only (prototypical), hoặc subspace-only (spectral), hoặc discriminative-only (RLS) — không có lý thuyết unified.

**N2 — Regime-Adaptive**: PSR tự động chuyển đổi giữa centroid-dominant (cross-domain) và subspace-dominant (same-domain) mà không cần hyperparameter chọn chế độ.

**N3 — Information-Geometric Foundation**: Kết nối routing CL với Fisher-Rao metric trên statistical manifold — framework khái quát hóa cho mọi backbone (T5, LLaMA, hoặc bất kỳ frozen encoder nào).

**N4 — RMT-Informed Regularization**: Tích hợp random matrix theory (Marchenko-Pastur correction, Ledoit-Wolf shrinkage) để đảm bảo routing robust ngay cả few-shot ($n_t \ll d$).

### Lý thuyết — Routing Error Decomposition

**Định lý (PSR Routing Margin)**. Cho hai tasks $s, t$ với PPCA models $(\mu_s, V_s, \Lambda_s, \sigma_s^2)$ và $(\mu_t, V_t, \Lambda_t, \sigma_t^2)$. KL divergence phân rã thành:

$$D_{\text{KL}}(\mathcal{P}_t \| \mathcal{P}_s) = \underbrace{\frac{1}{2}(\mu_t - \mu_s)^\top \Sigma_s^{-1} (\mu_t - \mu_s)}_{\text{Mean Displacement } D_\mu} + \underbrace{\frac{1}{2}\left(\text{tr}(\Sigma_s^{-1}\Sigma_t) - d + \ln\frac{|\Sigma_s|}{|\Sigma_t|}\right)}_{\text{Distributional Shape } D_\Sigma}$$

Trong đó, dưới mô hình PPCA:

$$D_\Sigma = \underbrace{\frac{1}{2}\sum_{i,j} f(\lambda_{s,i}, \lambda_{t,j}, \theta_{ij})}_{\text{Subspace Angle Term}} + \underbrace{\frac{1}{2}\left(\frac{\sigma_t^2}{\sigma_s^2} - 1 - \ln\frac{\sigma_t^2}{\sigma_s^2}\right)(d-k)}_{\text{Spectral Ratio Term}}$$

trong đó $\theta_{ij}$ là principal angles giữa $\text{span}(V_s)$ và $\text{span}(V_t)$.

**Hệ quả**: Routing error probability $P(\text{error} \mid t) \leq \sum_{s \neq t} \exp\!\left(-\frac{1}{2} D_{\text{KL}}(\mathcal{P}_t \| \mathcal{P}_s)\right)$ (Chernoff bound).

→ **Cross-domain**: $D_\mu$ lớn → error nhỏ → nearest centroid đủ.
→ **Same-domain**: $D_\mu \approx 0$, nhưng $D_\Sigma > 0$ nhờ subspace angle → PSR vẫn route đúng.
→ **Cả hai nhỏ**: Routing inherently khó → đây là fundamental limit, không phải thiếu sót method.

### Memory Budget Comparison

| Phương pháp | Lưu per task | T=15, d=512 | Updatable? | Drift? |
|-------------|-------------|-------------|-----------|--------|
| GainLoRA (learned MLP) | ~103K params | ~6.2MB | Cần GPM protect | Có |
| SpecRoute (spectral) | $V_t, \sigma_t$ | ~60KB | Đóng băng LoRA | Không |
| RLS (V11) | $R \in \mathbb{R}^{E \times E}$ shared | ~16MB (E=2048) | Woodbury | Không |
| **PSR** | $\mu_t, V_t, \Lambda_t, \sigma_t^2$ | **~70KB** | Running stats | **Không** |

PSR budget ≈ SpecRoute (cùng loại thống kê GPM cho phép), nhưng thêm $\mu_t$ (d floats) và $\sigma_t^2$ (1 float) — chi phí không đáng kể.

---

## Contribution 2: Routing-Informed Adaptive Projection (RIAP) — *Nới null-space có điều kiện từ routing confidence*

### Tuyên bố Chính

> OAP (hiện tại) dùng overlap ratio $\rho_l$ để quyết định mức relaxation. RIAP thay thế heuristic này bằng **routing confidence từ PSR**: khi routing confident (margin cao) → safe to relax; khi uncertain → strict projection.

$$\beta_l^{\text{RIAP}} = \max\!\left(\beta_{\min},\; 1 - \eta \cdot \frac{D_{\text{KL}}^{\min}(t)}{\tau_{\text{KL}}}\right)$$

trong đó $D_{\text{KL}}^{\min}(t) = \min_{s < t} D_{\text{KL}}(\mathcal{P}_t \| \mathcal{P}_s)$ là khoảng cách tới task gần nhất.

### Tại sao Novel hơn OAP

| Aspect | OAP (hiện tại) | RIAP |
|--------|----------------|------|
| Quyết định relaxation | $\rho_l = \text{tr}(P_{\text{old}} C_t) / \text{tr}(C_t)$ | $D_{\text{KL}}^{\min}(t)$ từ PSR |
| Ý nghĩa | "Overlap lớn → relax" | "Routing confident → safe to relax" |
| Lý thuyết | Heuristic (monotone nhưng không optimal) | Trực tiếp từ routing error bound |
| Coupling | OAP và routing là independent | RIAP tích hợp routing ↔ protection |

### Coupling Toàn bộ Pipeline

```
PSR signatures ←── frozen embeddings (same stored stats)
      │
      ├── Routing: d_PSR(h, t) → task assignment
      │
      └── Protection: D_KL → β_l^RIAP → null-space relaxation
                                            │
                                            └── CPI init trên relaxed subspace
```

*Contribution 2 sẽ được phát triển chi tiết SAU khi Contribution 1 được validate.*

---

# PHẦN III — KẾ HOẠCH THÍ NGHIỆM CHO CONTRIBUTION 1

---

## Phase A: Khảo sát Hình học Feature Space (Geometric EDA)

> **Mục tiêu**: Trả lời Q1 — bản chất hình học là gì?

### A1. Dimensionality hữu hiệu (Effective Dimensionality)

**Đo đạc**: Cho mỗi task $t$, tính sample covariance $\hat{\Sigma}_t$ từ embeddings train split:
- **Explained variance ratio (EVR)**: $k^*_t = \min\{k : \sum_{i=1}^k \lambda_i / \sum_{j=1}^d \lambda_j \geq 0.95\}$
- **Participation ratio**: $\text{PR}_t = (\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$ (effective number of dimensions)
- **TwoNN intrinsic dimension** (Facco et al., 2017): non-parametric, không giả định linear subspace.

**Kỳ vọng**: Task embeddings sống trong subspace chiều rất thấp ($k^* \ll d$). Nếu đúng → low-rank PPCA hợp lý.

**So sánh**: T5 (d=512/1024) vs LLaMA (d=4096) — LLaMA có thể có dimensionality tương đối thấp hơn (do d lớn).

### A2. Gaussianity Tests

**Đo đạc**: Kiểm tra xem mỗi $\mathcal{P}_t$ có gần Gaussian không:
- **Henze-Zirkler multivariate normality test** trên top-$k^*$ PCA scores.
- **Marginal Anderson-Darling tests** trên các PC scores riêng lẻ.
- **QQ-plots** cho top 5 principal components.
- **Excess kurtosis** per dimension.

**Kỳ vọng**: Gần Gaussian (do Central Limit Theorem trên average pooling). Nếu không → cân nhắc kernel density hoặc mixture model.

### A3. Task Centroid Separation

**Đo đạc**: Ma trận khoảng cách $D_{st}$ giữa các task centroids $\mu_t$:
- Cosine distance: $1 - \cos(\mu_s, \mu_t)$
- Euclidean distance: $\|\mu_s - \mu_t\|_2$
- Mahalanobis distance (pooled covariance): $(μ_s - μ_t)^\top \hat{Σ}_{\text{pool}}^{-1} (μ_s - μ_t)$

**Phân tích**:
- Same-domain clusters (yelp/amazon/imdb/sst2): $D_{st}$ nhỏ?
- Cross-domain: $D_{st}$ lớn?
- Sentiment ↔ NLI ↔ QA: hierarchical clustering dendrogram.

### A4. Subspace Analysis — Principal Angles trên Grassmannian

**Đo đạc**: Cho top-$k$ eigenvectors $V_s, V_t$ ($k = 8, 16, 32$):
- **Principal angles** $\theta_1 \leq \ldots \leq \theta_k$ giữa $\text{span}(V_s)$ và $\text{span}(V_t)$.
- **Chordal distance**: $d_c = \sqrt{k - \|V_s^\top V_t\|_F^2}$
- **Geodesic distance**: $d_g = \|\boldsymbol{\theta}\|_2$
- **Projection distance**: $d_p = \|V_s V_s^\top - V_t V_t^\top\|_F$

**Phân tích**:
- Same-domain: $\theta_1 \approx 0$ (shared dominant direction)? Liệu $\theta_k$ vẫn phân biệt?
- Cross-domain: Principal angles lớn?
- So sánh Grassmann distance với centroid distance → khi nào subspace routing thắng centroid routing?

### A5. Anisotropy và Isotropy Test

**Đo đạc**: Đo mức bất đẳng hướng:
- **Anisotropy ratio**: $\lambda_1 / \lambda_d$ (eigenvalue spread)
- **Effective rank**: $\text{erank} = \exp(H(\hat{\lambda}))$ trong đó $H$ là entropy của normalized eigenvalue distribution.
- **Isotropy score** (Mu et al., 2018): $I(\mathcal{P}_t) = \frac{\lambda_{\min}(\hat{\Sigma}_t)}{\lambda_{\max}(\hat{\Sigma}_t)}$

**Kỳ vọng**:
- T5 embeddings: anisotropic (theo literature, BERT-family embeddings tend to be anisotropic).
- LLaMA embeddings: có thể khác (decoder-only, khác cơ chế pooling).

### A6. Visualization

**Đo đạc**: Chiếu lên 2D/3D:
- **PCA** (top 2/3 PCs): global linear structure.
- **t-SNE** (perplexity sweep 5-50): local neighborhood structure.
- **UMAP** (n_neighbors sweep): preserving both local and global topology.

**Phân tích**: Tasks có tạo thành clusters phân biệt không? Same-domain overlaps ra sao?

### A7. Few-shot Sensitivity (RMT Analysis)

**Đo đạc**: Subsample training data $n_t \in \{50, 100, 200, 500, 1000, \text{all}\}$:
- Tính $\hat{\mu}_t, \hat{V}_t, \hat{\Lambda}_t$ từ mỗi subsample.
- Đo **stability**: $\|V_t^{(n)} - V_t^{(\text{all})}\|_F$ vs $n$.
- Đo **routing accuracy** của PSR dưới mỗi sample size.
- So sánh raw PSR vs PSR + Ledoit-Wolf shrinkage covariance.

**Kỳ vọng**:
- Marchenko-Pastur: khi $d/n \to \gamma > 0$, sample eigenvalues bị inflated → cần shrinkage.
- Với $d=512, n=200$: $\gamma = 2.56$ → chế độ "few-shot" nặng, cần regularization.
- Với $d=512, n=1000$: $\gamma = 0.512$ → moderate, shrinkage vẫn có ích.

---

## Phase B: Khảo sát Distance Metric

> **Mục tiêu**: Trả lời Q2 — distance nào phù hợp nhất?

### B1. Nearest Centroid với Nhiều Metrics

**Setup**: Train centroids $\mu_t$ từ train split, route test samples, đo accuracy.
- **L2**: $d(h, t) = \|h - \mu_t\|_2$
- **Cosine**: $d(h, t) = 1 - \cos(h, \mu_t)$
- **Normalized L2** (sau $\ell_2$-norm): $d(h, t) = \|h/\|h\| - \mu_t/\|\mu_t\|\|_2$
- **Mahalanobis** (pooled $\Sigma$): $d(h, t) = \sqrt{(h-\mu_t)^\top \hat\Sigma^{-1}_{\text{pool}} (h-\mu_t)}$

### B2. Subspace Distance Variants

**Setup**: Tính $V_t$ (top-$k$ eigenvectors), route bằng projection residual.
- **Spectral affinity** (SpecRoute): $\alpha_t = \|V_t^\top h\|^2 / \|h\|^2$
- **Subspace residual**: $d(h, t) = \|h - V_t V_t^\top h\|_2$
- **Weighted spectral**: $\alpha_t^w = \sum_i \lambda_{t,i} (v_{t,i}^\top h)^2 / \sum_i \lambda_{t,i}$

### B3. Distribution-based Distance (PSR)

**Setup**: Fit PPCA per task, route bằng $d_{\text{PSR}}$.
- **PSR-full**: sử dụng toàn bộ formula.
- **PSR-no-mean**: bỏ thành phần $\mu_t$ (= cải tiến spectral affinity).
- **PSR-no-subspace**: $k=0$ (= regularized nearest centroid).
- **PSR-no-penalty**: bỏ complexity term (= Mahalanobis under PPCA).

### B4. Bures-Wasserstein Distance

**Setup**: Tính $W_2(\delta_h, \mathcal{P}_t)$ — OT distance giữa point $h$ và Gaussian $\mathcal{P}_t$.
Cho point vs Gaussian: $W_2^2(\delta_h, \mathcal{P}_t) = \|h - \mu_t\|^2 + \text{tr}(\Sigma_t)$ (known closed form).

**Biến thể**: Sinkhorn divergence, sliced Wasserstein (computational efficiency).

### B5. Grassmann Geodesic + Centroid Hybrid

$$d_{\text{hybrid}}(h, t) = \alpha \cdot d_{\text{centroid}}(h, t) + (1-\alpha) \cdot d_{\text{subspace}}(h, t)$$

Tune $\alpha$ trên validation split.

---

## Phase C: Khảo sát Routing Algorithm

> **Mục tiêu**: Trả lời Q3 — algorithm nào tốt nhất?

### C1. Parametric Generative Models
- **LDA** (shared covariance)
- **QDA** (per-task covariance, shrinkage-regularized)
- **PSR** (low-rank PPCA, proposed)
- **Naive Bayes** (diagonal covariance)

### C2. Discriminative Classifiers
- **Linear SVM** (one-vs-rest)
- **RBF-SVM** (non-linear boundary)
- **Logistic Regression** (L2-regularized)
- **Ridge Regression / RLS** (V11 proposal)

### C3. Non-parametric Methods
- **k-NN** ($k \in \{1, 3, 5, 10, 20\}$, các distance metrics từ Phase B)
- **Kernel Density Estimation** (bandwidth selection via Scott's rule)

### C4. Tree-based / Ensemble
- **Random Forest** (100-500 trees)
- **XGBoost / LightGBM**
- **Naive: majority vote** từ nearest examples

### C5. Few-shot Specialized
- **Prototypical Network** (centroid in embedding space — tương đương nearest centroid, nhưng test metric learning effect)
- **Matching Network** (attention-weighted kNN)
- **MAML-style** (meta-learn routing classifier)

### C6. Subspace / Manifold Methods
- **Spectral Affinity** (SpecRoute)
- **Grassmann Nearest Subspace**
- **Riemannian kNN** on Grassmannian (nếu subspace analysis cho thấy Grassmann structure rõ)

### Protocol chung cho Phase C

```
For each algorithm:
  1. Fit trên train embeddings (tất cả 15 tasks)
  2. Evaluate trên test embeddings: accuracy, per-task confusion matrix
  3. Evaluate robustness: subsample n_t ∈ {50, 100, 200, 500, all}
  4. Report: accuracy ± std (5 random subsamples)
  5. Report: same-domain accuracy vs cross-domain accuracy riêng
  6. Report: confusion matrix → xác định task pairs dễ nhầm
```

---

## Phase D: Comparative & Ablation Analysis

> **Mục tiêu**: Phân tích sâu, validate lý thuyết.

### D1. PSR Component Ablation

| Config | Mean | Subspace | Spectrum | Penalty | Expected behavior |
|--------|------|----------|----------|---------|-------------------|
| Centroid only | ✓ | ✗ | ✗ | ✗ | Good cross-domain, bad same-domain |
| Subspace only | ✗ | ✓ | ✗ | ✗ | Good same-domain (if orthogonal), bad cross-domain |
| PSR-light | ✓ | ✓ | ✗ | ✗ | Mean + subspace, no weighting |
| PSR-full | ✓ | ✓ | ✓ | ✓ | Best overall (hypothesis) |
| PSR-shrinkage | ✓ | ✓ | ✓ (LW) | ✓ | Best few-shot (hypothesis) |

### D2. Rank Sensitivity ($k$)

Sweep $k \in \{2, 4, 8, 16, 32, 64\}$:
- Routing accuracy vs $k$
- Memory budget vs $k$
- Optimal $k^*$ per benchmark
- Does $k^*$ correlate with effective dimensionality from A1?

### D3. Same-domain vs Cross-domain Breakdown

- Group tasks into domain clusters (sentiment, NLI, QA, summarization, dialogue).
- Report **intra-cluster** routing accuracy (hard case) vs **inter-cluster** (easy case) separately.
- Plot: routing accuracy vs $D_{\text{KL}}(\mathcal{P}_s, \mathcal{P}_t)$ → validate theoretical prediction.

### D4. T5 vs LLaMA Comparison

- Run **tất cả Phase A-C** trên cả T5 và LLaMA embeddings.
- Compare: effective dimensionality, anisotropy, cluster separation, best metric/algorithm.
- Hypothesis: LLaMA (d=4096) có thể có better separation nhưng cần stronger regularization.

### D5. Task Order Sensitivity

- For Long_Sequence Order 3 vs Order 4: routing accuracy thay đổi không?
- Theory: PSR routing **bất biến** theo task order (vì dùng frozen embeddings).
- Empirical validation.

### D6. Incremental Update Quality

- Simulate CL scenario: tại task $t$, chỉ có signatures $\{(\mu_1, V_1, ...), ..., (\mu_t, V_t, ...)\}$.
- Routing accuracy at task $t = 2, 3, ..., 15$.
- Compare: PSR (incremental mean + covariance) vs RLS (Woodbury) vs Oracle (tất cả data).

---

## Phase E: Validation Lý thuyết

### E1. KL Decomposition validates Routing Difficulty

- Tính $D_\mu, D_\Sigma$ cho mọi task pair $(s, t)$.
- Plot: routing confusion rate vs $D_{\text{KL}}$.
- Verify: task pairs có $D_{\text{KL}}$ nhỏ ↔ routing confusion lớn.

### E2. Grassmannian Packing Bound

- Compute pairwise $\delta_{ij} = \|V_i^\top V_j\|_F^2$.
- Verify: $T_{\max} \leq d/(k(1-\varepsilon))$ holds.
- Xác nhận: worst-case overlap $\delta_{\max}$ cho biết routing margin lower bound.

### E3. RMT Prediction

- Compare empirical eigenvalue distribution $\hat{\lambda}_i$ với Marchenko-Pastur prediction at ratio $\gamma = d/n$.
- Verify: eigenvalue correction (shrinkage) cải thiện routing ở few-shot regime.

---

# PHẦN IV — KẾ HOẠCH THỰC THI

---

## Thứ tự ưu tiên

```
Tuần 1: Phase A (geometric EDA) — script Python, chạy trên embeddings đã có
         → Xác nhận PPCA hypothesis, chọn k, hiểu geometry
Tuần 2: Phase B + C1-C2 (distance + generative/discriminative classifiers)
         → So sánh PSR vs nearest centroid vs spectral vs QDA
Tuần 3: Phase C3-C6 + Phase D (non-parametric, tree, ablations)
         → Complete routing comparison, validate PSR advantage
Tuần 4: Phase E (theory validation) + Write-up
         → Confirm KL decomposition, Grassmann bound, RMT prediction
```

## Deliverables

1. **`routing_analysis/analyze_geometry.py`**: Phase A scripts (dimensionality, Gaussianity, visualization).
2. **`routing_analysis/compare_routing.py`**: Phase B-C scripts (distance metrics, algorithms).
3. **`routing_analysis/ablation_psr.py`**: Phase D scripts (ablation, T5/LLaMA comparison).
4. **`routing_analysis/validate_theory.py`**: Phase E scripts (KL decomposition, RMT).
5. **Figures & tables** cho paper.

## File Output Conventions

Tất cả kết quả lưu tại `routing_analysis/results/`:
- `geometry_{model}_{benchmark}.json` — dimensionality, anisotropy, Gaussianity metrics
- `distance_matrix_{model}_{benchmark}_{metric}.npy` — pairwise task distances
- `routing_accuracy_{model}_{benchmark}.csv` — comparison table
- `confusion_{model}_{benchmark}_{method}.npy` — per-method confusion matrices

---

# PHẦN V — TÓM TẮT

| | Contribution 1 (PSR) | Contribution 2 (RIAP) |
|---|---|---|
| **Giải quyết** | Routing (task identification) | Protection (null-space relaxation) |
| **Novelty** | Unified distributional routing framework; subsumes centroid/spectral/QDA | Routing confidence → adaptive protection |
| **Foundation** | Information geometry + PPCA + RMT | PSR routing margin → relaxation bound |
| **Cross-domain** | Few-shot learning, Riemannian geometry, OT, TDA | Trust regions (optimization), IB theory |
| **Storage** | ~70KB / 15 tasks | Reuse PSR signatures |
| **Drift** | Zero (frozen features + frozen stats) | Zero (derived from PSR) |
| **Validate trước** | **← FOCUS HERE (Phase A-E)** | Sau khi Contribution 1 stable |

> **Nguyên tắc**: Theory predicts → Experiment validates → Nếu prediction sai, sử lý thuyết, không patch implement. (Bám sát `work_ethic.txt`.)
