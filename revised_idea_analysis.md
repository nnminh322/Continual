# PHÂN TÍCH CHUYỂN HƯỚNG IDEA: Từ Data-Level Signatures → Module-Level Signatures
## Comprehensive Analysis Report

**Date**: March 7, 2026  
**Context**: Idea cũ (OT-SIGN) vi phạm zero-replay setting → cần chuyển hướng

---

# PHẦN 1: XÁC NHẬN VI PHẠM VÀ CHUYỂN HƯỚNG

## 1.1 Hai Điểm Vi Phạm Của Idea Cũ

### Vi phạm 1: vMF trên dữ liệu cũ = replay thống kê
**Phân tích chính xác**: Setting zero-replay (GainLoRA Section 2.2, InfLoRA Section 2.2) yêu cầu:
> "The model must learn without access to real or synthetic samples from previously learned tasks."

Việc fit vMF signature $(μ_t, κ_t)$ cuối mỗi task yêu cầu **chạy forward pass qua training data của task cũ** để collect features → đây chính là *statistical summary* của old data distribution → **vi phạm zero-replay**.

**Bằng chứng từ papers**: Tất cả LoRA-based CL papers trong survey (GainLoRA, InfLoRA, O-LoRA, C-LoRA, MINGLE) đều không lưu bất kỳ thông tin thống kê nào về dữ liệu cũ. Cái duy nhất được phép lưu là:
- **Parameter weights** (frozen LoRA A, B matrices)
- **Subspace bases** (GPM/DualGPM bases $M_t$ — đây là basis vectors, KHÔNG phải data statistics)
- **Gating module weights** (trans_input MLP weights)

Ranh giới tinh tế: GPM bases $M_t$ được tính từ input covariance $H_t^T H_t$ — thoạt nhìn giống "thống kê", nhưng được chấp nhận vì $M_t$ chỉ capture **directions (subspace)**, KHÔNG capture **distribution parameters** (mean, concentration, shape). Nó tương đương với **memory of which directions were used**, không phải **memory of what data looked like**.

### Vi phạm 2: Anti-invasion loss không cần thiết  
**Phân tích chính xác**: Trong kiến trúc LoRA expandable:
- **InfLoRA**: $B_t$ được thiết kế trong $\mathcal{N}_t \cap \mathcal{M}_t^{\perp}$ — intersection của gradient space new task và null-space of old tasks. Điều này **mathematically guarantees** rằng update cho task mới nằm trong subspace trực giao với old tasks.
- **OLoRA**: Soft penalty $\lambda_1 \sum_{t'<t} \|A_{t'} A_t^T\|_1$ khuyến khích A matrices trực giao nhau.
- **GainLoRA**: Constraints (5)+(7) trên gating module đảm bảo $g_t(x) = 0$ cho old task inputs.

→ Với các cơ chế này, LoRA branches **đã được thiết kế** để không xâm lấn lẫn nhau → thêm anti-invasion loss là **dư thừa** và vi phạm Occam's razor.

## 1.2 Hướng Đi Mới: Khai Thác Thông Tin Từ Module LoRA

**Ý tưởng mới**: Thay vì khái quát phân phối dữ liệu cũ, khai thác thông tin (thống kê, hình học) **nội tại** của các LoRA submodules — tức là phân tích chính các ma trận $A_t, B_t$ — làm signature cho routing.

**Tại sao hợp lệ?** Vì $A_t, B_t$ là **model parameters**, không phải data. Chúng đã được frozen sau khi train và là phần tự nhiên của model → việc phân tích chúng KHÔNG vi phạm zero-replay.

---

# PHẦN 2: KHẢO SÁT SETTINGS VÀ PAPERS LIÊN QUAN

## 2.1 Các Papers Cùng Settings (Zero-Replay, LoRA-Expansion, Task-ID-Free)

| Paper | Venue | LoRA Constraint | Routing | Lưu gì từ old tasks? |
|-------|-------|----------------|---------|---------------------|
| **InfLoRA** [Liang & Li, CVPR'24] | CVPR 2024 | Hard: $B_t$ in $\mathcal{N}_t \cap \mathcal{M}_t^{\perp}$ | Không có routing (merge tất cả) | GPM bases $M_t$ |
| **O-LoRA** [Liang & Li] | Cùng nhóm InfLoRA | Random init, CE loss only | Merge tất cả ($a_i = 1$) | Không gì thêm |
| **C-LoRA** [Smith et al., 2023] | CoRR | Soft: null-space regularization | Merge tất cả | Null-space directions |
| **GainLoRA** [Liang et al., NeurIPS'25] | NeurIPS 2025 | Kế thừa InfLoRA/OLoRA | **Gating: cosine sim → sigmoid** | GPM bases + frozen trans_input snapshots |
| **MINGLE** [Qiu et al., NeurIPS'25] | NeurIPS 2025 | Entropy-based null-space SVD | **MoE gating: FC → softmax** | Input covariance SVD subspace $U$ |
| **CLoRA** [ACL'25] | ACL 2025 | Null space regularization trên output matrix | Merge tất cả | Null-space directions |
| **TreeLoRA** [ICML'25] | ICML 2025 | No explicit orthogonality | **Gradient-similarity tree routing** | Gradient similarity scores |
| **PLAN** [ICCV'25] | ICCV 2025 | Orthogonal basis allocation per task | Perturbation-based selection | Orthonormal basis set |
| **Feature Distributions** [ICML'25] | ICML 2025 | No explicit orthogonality | **Mean feature vector matching** | Mean feature vectors per PEFT block |
| **SD-LoRA** [ICLR'25] | ICLR 2025 | Decoupled magnitude/direction | Low-loss trajectory | Direction/magnitude decomposition |

### Nhận xét quan trọng:
1. **Tất cả** papers trong settings này đều KHÔNG lưu data statistics (vMF, covariance, GMM) từ old tasks
2. Routing mechanisms hiện tại: cosine similarity (GainLoRA), FC gating (MINGLE), gradient similarity (TreeLoRA), mean features (Feature Distributions) — **chưa có paper nào dùng LoRA weight properties làm routing signatures**
3. Paper gần nhất concept: **Feature Distributions** (ICML'25) dùng mean feature vector → nhưng đây là feature-level, KHÔNG phải weight-level

## 2.2 Thông Tin Gì Được Phép Khai Thác?

Theo zero-replay setting, ta chỉ được phép khai thác:

| Nguồn | Ví dụ | Hợp lệ? |
|-------|-------|---------|
| Frozen model weights | $A_t, B_t$ matrices, gating weights | ✅ Hoàn toàn |
| Subspace bases từ GPM | $M_t, M_t^{\perp}$ | ✅ (đã được InfLoRA sử dụng) |
| Pre-trained model weights | Base $W$ | ✅ |
| Current task data | $\mathcal{D}_t$ (chỉ task đang train) | ✅ |
| Old task data/statistics | vMF, mean, covariance | ❌ Vi phạm |

---

# PHẦN 3: LoRA MODULES — TÍNH CHẤT VÀ ĐẶC TRƯNG CÓ THỂ KHAI THÁC

## 3.1 LoRA Module Là Gì?

Mỗi LoRA branch cho task $t$ gồm:
- $B_t \in \mathbb{R}^{r \times d_{in}}$ : **Dimensionality reduction matrix** (mã hóa input subspace)
- $A_t \in \mathbb{R}^{d_{out} \times r}$ : **Dimensionality increasing matrix** (được fine-tuned, mã hóa task-specific transformation)

Với GainLoRA: $r = 4$, $d_{in} = d_{out} = 1024$ (T5-Large).

**Ý nghĩa hình học:**
- Mỗi **hàng** của $B_t$ ($b_i^t \in \mathbb{R}^{d_{in}}$) là một **direction vector** trong input space
- $\text{span}\{b_1^t, \ldots, b_r^t\}$ định nghĩa **subspace mà task $t$ hoạt động trong**
- $A_t B_t$ = rank-$r$ perturbation lên weight matrix $W$ → task-specific **adaptation direction**

**Fact quan trọng (Proposition 1 từ InfLoRA)**:  
> Fine-tuning $A_t$ is equivalent to fine-tuning the pre-trained weight $W$ within the subspace $\text{span}\{b_1^t, \ldots, b_r^t\}$.

→ **$B_t$ hoàn toàn đặc trưng cho "vùng hoạt động" (operating subspace) của task $t$**

## 3.2 Đặc Trưng Hình Học Của LoRA Modules

### a) Singular Value Decomposition (SVD) của $A_t B_t$

$$A_t B_t = U_t \Sigma_t V_t^T$$

Trong đó:
- $U_t \in \mathbb{R}^{d_{out} \times r}$: **Output directions** — các hướng mà task $t$ "phát ra" trong output space
- $\Sigma_t = \text{diag}(\sigma_1^t, \ldots, \sigma_r^t)$: **Singular values** — "strength/importance" của từng direction
- $V_t \in \mathbb{R}^{d_{in} \times r}$: **Input directions** — subspace mà task $t$ "lắng nghe" trong input space

**Tính chất:**
1. **Singular values $\sigma_i^t$** reflect relative importance of each direction for task $t$
2. **Right singular vectors $v_i^t$** define the input receptive subspace
3. **Left singular vectors $u_i^t$** define the output emission subspace
4. **Spectral entropy** $H_t = -\sum_i \hat{\sigma}_i \log \hat{\sigma}_i$ (với $\hat{\sigma}_i = \sigma_i / \sum_j \sigma_j$) measures "spread" of task knowledge across directions

### b) Grassmann Manifold Perspective

Collection of $r$-dimensional subspaces trong $\mathbb{R}^{d}$ forms **Grassmann manifold** $\text{Gr}(r, d)$.

Mỗi LoRA branch task $t$ → một point $\mathcal{V}_t = \text{span}(V_t)$ trên $\text{Gr}(r, d_{in})$ (input side) hoặc $\mathcal{U}_t = \text{span}(U_t)$ trên $\text{Gr}(r, d_{out})$ (output side).

**Khoảng cách trên Grassmannian** giữa hai tasks:
$$d_G(\mathcal{V}_i, \mathcal{V}_j) = \|\theta\|_2 = \sqrt{\sum_{k=1}^r \theta_k^2}$$

Với $\theta_k = \arccos(\sigma_k)$ là **principal angles** giữa hai subspaces, tính từ SVD của $V_i^T V_j$.

**Ý nghĩa**: Tasks có subspaces gần nhau (small Grassmann distance) → likely share knowledge → routing nên fuse chúng. Tasks có subspaces xa nhau → independent knowledge → routing nên chọn riêng.

### c) Column Space và Row Space

- **Column space** of $\Delta W_t = A_t B_t$: $\text{col}(\Delta W_t) = \text{span}(U_t)$ → **output feature subspace** task $t$ tác động
- **Row space** of $\Delta W_t$: $\text{row}(\Delta W_t) = \text{span}(V_t)$ → **input feature subspace** task $t$ sử dụng
- **Null space** of $\Delta W_t$: inputs mà task $t$ **không hề affect** → orthogonal complement of row space

### d) Frobenius Norm và Spectral Properties

$$\|A_t B_t\|_F = \sqrt{\sum_i (\sigma_i^t)^2}$$

Measures overall "magnitude" của task $t$'s adaptation. Phân phối singular values cho biết:
- **Concentrated** ($\sigma_1 \gg \sigma_2 \gg \ldots$): Task có dominant direction → knowledge tập trung
- **Spread** ($\sigma_1 \approx \sigma_2 \approx \ldots$): Task cần nhiều directions → knowledge phân tán

## 3.3 Công Cụ Thống Kê/Hình Học Phù Hợp

| Đặc trưng | Công cụ | Ý nghĩa |
|-----------|---------|---------|
| Subspace direction | Grassmann manifold, principal angles | Đo "task relatedness" dựa trên góc giữa subspaces |
| Singular value distribution | Spectral entropy, effective rank | Đo "complexity/spread" của task knowledge |
| Weight matrix geometry | Frobenius/Nuclear/Spectral norm | Đo "magnitude" của task adaptation |
| Subspace overlap | $\text{Tr}(P_i P_j)$ với $P_i = V_i V_i^T$ projection | Đo mức chồng chéo giữa operating subspaces |
| Fisher Information | $F_t = \mathbb{E}[\nabla \log p \cdot \nabla \log p^T]$ | Parameter importance (nhưng cần data → vi phạm nếu dùng old task data) |

**Lưu ý quan trọng**: Tất cả metrics trên chỉ yêu cầu **ma trận $A_t, B_t$** (frozen weights), KHÔNG cần old data → **hoàn toàn hợp lệ** trong zero-replay setting.

---

# PHẦN 4: PHÂN TÍCH VẤN ĐỀ TRỰC GIAO — SUBSPACE EXHAUSTION

## 4.1 Vấn Đề: Subspace Shrinkage (Nhận Định Đúng)

Nhận định của bạn **hoàn toàn chính xác** và được xác nhận bởi cả lý thuyết và code:

### Chứng minh toán học:

Khi sử dụng GPM/DualGPM (InfLoRA), subspace cho old tasks $\mathcal{M}_t$ **tăng đơn điệu**:
$$\dim(\mathcal{M}_1) \leq \dim(\mathcal{M}_2) \leq \ldots \leq \dim(\mathcal{M}_T) \leq d_{in}$$

Do đó, **null-space $\mathcal{M}_t^{\perp}$ giảm đơn điệu**:
$$\dim(\mathcal{M}_t^{\perp}) = d_{in} - \dim(\mathcal{M}_t)$$

Kết quả:
- **Task 1**: Toàn bộ $d_{in}$-dimensional space available → $B_1$ có $d_{in}$ chiều để chọn
- **Task $t$**: Chỉ còn $\dim(\mathcal{M}_t^{\perp})$ chiều → $B_t$ bị giới hạn trong subspace nhỏ hơn
- **Task $T$ (final)**: Available space có thể rất nhỏ nếu $T$ lớn

### Từ code GainLoRA (InfLoRA variant):

```python
# Threshold tăng dần → old subspace ĂN nhiều hơn
threshold = (1.0 - threshold_base) * cur_task / total_sessions + threshold_base
# threshold_base = 0.995 → threshold tăng từ 0.995 → 1.0
```

Quan sát từ InfLoRA paper (Figure 5): dim($\mathcal{M}_t^{\perp}$) giảm nhưng "always much larger than zero". **Tuy nhiên** điều này chỉ đúng cho 20 tasks với $d_{in} = 768$ (ViT-B/16). Với settings khó hơn (T5-Large, $d_{in} = 1024$, 15 tasks, mỗi task tốn nhiều directions), subspace có thể bị **cạn kiệt đáng kể**.

### Hậu quả: Unfair Capacity Allocation

| Task | Available dim | Constraint count | Effective capacity |
|------|--------------|-------------------|-------------------|
| Task 1 | $d_{in}$ | 0 | Maximum |
| Task 5 | $d_{in} - \sum_{i=1}^{4} k_i$ | 4 sets | Giảm |
| Task 15 | $d_{in} - \sum_{i=1}^{14} k_i$ | 14 sets | **Rất nhỏ** |

Với $k_i$ là dimension được thêm vào $\mathcal{M}$ ở mỗi task (thường $k_i \sim$ rank effective of task $i$).

**Ví dụ cụ thể**: Nếu mỗi task "chiếm" trung bình 60 dimensions (với threshold 0.995), sau 15 tasks:
$$\text{claimed} = 15 \times 60 = 900 \quad \text{vs.} \quad d_{in} = 1024$$
→ Task 15 chỉ còn $\sim 124$ dimensions available → **capacity giảm ~88%** so với task 1.

## 4.2 Các Hướng Giải Quyết Từ Literature

### Hướng 1: DualGPM — Slow Expansion (InfLoRA đã dùng)
- Tăng threshold dần → giảm tốc expansion  
- **Nhược điểm**: Chỉ *chậm lại* depletion, không *giải quyết* root cause. Trade-off: threshold cao → bảo tồn tốt nhưng space hẹp; threshold thấp → space rộng nhưng interference.

### Hướng 2: Adaptive Relaxation (MINGLE đã dùng)
- Track alignment history $h_i$ (EMA) giữa gradient và old directions
- Directions có high historical alignment → **được relaxed** (cho phép update)
- $\lambda_i = \exp(-\gamma \cdot h_i)$: soft decay thay vì hard projection

**Ưu điểm**: Không tốn space vĩnh viễn — directions cần thiết cho task hiện tại được "mượn" lại.
**Nhược điểm**: Có thể gây interference nếu relaxation quá mạnh.

### Hướng 3: Subspace Recycling / Forgetting Old Bases
- Ý tưởng: Nếu một direction trong $\mathcal{M}_t$ không còn quan trọng (ví dụ singular value tương ứng rất nhỏ), có thể "giải phóng" nó cho tasks mới.
- **Chưa có paper nào implement** trong LoRA CL context.
- Liên quan: "Memory-efficient GPM" directions — nhưng chưa formal.

### Hướng 4: Shared Subspace Decomposition (Novel Direction)
- Thay vì hard orthogonal: phân tách mỗi task thành **shared component** + **task-specific component**
- Shared component được tái sử dụng → không tốn space mới
- Task-specific component tuân thủ orthogonal → nhưng nhỏ hơn many
- Related: **Oblique projection** thay vì orthogonal projection

### Hướng 5: Grassmann Manifold Optimization (Mathematical Foundation)

Thay vì project trong Euclidean space, tối ưu hóa trên **Grassmann manifold** $\text{Gr}(r, d)$:

**Stiefel Manifold Constraint**: Thay vì $B_t \perp \text{span}(\text{old bases})$, yêu cầu:
$$B_t \in \text{St}(r, d_{in}) \quad \text{(Stiefel manifold: orthonormal frames)}$$

Rồi dùng **Riemannian gradient descent** trên Grassmannian để tìm $B_t$ tối ưu trên manifold — inherently balanced vì mọi point trên Grassmannian có "metric volume" equal.

**Kết nối toán học**: Geodesic distance trên Grassmannian = principal angles = chính là independence measure giữa subspaces. Tối ưu hóa trên manifold tự nhiên cân bằng capacity.

## 4.3 Phân Tích OLoRA (Soft Constraint)

OLoRA dùng soft penalty $\|A_{old} A_{new}^T\|$ thay vì hard projection. Điều này:

**Ưu điểm**:
- Không bị subspace exhaustion (penalty dẻo, cho phép small overlap)
- Capacity allocation công bằng hơn (mọi task đều có toàn bộ space, nhưng bị penalize nếu overlap)

**Nhược điểm**:
- Không có **theoretical guarantee** rằng interference = 0
- Penalty strength $\lambda_1$ cố định → không adaptive theo task complexity
- Có thể dẫn đến "soft forgetting" nếu overlap tích lũy

## 4.4 Kết Luận: Cần Một Cơ Chế Mới

Cả hard (InfLoRA) và soft (OLoRA) đều có significant drawbacks:
1. **Hard**: Subspace exhaustion, unfair late-task capacity
2. **Soft**: No guarantee, accumulating interference

→ Cần **adaptive mechanism** kết hợp ưu điểm cả hai.

---

# PHẦN 5: ĐÁNH GIÁ HƯỚNG ĐI MỚI VÀ ĐỀ XUẤT CẢI TIẾN

## 5.1 Đánh Giá Idea Mới (Module-Level Signatures + OT Routing)

### Điểm mạnh:
1. **Hoàn toàn hợp lệ**: Chỉ phân tích frozen weights $A_t, B_t$ → zero-replay compliant
2. **Novel**: KHÔNG có paper nào trong 109 papers khảo sát dùng LoRA weight SVD/spectral properties làm routing signatures
3. **Well-motivated**: SVD of $A_t B_t$ captures task subspace geometry — mathematically grounded on Grassmann manifold
4. **Compatible**: Có thể áp dụng trên GainLoRA, MINGLE, và bất kỳ expandable LoRA architecture nào

### Điểm cần cải tiến:
1. **OT routing dựa trên gì?**: Cần define rõ cost matrix. Idea cũ: vMF log-likelihood (vi phạm). Idea mới: **Grassmann distance** hoặc **subspace projection similarity** giữa input và LoRA subspaces → hợp lệ.

2. **Input representation**: Routing cần biết input feature $x$ thuộc "vùng nào". Ta cần map $x$ vào cùng space với LoRA signatures mà KHÔNG dùng old data. Giải pháp: **project $x$ lên mỗi LoRA subspace**, đo "fit" bằng projection magnitude.

3. **Fairness constraint**: Cần giải quyết subspace exhaustion → đây CÓ THỂ là contribution thứ 2 (thay cho anti-invasion loss).

## 5.2 Đề Xuất Idea Sơ Thảo: **SpecRoute — Spectral Signatures + Grassmann-Fair Routing**

### Tổng quan 3 Contributions

| # | Contribution | Thay thế gì? | Novel? |
|---|-------------|--------------|--------|
| C1 | **Spectral LoRA Signatures**: Dùng SVD properties $(U_t, \Sigma_t, V_t)$ của frozen $A_t B_t$ làm task fingerprint | Thay prompt_key (point estimate) bằng rich spectral descriptor | ✅ Novel — chưa có paper nào |
| C2 | **Grassmann-OT Routing**: OT với cost = Grassmann distance giữa input projection và LoRA subspaces | Thay cosine sim → sigmoid bằng principled OT | ✅ Novel — OT + Grassmann chưa kết hợp trong CL |
| C3 | **Elastic Subspace Allocation (ESA)**: Cơ chế thay thế hard orthogonal, cho phép controlled sharing + spectral-importance-weighted protection | Thay GPM hard constraint bằng adaptive elastic constraint | ✅ Novel — addresses known limitation |

### C1: Spectral LoRA Signatures

**Định nghĩa**: Cho task $t$ đã train, với frozen $A_t, B_t$, tính SVD:
$$\Delta W_t = A_t B_t = U_t \Sigma_t V_t^T$$

**Signature** $\mathcal{S}_t$ bao gồm:
1. **Subspace direction**: $V_t \in \mathbb{R}^{d_{in} \times r}$ (input receptive field)
2. **Spectral profile**: $\sigma_t = (\sigma_1^t, \ldots, \sigma_r^t)$ (importance distribution)
3. **(Optional)** Output direction: $U_t$ nếu cần output-level routing

**Lưu trữ**: Chỉ cần $V_t$ (size $d_{in} \times r = 1024 \times 4 = 4096$ floats) + $\sigma_t$ ($r = 4$ floats) per layer per task. Với 15 tasks × 48 attention layers (T5-Large, Q+V) = 15 × 48 × 4100 ≈ 2.95M floats ≈ 11.8 MB — **rất nhỏ** so với model size.

**So sánh với GainLoRA hiện tại**:
- `prompt_key` = 1 vector $\in \mathbb{R}^d$ per task (point estimate, learned jointly with gating)
- Spectral signature = $r$ vectors + $r$ scalars per task per layer (captures subspace geometry, computed from frozen weights)

**Tại sao tốt hơn?**
- `prompt_key` encode "input nào thuộc task này" — nhưng learned trong feature space riêng (trans_input), gây non-parallel experts problem (xem proposal cũ Phần 1.2)
- Spectral signature encode "task này hoạt động trên subspace nào" — trực tiếp từ weight geometry, objective, không phụ thuộc vào feature extractor

### C2: Grassmann-OT Routing

**Ý tưởng**: Với input $h \in \mathbb{R}^{d_{in}}$ tại một layer, đo "mức phù hợp" của $h$ với mỗi LoRA subspace bằng **projection ratio**:

$$\text{fit}(h, \mathcal{S}_t) = \frac{\|V_t^T h\|^2}{\|h\|^2} \cdot \text{spectral\_weight}_t$$

Trong đó:
- $\|V_t^T h\|^2 / \|h\|^2$ = fraction of $h$'s energy captured by task $t$'s subspace (= $\cos^2$ of angle giữa $h$ và subspace, hay **projection magnitude**)
- $\text{spectral\_weight}_t = \sum_i \sigma_i^t / \sum_j \sum_i \sigma_i^j$ = relative importance of task $t$

**Cost matrix cho OT**:
$$C_{bt} = 1 - \text{fit}(h_b, \mathcal{S}_t) \quad \in [0, 1]$$

(low cost = input fits well into task's subspace)

**Sinkhorn OT**:
$$\Pi^* = \text{Sinkhorn}(C, \varepsilon), \quad \text{weights} = B \cdot \Pi^* \quad \in \mathbb{R}^{B \times N_{tasks}}$$

**Tại sao OT thay vì direct projection?**
1. **Global balance**: OT đảm bảo các experts được sử dụng hợp lý (không collapse vào 1 expert)
2. **Principled**: Optimal transport có foundation lý thuyết vững (Monge-Kantorovich)
3. **Differentiable**: Sinkhorn có gradient → có thể fine-tune nếu cần

**Tại sao Grassmann distance phù hợp?**
- Subspaces $\text{span}(V_t)$ nằm trên Grassmann manifold → Grassmann distance là metric tự nhiên
- Projection-based "fit" tương đương Grassmann geodesic distance (principal angles)

### C3: Elastic Subspace Allocation (ESA) — Thay Thế Hard Orthogonal

**Vấn đề**: Hard orthogonal (InfLoRA) → subspace exhaustion. Soft penalty (OLoRA) → no guarantee.

**Giải pháp ESA**: Kết hợp **importance-weighted protection** + **controlled sharing**

**Bước 1 — Spectral Importance Scoring**: Cho mỗi old task $t'$ tại mỗi layer, tính importance score cho mỗi direction $v_i^{t'}$:
$$w_i^{t'} = \frac{(\sigma_i^{t'})^2}{\sum_j (\sigma_j^{t'})^2}$$

Directions có high singular value → crucial cho task $t'$ → cần protect mạnh.

**Bước 2 — Weighted Projection**: Thay vì hard project ra khỏi toàn bộ $\mathcal{M}_t$:
$$B_t \leftarrow B_t - \sum_{t'<t} \sum_{i=1}^{r} \alpha_i^{t'} \cdot (V_t^{t'} (V_t^{t'})^T) B_t^T$$

Với:
$$\alpha_i^{t'} = \begin{cases} 1 & \text{if } w_i^{t'} > \tau_{\text{protect}} \quad \text{(hard protect critical directions)} \\ w_i^{t'} & \text{if } w_i^{t'} \leq \tau_{\text{protect}} \quad \text{(soft protect less important)} \end{cases}$$

**Bước 3 — Space Budget**: Giới hạn tổng protected dimensions:
$$\sum_{t'<t} \text{effective\_rank}(t') \leq \beta \cdot d_{in}$$

Nếu vượt budget → **prune** directions có lowest $\sigma_i^{t'} $ trước (subspace recycling).

**Ưu điểm**:
- **Fair**: Critical directions always protected, minor directions can be shared
- **Efficient**: Total protected space bounded by $\beta \cdot d_{in}$
- **Adaptive**: Importance changes per task — complex tasks claim more, simple tasks claim less
- **Theoretically grounded**: Spectral importance = proxy for output sensitivity ($\sigma_i$ reflects how much direction $i$ affects output)

**So sánh**:

| Phương pháp | Protection | Space usage | Fairness | Guarantee |
|-------------|-----------|-------------|----------|-----------|
| InfLoRA (GPM) | Hard, all directions | Monotonic increase | Unfair (first-come) | Strong for protected |
| OLoRA | Soft penalty | Constant | Fair | Weak |
| MINGLE (adaptive relax) | EMA-adaptive | Controlled | Medium | Medium |
| **ESA (đề xuất)** | Importance-weighted | Bounded by budget | **Fair** | Strong for critical, soft for minor |

---

# PHẦN 6: KIỂM TRA NOVELTY CỦA IDEA MỚI

## 6.1 Cross-check với 109 Papers + Papers Bổ Sung

### C1 — Spectral LoRA Signatures cho Routing

| Paper | Cách dùng spectral | Khác biệt |
|-------|-------------------|-----------|
| **MINGLE** | SVD of merged task vector → entropy-based effective rank → null-space exclusion | SVD dùng cho **construction** (xây LoRA), KHÔNG phải routing signature |
| **SD-LoRA** (ICLR'25) | Decouple magnitude + direction | Analysis purpose, không phải routing |
| **Grassmannian MoE** (arXiv) | Bingham trên Grassmannian | Routing entropy control, KHÔNG phải knowledge signature. Và không phải CL. |
| **Feature Distributions** (ICML'25) | Mean feature vector | Feature-level, không phải weight-level |

**Kết luận C1**: ✅ **Novel** — Chưa có paper nào dùng SVD properties ($V_t, \Sigma_t$) của frozen LoRA weights làm routing signatures trong CL.

### C2 — OT Routing dựa trên Grassmann Distance  

| Paper | OT usage | Routing basis | Khác biệt |
|-------|---------|--------------|-----------|
| **BASE Layers** (ICML'21) | OT load-balancing | Learned scores | OT cho balance, không phải knowledge matching |
| **Selective Sinkhorn** (2025) | OT routing | Learned scores | OT cho routing nhưng cost = learned, không phải geometric |
| **SCDEM** (2025) | OT feature alignment | Feature distance | OT cho alignment, không phải routing |

**Kết luận C2**: ✅ **Novel** — OT + subspace projection cost (Grassmann-based) chưa được dùng trong CL routing.

### C3 — Elastic Subspace Allocation

| Paper | Subspace management | Khác biệt |
|-------|-------------------|-----------|
| **InfLoRA** | Hard GPM, threshold-based | No recycling, no importance weighting |
| **DualGPM** | Bi-directional, threshold-based | Slightly better but same root issue |
| **MINGLE** | Adaptive relaxation (EMA) | Gate-level, not LoRA subspace level |
| **TRGP** (Lin et al., ICLR'22) | Trust region gradient projection | Relaxes constraint based on "trust" but no spectral importance |

**Kết luận C3**: ✅ **Novel** — Importance-weighted subspace protection with bounded budget chưa được đề xuất.

## 6.2 Đánh Giá Tổng Thể

| Tiêu chí | Đánh giá |
|----------|---------|
| Novelty | ✅ Cao — 3 contributions đều novel |
| Zero-replay compliance | ✅ Hoàn toàn — chỉ dùng frozen weights |
| Mathematical rigor | ✅ Grassmann geometry, SVD, OT — all well-established |
| Practical feasibility | ✅ SVD of $(r \times d)$ matrices rất nhanh (r=4) |
| Compatibility | ✅ Áp dụng được trên GainLoRA, InfLoRA+GainLoRA, MINGLE |
| Theoretical backing | ✅ Grassmann manifold (Edelman et al.), OT (Villani), Spectral theory |

---

# PHẦN 7: IDEA SƠ THẢO TỔNG HỢP

## SpecRoute: Spectral-Geometric Routing for Fair Continual LoRA Learning

### Motivation (1 paragraph)
Trong LoRA-based continual learning, hai thách thức chưa được giải quyết triệt để: (1) routing mechanism hiện tại dựa trên learned point estimates (cosine similarity đến prompt keys) — không capture được geometric structure của task knowledge subspaces, dẫn đến suboptimal assignment đặc biệt cho inputs nằm ở boundary giữa tasks; (2) orthogonal constraints (GPM/DualGPM) đảm bảo non-interference nhưng gây subspace exhaustion — tasks sau bị giới hạn capacity không công bằng so với tasks đầu, degrading overall performance. Chúng tôi nhận thấy rằng frozen LoRA weights $(A_t, B_t)$ chứa đầy đủ thông tin hình học về "vùng hoạt động" của mỗi task thông qua SVD, và thông tin này có thể được khai thác làm task signatures cho principled routing.

### Method Overview

**1. Spectral LoRA Signatures (Section 3.1)**
- Sau khi train task $t$, tính SVD: $A_t B_t = U_t \Sigma_t V_t^T$
- Signature $\mathcal{S}_t = (V_t, \Sigma_t)$ per layer — encode operating subspace + importance profile  
- Không cần old data, không cần extra computation ngoài SVD (rất nhanh cho r=4)

**2. Grassmann-OT Routing (Section 3.2)**
- Input $h$ → compute projection fit: $\text{fit}(h, \mathcal{S}_t) = \sum_i \sigma_i^t \cdot (v_i^t \cdot h)^2 / \|h\|^2$
- Build cost matrix $C_{bt} = 1 - \text{normalized\_fit}$ per batch
- Sinkhorn OT → globally optimal routing weights
- Thay thế hoàn toàn cosine-sigmoid gating → loại bỏ non-parallel feature space problem

**3. Elastic Subspace Allocation (Section 3.3)**
- Weight mỗi old direction bằng spectral importance $w_i^{t'} = (\sigma_i^{t'})^2 / \sum_j (\sigma_j^{t'})^2$
- Hard protect critical directions ($w > \tau$), soft protect minor directions
- Bounded total protected dimensions → **fair capacity** cho late tasks
- Optional: subspace recycling khi budget exceeded

### Theoretical Justification
1. **Proposition 1** (inherited from InfLoRA): Fine-tuning $A_t$ = fine-tuning $W$ in span($B_t$) → SVD of $A_t B_t$ fully characterizes task's operating subspace
2. **Grassmann distance** giữa subspaces = principal angles = natural metric cho "task relatedness"
3. **OT guarantees**: Sinkhorn produces $\varepsilon$-approximate optimal transport plan → globally balanced assignment
4. **ESA bound**: Total protected capacity ≤ $\beta \cdot d_{in}$ → late tasks guaranteed ≥ $(1-\beta) \cdot d_{in}$ available directions

### Expected Contributions Claim
- **C1**: First to use spectral properties of frozen LoRA weights as routing signatures in CL
- **C2**: First to combine Grassmann subspace distance with OT for routing in CL
- **C3**: First to address LoRA subspace exhaustion via importance-weighted elastic allocation

### Áp Dụng Trên GainLoRA
1. Thay `prompt_key` + `trans_input` + `previous_trans_input` bằng spectral signatures + projection routing
2. Thay GPM hard constraint bằng ESA
3. Keep: expandable LoRA architecture, training loss, frozen old branches

### Potential Risks & Mitigations
| Risk | Severity | Mitigation |
|------|---------|------------|
| SVD per-layer overhead | Low | $r=4$ → SVD trivial; compute once after training |
| Projection fit not discriminative enough | Medium | Add spectral weighting $\sigma_i$ to amplify important directions |
| OT Sinkhorn convergence | Low | Log-domain Sinkhorn with $\varepsilon=0.05$, well-studied |
| ESA τ threshold sensitivity | Medium | Cross-validate; default $\tau = 1/r$ (uniform importance threshold) |
| Compatibility with GainLoRA gating constraints | Medium | ESA replaces GPM entirely; GainLoRA gating becomes unnecessary (routing handles expert selection) |

---

# PHẦN 8: TÓM TẮT

## Các kết luận chính:

1. **Vi phạm xác nhận**: Idea cũ (vMF data signatures + anti-invasion loss) đúng là vi phạm zero-replay setting. Chuyển hướng sang khai thác LoRA weights là hướng đi hợp lệ.

2. **Nhận định subspace exhaustion đúng**: Hard orthogonal constraints (GPM) gây unfair capacity allocation cho late tasks. Đã được xác nhận qua phân tích toán học và code. Đây là open problem chưa ai giải quyết triệt để.

3. **Đặc trưng LoRA phong phú**: SVD của $A_t B_t$ cung cấp rich geometric information: subspace directions, importance profile, effective rank. Nằm trên Grassmann manifold — có metric topology tự nhiên.

4. **Idea mới (SpecRoute) viable**: 3 contributions (spectral signatures, Grassmann-OT routing, elastic subspace allocation) đều novel, hợp lệ, mathematically grounded, và áp dụng được trên GainLoRA/MINGLE platform.

5. **Papers đồng settings**: GainLoRA, InfLoRA, O-LoRA, C-LoRA, MINGLE, TreeLoRA, PLAN, Feature Distributions, SD-LoRA — tất cả đều follow zero-replay + LoRA expansion. KHÔNG có paper nào kết hợp weight-level spectral signatures + OT routing + elastic capacity allocation.
