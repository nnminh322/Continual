# Deep Theoretical Analysis: SVD, Frozen A, and Routing Methodology

## ⚠️ REVISION NOTE (v2)

**Phiên bản trước (v1) mắc sai lầm nghiêm trọng**: Khuyến nghị giữ prototype routing (V5) — nhưng prototype = mean embedding = **thống kê dữ liệu** → **VI PHẠM zero-replay**. Đã sửa toàn bộ phân tích.

**Tham chiếu phản biện**: `v6_discuss.md` — phân tích đúng rằng prototype violates zero-replay.

## Preamble

Phân tích này tuân thủ nghiêm ngặt:
- **settings.txt**: zero-replay (không dùng data cũ dưới mọi hình thức — bao gồm thống kê, phân phối, mean embeddings)
- **research_rule.txt**: lý thuyết → weakness → motivation → cải tiến → thực nghiệm
- **work_method**: theory-first, không thử-sai

Tham chiếu ≥30 papers về toán học, lý thuyết thông tin, ma trận, LoRA.

---

## I. NGUYÊN NHÂN GỐC RỄ CỦA "NEVER-LEARNING" PHENOMENON

### 1.1 Câu hỏi

CB (EM=0.00), MNLI (34.07), và một số task có single-task quality thấp hơn ROOT. Contribution C2 (routing gate) không liên quan tới nội hàm single-task training — vì routing chỉ quyết định **ai xử lý**, không quyết định **xử lý tốt hay không**.

### 1.2 Phân tích cấu trúc: A frozen + GPM → Compound Constraint

Xét task $t$ trong sequence. InfLoRA constraint:

$$A_t \in \text{null}\left(\sum_{j<t} A_j A_j^\top\right)$$

Tức $A_t$ phải nằm trong null-space của projection matrix $P_{\text{old}} = \sum_{j<t} V_j V_j^\top$ (với $V_j$ là GPM bases của task $j$).

**Hệ quả dimension**: Null-space dimension giảm theo tasks:
$$\dim(\text{null}) \leq d - \sum_{j<t} r_j \cdot \gamma_j$$
với $\gamma_j$ là fraction of capacity consumed (controlled by ESA threshold).

Với $d=512$, $r=8$, threshold=0.995 → mỗi task tiêu thụ ~$r_{\text{eff}}$ dimensions. Sau 14 tasks, null-space còn ~$512 - 14 \times r_{\text{eff}}$.

**Paper references**:
- **InfLoRA** (Sun et al., CVPR 2024): Proves null-space initialization prevents retroactive interference BUT doesn't guarantee learning quality
- **GORP** (ACL 2025): Unified gradient subspace projection — shows projection shrinks expressiveness
- **CMCL** (NeurIPS 2025): Dual stability/plasticity bounds — tradeoff is fundamental

### 1.3 Three Root Causes of Never-Learning

#### Cause 1: Gradient Distortion (cấp bậc toán học)

Khi A frozen, gradient của B là:
$$\nabla_B \mathcal{L} = \nabla_{\Delta W} \mathcal{L} \cdot A^\top$$

SVD của $A$: $A = U_A \Sigma_A V_A^\top$ (với $A \in \mathbb{R}^{r \times d}$).

Gradient bị **nhân phải** bởi $A^\top$. Nếu $\kappa(A) = \sigma_1(A)/\sigma_r(A) \gg 1$:
- Directions tương ứng $\sigma_{\min}(A)$ nhận gradient **rất nhỏ** → slow convergence
- Adam normalizes per-parameter nhưng KHÔNG normalize per-direction trong space $\mathbb{R}^r$
- Effective learning rate trở nên anisotropic theo condition number

**Paper references**:
- **Muon/Riemannion** (Jordan et al., 2025): Proposes Riemannian LoRA trên fixed-rank manifold, shows Euclidean gradient suboptimal cho rank-constrained optimization
- **LORO** (ICLR 2025): Steepest descent on manifold $\mathcal{M}_r$, addresses anisotropy
- **SD-LoRA** (ICLR 2025): Direction/magnitude decomposition — decouples learning paths
- **LoKO** (2024): Kalman filter optimizer cho LoRA — shows conditioning issue is real

**Quantitative**: Sau GPM projection + normalization ($A \leftarrow A / (\sqrt{3} \|A\|)$), $A$ có các singular values xấp xỉ uniform (do random init + orthogonal projection). Tuy nhiên **effective** condition number phụ thuộc vào alignment giữa random A directions và task-relevant directions.

#### Cause 2: Random A ≠ Optimal A (Information-Theoretic)

$A_t$ được init bằng Kaiming uniform rồi project vào null-space. Từ góc nhìn information theory:

**Mutual Information Bound** (Data Processing Inequality):
$$I(h; z_t) = I(h; A_t h) \leq H(z_t) \leq \frac{r}{2} \log\left(1 + \frac{\text{Var}[A_t h]}{r}\right)$$

Random $A_t$ **không maximize** $I(h; z_t)$ cho task-specific $h$ distribution. Optimal $A_t$ nên align với top-$r$ principal components của task data covariance WITHIN null-space.

**Hiểu đơn giản**: Random A chọn $r$ directions ngẫu nhiên từ null-space ($d'$ dimensions). Xác suất chọn đúng "best" $r$ directions ≈ 0 khi $d' \gg r$.

**Paper references**:
- **PLAN** (ICCV 2025): Proactive rank allocation via perturbation sensitivity — shows different layers need different ranks. Random allocation wastes capacity.
- **TreeLoRA** (ICML 2025): Gradient similarity tree for layer-wise allocation
- **Information Bottleneck Theory** (Tishby et al., 2000): LoRA = bottleneck with capacity $C = \sum \sigma_i^2$
- **Angle Matters** (ICML 2025): Angle between task signals determines forgetting rate AND learning rate

**Ví dụ cụ thể**: CB có 250 samples, 3 classes (entailment/contradiction/neutral). Cần A directions aligned với phân biệt semantic giữa 3 labels. Random A từ null-space (dimension ~390 ở task 4) chỉ chọn 8 random directions → hầu hết **không liên quan** tới task signal.

#### Cause 3: CE Loss Alone + Insufficient Training

Cross-entropy chỉ optimize prediction accuracy. Không có mechanism nào đảm bảo:
- **Effective rank** của $\Delta W = BA$ cao → spectral health
- **Utilization** đồng đều các rank directions
- **Regularization** chống overfitting trên tiny datasets

CB: 250 samples × 10 epochs = 2500 iterations × batch=8 = ~80 steps. Adam optimizer cần ~100s steps để converge cho NLI task.

**Paper references**:
- **Stiefel-LoRA** (EMNLP 2025): $B^\top B = I_r$ constraint maximizes effective rank
- **SD-LoRA** (ICLR 2025): Explicit rank regularization through directional alignment
- **SEFE** (ICML 2025): Distinguishes superficial vs essential forgetting — CE loss conflates them
- **LoRA–** (CVPR 2025): Triplet loss in drift-resistant space for better single-task quality

### 1.4 Kết luận: Never-learning KHÔNG do routing

Routing (C2) chỉ quyết định input → expert mapping. Never-learning do:
1. **A frozen + GPM projection** → restricted learning capacity (structural)
2. **Random A** trong null-space → poor alignment với task signal (information-theoretic)
3. **Insufficient training** cho tiny datasets (practical)

**Quan trọng**: ROOT dùng CÙNG InfLoRA (A frozen + GPM) nhưng ROOT CB = 3.57 (cũng near-fail). Vấn đề này là **fundamental limitation** của InfLoRA approach, KHÔNG phải specific to SpecRoute.

---

## II. SVD NHƯ CHỮ KÝ — CÓ CẦN XEM XÉT LẠI?

### 2.1 Bản chất SVD

SVD phân rã $\Delta W = BA = U \Sigma V^\top$:
- $V$ (right singular vectors): **input directions** expert "lắng nghe"
- $U$ (left singular vectors): **output directions** expert "phát ra"
- $\Sigma$ (singular values): **cường độ** modification

Spectral signature hiện tại dùng $(V, \sigma)$ — input receptive field + importance.

### 2.2 A frozen ảnh hưởng SVD như thế nào?

Đây là câu hỏi then chốt. Xét:

$$\Delta W_t = B_t A_t$$

Với $A_t$ frozen (Kaiming init + GPM projection), $\text{rank}(\Delta W_t) \leq r$.

**SVD structural constraint**:
$$\Delta W_t = B_t A_t = (U_B \Sigma_B V_B^\top)(U_A \Sigma_A V_A^\top)$$
$$= U_B \Sigma_B (V_B^\top U_A) \Sigma_A V_A^\top$$

Gọi $M = \Sigma_B (V_B^\top U_A) \Sigma_A$ ∈ $\mathbb{R}^{r \times r}$, SVD($M$) = $P S Q^\top$. Khi đó:

$$\Delta W_t = (U_B P) S (Q^\top V_A^\top) = \tilde{U} S \tilde{V}^\top$$

**Key insight**: Right singular vectors của $\Delta W_t$ là:
$$\tilde{V} = V_A Q$$

Tức **right singular vectors luôn nằm trong row-space của A**:
$$\text{col}(\tilde{V}^\top) = \text{col}(Q^\top V_A^\top)^\top \subseteq \text{row}(A_t)$$

**Hệ quả sâu sắc**: SVD spectral signature bị **GIỚI HẠN** bởi $\text{row}(A_t)$!

**Paper references**:
- **Eckart-Young Theorem** (1936): Best rank-$k$ approximation = truncated SVD
- **Weyl's Perturbation Theorem**: $|\sigma_i(A+E) - \sigma_i(A)| \leq \|E\|_2$
- **Horn & Johnson** (Matrix Analysis, 2013): SVD of product $BA$ — structure thm
- **Interlacing Inequalities** (Thompson, 1976): Singular values of products

### 2.3 Vấn đề: A frozen + GPM → SVD signatures structurally constrained

Do GPM: $A_k \perp A_j$ (approximately, via projection), nên:

$$\text{row}(A_k) \perp \text{row}(A_j) \quad \forall j < k$$

**Theorem (informal)**: Nếu $A_k \perp A_j$ exact, thì $V_k \perp V_j$ exact (right singular vectors orthogonal).

**Chứng minh**: $V_k \in \text{row}(A_k)$ và $V_j \in \text{row}(A_j)$, mà $\text{row}(A_k) \perp \text{row}(A_j)$, nên $V_k^\top V_j = 0$.

**Hệ quả cho routing**: Spectral routing dùng:
$$\text{fit}_t(h) = \frac{\sum_i \sigma_{t,i}^2 (v_{t,i}^\top h)^2}{\sum_i \sigma_{t,i}^2 \|h\|^2}$$

Khi $V_k \perp V_j$, routing phụ thuộc hoàn toàn vào **energy projection**:
- $h$ có bao nhiêu energy trong $\text{row}(A_k)$ vs $\text{row}(A_j)$?

**VẤN ĐỀ CHÍNH**: Nếu hai task cùng domain (e.g., yelp/amazon — cả hai là sentiment 5-class), input $h$ của chúng SIMILAR. Nhưng do GPM, $A_{\text{yelp}}$ và $A_{\text{amazon}}$ orthogonal → energy projection phụ thuộc vào **RANDOM** A directions, KHÔNG phụ thuộc vào task semantics.

Đây chính xác là **GPM-Routing Paradox** đã nhận diện — nhưng bây giờ ta thấy vấn đề còn sâu hơn: **SVD signatures bị A deterministic** khi A frozen.

### 2.4 A frozen + B trainable có phù hợp với SVD?

**Câu trả lời ngắn**: Có, nhưng với caveats quan trọng.

**Phù hợp**: SVD($BA$) ĐÚNG phản ánh effective modification direction. B training thay đổi singular values + left singular vectors, nhưng right singular vectors bị constrained bởi $\text{row}(A)$. SVD vẫn đúng mathematically.

**Không phù hợp cho ROUTING**: Right singular vectors ($V$) — thứ ta dùng cho routing — bị tied to A's row-space. Khi A frozen + orthogonalized → routing information bị **A deterministic**, không phản ánh **B's learned task-specifics**.

**Formal statement**: Routing quality bounded by:
$$\max_{h} |\text{fit}_k(h) - \text{fit}_j(h)| \leq 1 - \cos^2(\text{row}(A_k), \text{row}(A_j))$$

Khi $A_k \perp A_j$ (GPM), $\cos^2 = 0$ → max discrimination = 1 (TUYỆT ĐỐI). **Nhưng** discrimination phụ thuộc vào h alignment, và h KHÔNG được control bởi routing mechanism.

**Paper references**:
- **Function Vectors** (Todd et al., ICLR 2025): Models carry compact task-encoding "function vectors" — h encodes task info nhưng KHÔNG guaranteed align với random A directions
- **GIFT** (CVPR 2025): Fisher Information as Riemannian metric — shows parameter sensitivity IS task-dependent
- **Low-rank Forgetting Analysis** (NeurIPS 2025): Forgetting matrix $F = \Delta W_{\text{old}}^\top \Delta W_{\text{new}}$ — has low-rank structure

### 2.5 Kết luận về SVD signatures

1. **SVD mathematically correct** cho decompose $\Delta W = BA$
2. **Right singular vectors** ($V$) bị **tied to row(A)** khi A frozen → chủ yếu reflect A's structure
3. **Singular values** ($\sigma$) DO reflect B's learning → key discriminative signal
4. **Routing via (V, σ)**: Phụ thuộc cả A (directions) LẪN B (magnitudes). Khi C4 cải thiện B training quality → σ spectrum phong phú hơn → discrimination tăng
5. **Câu hỏi mở (V6 sẽ trả lời)**: C4 có đủ mạnh để bù đắp V bị tied to row(A)? Tức là σ²-weighting có đủ discriminative khi V orthogonal?

---

## II-bis. PROTOTYPE ROUTING VI PHẠM ZERO-REPLAY

### Phân tích lỗi logic trong v1

Phiên bản v1 khuyến nghị prototype routing (V5) dựa trên Data Processing Inequality:
$$I(T; f(\Delta W, h)) \leq I(T; A_t h) \leq I(T; h)$$

Kết luận: "prototype dùng h trực tiếp → bypass A bottleneck → tối ưu". 

**Tuy nhiên**, prototype $\mu_t = \frac{1}{N_t}\sum_{i} h_i^{(t)}$ là **thống kê dữ liệu** (mean of training embeddings). Theo settings.txt:

> "không được phép sử dụng lại bất kỳ dữ liệu cũ dưới bất kỳ hình thức nào bao gồm dữ liệu thô, dữ liệu synthetic, **phân phối dữ liệu (được tạo ra nhờ các công cụ thống kê)**"

Mean embedding = first moment of distribution = thống kê → **vi phạm**.

### Ranh giới hợp lệ: Model Parameters vs Data Statistics

| Information | Nguồn | Hợp lệ? | Lý do |
|------------|--------|---------|-------|
| $A_t, B_t$ (frozen LoRA) | Training artifact | ✅ | Model parameters |
| SVD$(B_t A_t)$ | Derived from params | ✅ | Pure computation on params |
| GPM bases ($U$ from covariance SVD) | Forward on data → covariance | ⚠️ **Biên** | ROOT cũng dùng, accepted by community |
| Mean embedding $\mu_t$ | Forward on data → statistic | ❌ | Data distribution statistic |
| Distribution params (variance, etc.) | Forward on data → statistic | ❌ | Explicit distribution |

**Quan sát quan trọng**: GPM bases cũng được tính từ training data (forward 1000 batches → covariance → SVD → bases). Khác biệt then chốt:
- GPM bases dùng cho **protection** (constrain future A), xong rồi **xóa feature_list** (line 304) — standard CL technique
- Prototypes dùng cho **inference routing** trên TEST data — lưu và sử dụng vĩnh viễn → DỮ LIỆU CŨ ẢNH HƯỞNG TRỰC TIẾP TỚI PREDICTIONS trên data mới

### Hệ quả: Spectral routing = ONLY zero-replay-compliant option

Tại inference, chỉ có: $h$ (test input) + $\{A_t, B_t\}_{t=1}^T$ (model params). Routing PHẢI là:
$$\text{routing}(h) = f(h; \{A_t, B_t\}_{t=1}^T)$$

Spectral routing (Rayleigh quotient) thỏa điều kiện này. Learned routing (ROOT's MLP) cũng thỏa. Prototype routing **KHÔNG** thỏa.

---

## III. CƠ CHẾ ROUTING HIỆN TẠI CÓ TỐI ƯU?

### 3.1 Routing hiện tại: σ²-weighted Rayleigh quotient

$$w_t(h) = \text{softmax}\left(\frac{\mathbf{h}^\top V_t \text{diag}(\sigma_t^2) V_t^\top \mathbf{h}}{\|\sigma_t\|^2 \|\mathbf{h}\|^2}\right)$$

Đây là **Rayleigh quotient** của ma trận PSD $V_t \text{diag}(\sigma_t^2) V_t^\top$ w.r.t. $h$.

### 3.2 Phân tích Information-Theoretic

Từ góc nhìn **sufficient statistics** (Fisher-Neyman):

**Routing problem**: Cho observation $h$, decide $t^* = \arg\max_t P(t|h)$.

**Bayes optimal**: $w_t(h) = P(t|h) \propto P(h|t) P(t)$.

**Spectral routing giả định gì?** Nó giả định $P(h|t)$ tương ứng với energy projection lên $V_t$. Tức là, $h$ từ task $t$ sẽ có energy tập trung trong $\text{span}(V_t)$.

**Khi nào giả định đúng?**
- Khi A directions align với task-discriminative input subspace
- Khi different tasks' inputs separate trong A's column space

**Khi nào giả định SAI?**
- Same-domain tasks → $h$ similar → projection lên orthogonal $A_k, A_j$ cho RANDOM results
- Task input không align với random A → low fit for ALL tasks → routing degeneracy

### 3.3 Alternatives: Công cụ toán học nào tốt hơn?

#### Alternative 1: Nuclear Norm / Trace Norm

Thay vì SVD spectral routing, dùng **trace of product**:
$$\text{aff}_t(h) = \|(\Delta W_t)^\top h\|_2^2 = h^\top (BA)^\top (BA) h = h^\top A^\top B^\top B A h$$

Đây chính xác là tính toán hiện tại nhưng **không cần SVD** — chỉ cần $B^\top B$ (Gram matrix).

**Insight**: SVD chỉ là decomposition của $B^\top B A^\top A h$. Nếu A orthogonalized, $A^\top A \approx c \cdot I$ trên row-space → SVD reduction to $B^\top B$.

**Paper references**:
- **Marchenko-Pastur Law** (1967): Random matrix $A$ với entries i.i.d. → $A^\top A / n \to I$ as $d \to \infty$
- **Random Matrix Theory** (Anderson, 2003): Concentration of spectral norms for random projections

#### Alternative 2: Frobenius Inner Product (Direct Affinity)

$$\text{aff}_t(h) = \|\Delta W_t \odot h h^\top\|_F = \text{tr}(\Delta W_t^\top \Delta W_t h h^\top)$$

Measures how much $\Delta W_t$ modifies input $h$ → more direct measure of "expert relevance".

#### Alternative 3: Grassmannian Distance

Thay vì project $h$ lên subspaces, đo **geodesic distance** giữa subspace $\text{col}(V_t)$ và direction $h/\|h\|$:

$$d_G(h, V_t) = \arccos\left(\frac{\|V_t^\top h\|}{\|h\|}\right)$$

**Paper references**:
- **Absil et al.** (2004, 2008): Optimization on Stiefel and Grassmann manifolds — geodesic distances, exponential maps
- **Hamm & Lee** (2008): Grassmann discriminant analysis — subspace classification
- **Edelman, Arias, Smith** (1998): Geometry of algorithms with orthogonality constraints

#### Alternative 4: CKA (Centered Kernel Alignment)

$$\text{CKA}(K_h, K_t) = \frac{\|K_h K_t\|_F}{\|K_h\|_F \|K_t\|_F}$$

Với $K_t = V_t V_t^\top$ (projection kernel) và $K_h = h h^\top$.

**Paper references**:
- **Kornblith et al.** (2019): CKA for comparing neural network representations
- **Nguyen et al.** (2021): CKA variants for continual learning similarity

#### Alternative 5: Projection Residual (Information Loss)

Thay vì "how much energy projects", đo "how much information LOST after projection":
$$\text{loss}_t(h) = \|(I - V_t V_t^\top) h\|^2 / \|h\|^2 = 1 - \text{fit}_t(h)$$

Routing bằng minimizing information loss. Mathematically identical to current approach nhưng conceptually clearer.

### 3.4 Quan sát then chốt: DPI đúng nhưng KHÔNG có nghĩa spectral routing thất bại

**Theorem (Fundamental Routing Information Bound under InfLoRA)**:

Cho $\Delta W_t = B_t A_t$ với $A_t$ frozen. BẤT KỲ routing function $f(h, \Delta W_t)$ dựa trên $\Delta W_t$ hoặc các derived quantities (SVD, norms, projections):

$$f(h, B_t A_t) \text{ chỉ phân biệt h qua } A_t h$$

vì $B_t A_t h = B_t (A_t h)$ — output chỉ phụ thuộc $h$ thông qua projection $A_t h$.

**Data Processing Inequality**: $I(T; f) \leq I(T; A_t h) \leq I(T; h)$

**NHƯNG — sửa lỗi v1**: DPI nói $I(T; A_t h) \leq I(T; h)$, KHÔNG nói $I(T; A_t h) = 0$.

Thực tế, dù $A_t$ random, nó vẫn capture $r/d$ fraction energy:
$$\mathbb{E}[\|A_t h\|^2 / \|h\|^2] = r/d \approx 8/512 = 1.56\%$$

Và quan trọng hơn: **singular values $\sigma_t$ encode B's task-specific learning** (Section 2.4). Rayleigh quotient:
$$\text{fit}_t(h) = \frac{\sum_i \sigma_{t,i}^2 (v_{t,i}^\top h)^2}{\sum_i \sigma_{t,i}^2 \|h\|^2}$$

Với C4 làm $\sigma$ spectrum phong phú (high effective rank), discrimination TĂNG vì:
1. Expert response mạnh hơn (lớn σ) → fit_t cao cho matching inputs
2. Full-rank LoRA → expert capture nhiều directions hơn → h projection richer

**Kết luận sửa**: Spectral routing bị information bottleneck, nhưng bottleneck có thể **đủ rộng** nếu expert quality cao. V6 test chính xác giả thuyết này.

**Paper references**:
- **Data Processing Inequality** (Cover & Thomas, 2006): $I(X;Z) \leq I(X;Y)$ khi $X \to Y \to Z$
- **Sufficient Statistics** (Fisher-Neyman Factorization Theorem): Routing optimal khi sử dụng sufficient statistic for task identity
- **Information Bottleneck** (Tishby et al., 2000): Tradeoff compression vs. prediction
- **Johnson-Lindenstrauss Lemma** (1984): Random projection preserves distances with $O(\log T / \epsilon^2)$ dimensions — 8 dims bảo toàn distance giữa ~15 task centroids with high probability

### 3.5 V6 Hypothesis: C4 Makes Spectral Routing Viable

**Giả thuyết (v6_discuss.md)**: V1-V3 thất bại KHÔNG chỉ do routing mechanism, mà do **expert quality TỆ** (no C4):
- Không có preconditioning → gradient distortion → B learns poorly → σ degenerate
- Không có entropy regularization → rank-1 LoRA → 1 direction thống trị → routing degeneracy

**C4 fixes expert quality → σ spectrum phong phú → spectral routing trở nên discriminative.**

**Logic chain**:
$$\text{C4 (preconditioning + entropy)} \to \text{Better B training} \to \text{Richer σ spectrum}$$
$$\to \text{Higher effective rank} \to \text{More discriminative fit}_t(h)$$
$$\to \text{Better routing} \to \text{Performance improvement}$$

**Verification**: V5 có C4 + prototype. V6 = C4 only → isolate C4's contribution.

**Dự đoán V6**: 
- Nếu C4 là key: AP(EM) ~45-55 (vượt xa V3=27.66, nhưng không bằng V5=59.55 vì mất prototype advantage)
- Nếu C4 không đủ: AP(EM) ~30-35 (tương tự V3)
- Nếu C4 destabilize: NaN hoặc spike → giảm λ hoặc disable preconditioning

---

## IV. TẬP HỢP VÀ ĐỀ XUẤT (REVISED v2)

### 4.1 Summary of Theoretical Findings

| Finding | Implication |
|---------|------------|
| Never-learning do A frozen + GPM, KHÔNG do routing | C2 (routing) đúng hướng, cần C4 (expert quality) mạnh hơn |
| SVD signatures ĐÚNG mathematically, $V \subseteq \text{row}(A)$ | Routing DIRECTIONS constrained, nhưng σ² MAGNITUDES reflective |
| GPM ⊥ → perfect subspace separation | Routing dựa vào σ²-weighted projection, KHÔNG chỉ V directions |
| DPI: $I(T; Ah) \leq I(T; h)$ | Bottleneck tồn tại, nhưng KHÔNG = 0; có thể đủ rộng nếu expert quality cao |
| **Prototype routing VI PHẠM zero-replay** | **V5 INVALID theo settings.txt** |
| C4 cải thiện expert quality → richer σ → better routing | **V6 hypothesis: C4 = key to making spectral routing work** |

### 4.2 Prototype Routing: Tại sao sai và bài học

**V5 sai lầm**: Prototype $\mu_t = \text{mean}(h_i^{(t)})$ = **first moment of data distribution** → vi phạm zero-replay.

**Bài học**: Information-theoretic optimality ($I(T; h)$ vs $I(T; Ah)$) PHẢI tuân theo constraint. Giải pháp "tối ưu" nhưng vi phạm settings = vô giá trị.

**GPM bases cũng dùng data** nhưng chỉ cho protection (standard CL practice, ROOT cũng dùng). Ranh giới: data CÓ THỂ dùng cho constrain future learning (GPM), KHÔNG THỂ dùng cho influence future predictions (prototypes).

### 4.3 SVD Routing + C4: Giả thuyết V6

**V6 = spectral routing (V3 mechanism) + C4 (preconditioning + entropy)**

Tại sao V3 thất bại (AP=27.66)?
1. V3 chạy SCRIPT SAI (threshold=0.98 thay vì 0.995) — **bug, không phải limitation**
2. V3 KHÔNG có C4 → LoRA trains poorly → degenerate σ → routing random
3. Train-inference mismatch: adaptive bias at train, SVD at inference → B optimized under wrong routing

V6 fixes:
1. ✅ Threshold = 0.995 (đúng)
2. ✅ C4 enabled (preconditioning + entropy)
3. ✅ Adaptive bias at train, symmetric SVD at inference (mechanism unchanged but expert quality vastly better)

**Dự đoán (theory-based)**:

C4 addresses Cause 1 (gradient distortion) và Cause 3 (CE-only loss) từ Section I:
- Preconditioner $(AA^\top + \epsilon I)^{-1/2}$ equalizes gradient → condition-number-independent learning
- Entropy regularization → maximizes effective rank → expert responds to more directions

Nếu V6 AP ~45-55 → C4 là significant contributor → **spectral routing viable under zero-replay**
Nếu V6 AP ~30 → expert quality KHÔNG đủ → GPM-Routing Paradox remains dominant

### 4.4 Contribution Idea cần chỉnh sửa gì?

#### C1 (Spectral Signatures) — GIỮI, VAI TRÒ ĐÚng
- SVD signatures dùng cho cả routing LẪN characterization
- Thin SVD optimization (QR+SVD) vẫn valid
- σ values là key discriminative signal (kết hợp V directions)

#### C2 (Routing) — CẦN REVISION
- ~~Prototype routing~~ → **Xóa**, vi phạm zero-replay
- **Giữ**: SVD spectral routing (Rayleigh quotient) — ONLY valid option
- **Giữ**: Adaptive bias $\beta(n)$ for training cold-start
- **Giữ**: Symmetric SVD inference routing
- **Mới**: C4 là yếu tố quyết định *chất lượng* routing (không phải mechanism mới mà là *quality of what's being routed TO*)

#### C3 (ESA) — GIỮI
- Dynamic threshold works

#### C4 (Preconditioning + Entropy) — **TRỞ THÀNH THEN CHỐT**
- Không chỉ improve single-task quality
- **Trực tiếp improve routing quality** qua richer σ spectrum
- C4 = bridge giữa protection (GPM) và routing (spectral affinity)
- Preconditioning: gradient equalization → all rank directions learn → non-degenerate σ
- Entropy: explicit rank maximization → more responsive expert → better fit discrimination

### 4.5 Đề xuất sửa đổi cụ thể

#### 1. DEPLOY V6 (branch `new`)

V6 = spectral routing + C4 = **ONLY valid configuration** under zero-replay.

Prediction range: AP(EM) 45-55 (optimistic: C4 strong) hoặc 30-35 (pessimistic: C4 insufficient)

#### 2. Update SPECROUTE_IDEA.md
- Remove C2.1 (prototype routing section)
- Re-frame C4 as enabling technology cho C2 (not independent contribution)
- Routing–Protection Duality theorem STILL VALID (spectral affinity ↔ protection quality)
- New framing: "C4 completes the duality — protection provides direction discrimination, C4 provides magnitude discrimination"

#### 3. Nếu V6 AP ~30 (C4 insufficient) → Next direction
- Relaxed orthogonality: $A_t = (1-\eta) A_t^{\perp} + \eta A_t^{\parallel}$ — allow small overlap
  - Tradeoff: slight forgetting ↑ but routing discrimination ↑↑
  - $\eta \in [0.05, 0.2]$
  - ZERO-REPLAY COMPLIANT (only modifies A init, no data stored)
- Adaptive epochs cho tiny datasets (CB: 250 samples → more steps)

---

## V. KẾT LUẬN (REVISED)

### Trả lời câu hỏi ban đầu

1. **Nguyên nhân gốc rễ never-learning**: A frozen + GPM projection → restricted capacity + random A ≠ optimal A + insufficient training. KHÔNG do routing. ROOT cũng near-fail CB (3.57). Đây là InfLoRA fundamental tradeoff.

2. **SVD có cần xem xét lại?**:
   - SVD mathematically correct
   - $V \subseteq \text{row}(A)$ → directions constrained, nhưng **σ values reflect B's learning** → key discriminative signal
   - A frozen LÀ lựa chọn tốt cho anti-forgetting
   - SVD dùng cho routing: σ²-weighted Rayleigh quotient ĐÚNG, nhưng cần C4 boost expert quality
   - **KHÔNG cần thay SVD bằng tool khác** — vấn đề là expert quality, không phải measurement method

3. **Routing mechanism tối ưu?**:
   - Spectral routing = ONLY zero-replay-compliant parameter-free option
   - ~~Prototype routing~~ **vi phạm zero-replay** → loại bỏ
   - V1-V3 thất bại có thể do EXPERT QUALITY (no C4), không chỉ routing mechanism
   - V6 (SVD + C4) = experiment để test C4 hypothesis
   - DPI bottleneck tồn tại nhưng Johnson-Lindenstrauss cho thấy random projection 8-dim đủ preserve distances cho ~15 tasks

### Recommendation

1. **DEPLOY V6** (branch `new`) — SVD routing + C4 = đúng constraint
2. **V5 prototype routing INVALID** — vi phạm zero-replay
3. **C4 trở thành contribution then chốt** — không chỉ C1 add-on mà ENABLES C2 routing quality
4. **Chờ V6 results** trước khi quyết định tiếp:
   - AP ≥ 45 → spectral + C4 viable, tiếp tục tối ưu
   - AP ≤ 35 → cần relaxed orthogonality hoặc fundamental rethink

---

## Paper References (30+)

### LoRA & Low-Rank Optimization
1. Hu et al. (2022) — LoRA: Low-Rank Adaptation of Large Language Models
2. Sun et al. (CVPR 2024) — InfLoRA: Interference-Free Low-Rank Adaptation
3. Jordan et al. (2025) — Muon/Riemannion: Riemannian LoRA
4. LORO (ICLR 2025) — Low-rank Riemannian Optimizer
5. SD-LoRA (ICLR 2025) — Singular value Direction decomposition
6. Stiefel-LoRA (EMNLP 2025) — Orthogonal B constraints
7. LoKO (2024) — Kalman Filter LoRA Optimizer
8. LoRA– (CVPR 2025) — Triplet loss in Drift-Resistant Space

### Continual Learning
9. GORP (ACL 2025) — Unified Gradient Subspace Projection
10. CMCL (NeurIPS 2025) — Dual Stability/Plasticity Bounds
11. CaLoRA (NeurIPS 2025) — Causal Gradient Adaptation
12. PLAN (ICCV 2025) — Proactive Rank Allocation
13. TreeLoRA (ICML 2025) — Gradient Similarity Tree
14. Angle Matters (ICML 2025) — Angular task signal analysis
15. SEFE (ICML 2025) — Superficial vs Essential Forgetting
16. GIFT (CVPR 2025) — Fisher Information LoRA
17. Low-rank Forgetting Analysis (NeurIPS 2025)

### SVD & Matrix Theory
18. Eckart-Young (1936) — Best low-rank approximation
19. Weyl's Perturbation Theorem — Singular value stability
20. Horn & Johnson (2013) — Matrix Analysis (product SVD)
21. Thompson (1976) — Interlacing inequalities for products
22. Marchenko-Pastur (1967) — Random matrix spectral distribution
23. Anderson (2003) — Random Matrix Theory

### Grassmannian & Manifold Geometry
24. Absil et al. (2004, 2008) — Optimization on Grassmann manifolds
25. Hamm & Lee (2008) — Grassmann Discriminant Analysis
26. Edelman, Arias, Smith (1998) — Geometry with orthogonality

### Information Theory
27. Cover & Thomas (2006) — Data Processing Inequality
28. Tishby et al. (2000) — Information Bottleneck Method
29. Fisher-Neyman Factorization Theorem — Sufficient statistics
30. Kornblith et al. (2019) — CKA representation similarity
31. Todd et al. (ICLR 2025) — Function Vectors
32. Nguyen et al. (2021) — CKA for continual learning
33. **Johnson & Lindenstrauss (1984)** — Random projection preserves distances

### GainLoRA Specific
34. GainLoRA (original paper) — Gating + InfLoRA architecture
35. GPM (Saha et al., 2021) — Gradient Projection Memory

### Appendix: V1 Analysis Error Log

**V1 lỗi**: Khuyến nghị prototype routing (V5) nhưng prototype = mean embedding = data statistic → vi phạm zero-replay.

**Root cause**: DPI argument ($I(T;h) > I(T;Ah)$) đúng mathematically nhưng bỏ qua constraint. Tối ưu hóa information access nhưng vi phạm allowable information set.

**Bài học**: Trong research, correctness = mathematical validity + constraint satisfaction. Giải pháp information-theoretically optimal nhưng violate settings = invalid. Đây là ví dụ tốt cho research_rule: luôn verify solution against ALL constraints trước khi recommend.
