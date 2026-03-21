# SpecRoute: Định tuyến Phổ thông qua Duality Định tuyến–Bảo vệ trong Học liên tục với LoRA

> **Tài liệu thiết kế chính thức — V7**
> Ràng buộc: Zero-replay nghiêm ngặt. Theory-first.

---

## 1. Đặt bài toán

**Setting.** Học liên tục với LoRA mở rộng trên một LLM đóng băng.
Các tasks $\mathcal{T}_1, \ldots, \mathcal{T}_T$ đến tuần tự. Với mỗi task $t$:

- Một adapter low-rank $\Delta W_t = B_t A_t$ ($A_t \in \mathbb{R}^{r \times d}$, $B_t \in \mathbb{R}^{d \times r}$) được thêm vào mọi phép chiếu attention.
- Chỉ $B_t$ được huấn luyện; $A_t$ bị đóng băng sau khi khởi tạo null-space (InfLoRA).
- Sau khi huấn luyện, cả $A_t, B_t$ đều bị đóng băng và một nhánh mới được tạo cho task tiếp theo.

**Inference.** Cho input $x$ *không có* task identifier, mô hình phải tạo ra output đúng:

$$y = f\!\Bigl(W_0\, x \;+\; \sum_{t=1}^{T} w_t(x)\; B_t A_t\, x\Bigr)$$

**Ba bài toán con kết hợp:**

| Bài toán con | Mục tiêu | Yêu cầu hình thức |
|:------------:|---------|-------------------|
| **Routing (R)** | Gán input đúng expert | $w_{t^*}(x) \gg w_t(x)$ với $t \neq t^*$ |
| **Protection (P)** | Ngăn suy giảm expert cũ | $\Delta W_t$ không đổi sau task $t$ |
| **Allocation (A)** | Quản lý capacity không gian con hữu hạn | $\sum_t \dim\bigl(\mathrm{span}(A_t)\bigr) \leq d$ |

**Ràng buộc setting:** *Zero-replay* — không tái sử dụng dữ liệu task cũ dưới bất kỳ hình thức nào (thô, synthetic, hay phân phối thống kê).

---

## 2. Quan sát: Duality Ẩn

### 2.1 Tiếp cận của GainLoRA và Điểm yếu

GainLoRA (NeurIPS 2025) xử lý R, P, A như các bài toán **độc lập**:

| Khía cạnh | Cơ chế | Chi phí |
|-----------|--------|---------|
| R | MLP `trans_input` học được + `prompt_key` học được → cosine gating | Tham số thêm + subspace GPM |
| P | GPM chiếu gradient vào null-space của task cũ | Tiêu thụ subspace mỗi task |
| A | Threshold tăng dần $\varepsilon_t \nearrow 1$ | Task sau bị ràng buộc hơn |

**Điểm yếu cơ bản:** Vì routing được *học*, nó tạo ra vòng lặp xấu:

1. `trans_input` thay đổi mỗi task → routing space trôi dạt → prompt keys cũ mất alignment → routing suy giảm.
2. GPM phải bảo vệ routing params → *tiêu thụ subspace có thể dùng cho task learning*.
3. KL distillation trên routing cần thiết → yêu cầu replay hoặc frozen copies → overhead bộ nhớ.

### 2.2 Nhận thức then chốt

Chúng tôi quan sát rằng GPM đảm bảo xấp xỉ các subspace input của expert trực giao nhau:

$$\mathrm{span}(V_i) \;\approx\perp\; \mathrm{span}(V_j), \qquad i \neq j$$

trong đó $V_t$ là các right singular vectors của $\Delta W_t$. Tính trực giao này, được đảm bảo cho mục đích **bảo vệ**, đồng thời cung cấp tiêu chí **routing** tự nhiên: vì các subspace không chồng chéo, đo lường mức độ một input căn chỉnh với từng subspace sẽ xác định duy nhất task gốc.

> **Duality Định tuyến–Bảo vệ.**
> Chống quên (bảo vệ subspace trực giao) và nhận diện task (routing phân biệt)
> là *hai biểu hiện kép của cùng một cấu trúc phổ*.
> Giải quyết một bài toán tự động giải quyết bài toán kia.

**Hệ quả:**
- Không cần tham số routing học được → không trôi dạt routing, không tốn GPM cho routing.
- Không cần replay để duy trì routing → tự nhiên tuân thủ zero-replay.
- Độ chính xác routing được *đảm bảo* bởi chất lượng bảo vệ (được hình thức hoá bên dưới).

---

## 3. Khung Lý thuyết

### 3.1 Spectral Expert Signatures

**Định nghĩa 1** *(Spectral Signature).* Với expert đóng băng $\Delta W_t = B_t A_t$ và thin SVD

$$\Delta W_t = U_t\, \Sigma_t\, V_t^\top, \qquad V_t \in \mathbb{R}^{d \times r},\; \Sigma_t = \mathrm{diag}(\sigma_{t,1}, \ldots, \sigma_{t,r}),$$

spectral signature là $\mathcal{S}_t = (V_t,\, \boldsymbol{\sigma}_t)$ trong đó:

- $V_t$: **input receptive field** — $r$ hướng input mà expert xử lý,
- $\boldsymbol{\sigma}_t$: **sensitivity spectrum** — hệ số khuếch đại biến đổi dọc mỗi hướng.

**Góc nhìn lý thuyết thông tin.** Xem $\Delta W_t$ như một kênh tuyến tính, cột của $V_t$ là *input modes* của kênh và $\sigma_{t,i}^2$ là *gain* của mode $i$. Tổng channel capacity (Frobenius energy) là $\|\Delta W_t\|_F^2 = \sum_i \sigma_{t,i}^2$.

### 3.2 Spectral Affinity

**Định nghĩa 2** *(Spectral Affinity).* Độ tương hợp của input $h \in \mathbb{R}^d$ với expert $t$:

$$\alpha_t(h) \;=\; \frac{h^\top M_t\, h}{\mathrm{tr}(M_t)\;\|h\|^2}, \qquad M_t = V_t\, \mathrm{diag}(\boldsymbol{\sigma}_t^2)\, V_t^\top$$

Khai triển:

$$\alpha_t(h) = \frac{\displaystyle\sum_{i=1}^{r} \sigma_{t,i}^2\;(v_{t,i}^\top h)^2}{\displaystyle\Bigl(\sum_{i=1}^{r} \sigma_{t,i}^2\Bigr)\,\|h\|^2}$$

**Tính chất:**

| Tính chất | Phát biểu |
|-----------|-----------|
| Dải giá trị | $\alpha_t(h) \in [0,\, 1]$ — weighted Rayleigh quotient chuẩn hoá |
| Energy ratio | $\alpha_t(h) = \|\Delta W_t\, h\|^2 \;/\; \bigl(\|\Delta W_t\|_F^2\, \|h\|^2\bigr)$ |
| Ý nghĩa | Phần channel capacity của expert $t$ được kích hoạt bởi $h$ |
| In-distribution | $h \in \mathrm{span}(V_t) \;\Rightarrow\; \alpha_t(h) \geq \kappa_{\min}(t) > 0$ |
| Out-of-distribution | $h \perp \mathrm{span}(V_t) \;\Rightarrow\; \alpha_t(h) = 0$ chính xác |

### 3.3 Định lý Duality Định tuyến–Bảo vệ

**Định nghĩa 3** *(Subspace Overlap).* Độ chồng chéo giữa các expert $i$ và $j$:

$$\delta_{ij} = \|V_i^\top V_j\|_F^2 = \sum_{k=1}^{r} \cos^2 \theta_{ij}^{(k)}$$

trong đó $\theta_{ij}^{(k)}$ là các *principal angles* giữa $\mathrm{span}(V_i)$ và $\mathrm{span}(V_j)$.

---

**Định lý 1** *(Duality Định tuyến–Bảo vệ).* Nếu GPM đảm bảo $\delta_{ij} \leq \varepsilon$ với mọi $i \neq j$, thì với mọi unit input $h \in \mathrm{span}(V_{t^*})$, **routing margin** thoả mãn:

$$\boxed{\;\alpha_{t^*}(h) \;-\; \max_{t \neq t^*}\, \alpha_t(h) \;\;\geq\;\; \kappa_{\min}(t^*)\; -\; \varepsilon\, \kappa_{\max}\;}$$

trong đó:

$$\kappa_{\min}(t) = \frac{\sigma_{t,\min}^2}{\sum_i \sigma_{t,i}^2}, \qquad \kappa_{\max} = \max_t\, \frac{\sigma_{t,\max}^2}{\sum_i \sigma_{t,i}^2}$$

**Chứng minh.**

*Cận dưới cho expert đúng.* Viết $h = V_{t^*}\, c$ với $\|c\| = 1$. Khi đó $(v_{t^*,i}^\top h)^2 = c_i^2$ và $\sum c_i^2 = 1$:

$$\alpha_{t^*}(h) = \frac{\sum_i \sigma_{t^*,i}^2\, c_i^2}{\sum_i \sigma_{t^*,i}^2} \;\geq\; \kappa_{\min}(t^*)$$

*Cận trên cho expert sai.* Với $t \neq t^*$:

$$\|V_t^\top h\|^2 \leq \delta_{t,t^*} \leq \varepsilon \;\;\Rightarrow\;\; \alpha_t(h) \leq \kappa_{\max}\, \varepsilon \qquad\square$$

---

**Hệ quả 1** *(Routing Confidence).* Với softmax routing nhiệt độ $\tau$:

$$w_{t^*}(h) \;\geq\; \frac{1}{1 + (T{-}1)\,\exp\!\bigl(-m/\tau\bigr)}, \qquad m = \kappa_{\min}(t^*) - \varepsilon\, \kappa_{\max}$$

Để đạt confidence mục tiêu $w_{t^*} \geq 1 - \delta$, đặt $\tau \leq m \,/\, \ln\!\bigl(\tfrac{T-1}{\delta}\bigr)$.

---

**Hệ quả 2** *(Capacity Bound — Kết nối Grassmannian).* Số lượng tối đa subspace $r$-chiều trong $\mathbb{R}^d$ với pairwise overlap $\delta \leq \varepsilon$:

$$T_{\max} \;\leq\; \frac{d}{r\,(1 - \varepsilon)}$$

Với T5-Small ($d = 512$, $r = 8$, $\varepsilon = 0.02$): $T_{\max} \leq 65 \gg 15$ tasks. Điều này kết nối capacity học liên tục với lý thuyết Grassmannian packing.

### 3.4 Drift Invariance

**Mệnh đề 1** *(Drift-Free Routing).* Hàm routing $h \mapsto \alpha_t(h)$ hoàn toàn ổn định qua tất cả các task.

**Chứng minh.** Routing input được tính từ frozen embedding table, *trước* bất kỳ transformer block nào. LoRA chỉ tồn tại trong các attention layer sâu hơn → $h$ độc lập với mọi tham số LoRA. Kết hợp với $\mathcal{S}_t$ đóng băng, $\alpha_t(h)$ bất biến với mọi thay đổi tích luỹ. $\square$

### 3.5 Vấn đề then chốt: Null-Space Collapse

Định lý 1 giả định $h \in \mathrm{span}(V_{t^*})$. Trong thực tế điều này đòi hỏi hai điều kiện:

**(A) Expert phải học được:** $A_t$ phải nằm trong subspace có liên quan đến task $t$ để $B_t$ có thể học các biến đổi có ý nghĩa, tạo ra $\sigma_{t,i} > 0$ đáng kể.

**(B) Input phải có projection:** Inputs thực tế của task $t$ phải chiếu có năng lượng lên $\text{span}(V_t)$.

**InfLoRA gốc vi phạm (A):** $A_t$ được khởi tạo ngẫu nhiên (Kaiming) rồi chiếu vào null-space của GPM:

$$A_t \leftarrow A_t - A_t P_{\text{old}}, \quad \text{normalize}$$

Null-space (sau $t-1$ tasks) là một không gian $d - N_{\text{protected}}$ chiều. Kaiming random trong không gian này KHÔNG ĐẢM BẢO alignment với các hướng có liên quan đến task $t$. Khi null-space thu hẹp dần (Layer 7: 8/512 → 161/512 → 344/512 qua 13 tasks), xác suất random init bắt được đúng hướng task-relevant giảm theo.

**Hệ quả thực nghiệm (V6):** IMDB (task 8) — eval_loss dừng ở 6.37 sau 10 epoch, EM=0.0 suốt quá trình, expert thực sự không thể học bất cứ điều gì hữu ích.

---

## 4. Các Thành phần Framework

### C1 — Spectral Expert Signatures

Sau khi train task $t$, tính $\mathcal{S}_t = (V_t, \boldsymbol{\sigma}_t)$ qua **thin SVD**:

$$B_t,\, A_t \;\xrightarrow[\text{QR + SVD}]{O(dr^2)}\; (V_t,\, \boldsymbol{\sigma}_t)$$

- QR decomposition của $B$ và $A^\top$, sau đó SVD của $r \times r$ core → chính xác, $O(dr^2)$ so với $O(d^2 r)$.
- Lưu trên mỗi LoRA layer (encoder Q, V; decoder self/cross Q, V).
- **Bất biến** theo cấu trúc: weights đóng băng → signatures đóng băng → không trôi dạt.

### C2 — Spectral Affinity Routing

**Inference** (tất cả task, routing SVD đối xứng):

$$w(h) = \mathrm{softmax}\!\left(\frac{[\alpha_1(h),\; \ldots,\; \alpha_T(h)]}{\tau}\right)$$

Mọi task đều dùng cùng công thức $\sigma^2$-weighted spectral affinity (Định nghĩa 2). Sau khi train task $t$, tính $\mathcal{S}_t$ một lần qua `prepare_inference_routing()` và sử dụng cùng signatures của task cũ.

**Training** (task $t$, SVD cuối chưa biết vì $B_t$ đang train):

$$\alpha_t^{\mathrm{train}}(h) = \frac{\|A_t\, h\|^2}{r\,\|h\|^2} + \beta(n), \qquad \beta(n) = \tau \cdot \ln\!\left(\frac{\alpha_{\mathrm{target}} \cdot n}{1 - \alpha_{\mathrm{target}}}\right)$$

trong đó $n = |\{\text{task cũ}\}|$ và $\alpha_{\mathrm{target}} \in (0,1)$ là routing weight mục tiêu cho task hiện tại (mặc định 0.8).

**Lý giải proxy A-row:** Với $B_t$ full-rank bất kỳ, column span của $V_t$ (từ SVD của $B_t A_t$) bằng $\mathrm{range}(A_t^\top)$. Vì vậy các hàng $A$ span *cùng* subspace input mà $V_t$ hội tụ sẽ capture. Proxy đo alignment input với subspace này dùng weighting đồng đều (chưa có $\sigma$).

**Lý giải adaptive $\beta(n)$:** Bias hằng số gây routing weight của task hiện tại giảm $O(1/n)$ theo số task (softmax dilution). Công thức adaptive chuẩn hoá điều này: giải $w_t = \alpha_{\mathrm{target}}$ trong phương trình softmax cho ra closed-form trên. Đảm bảo task hiện tại nhận được routing weight $\approx \alpha_{\mathrm{target}}$ bất kể $n$.

**Lý giải symmetric inference:** Tại inference, $B_t$ đóng băng và $\Delta W_t = B_t A_t$ có SVD xác định. Dùng cùng $\sigma^2$-weighted Rayleigh quotient cho tất cả task đảm bảo *đối xứng đo lường* — mọi affinity đều sống trên cùng metric space, và Định lý 1 áp dụng thống nhất.

| Phase | Cơ chế routing | Nhiệt độ | Lý do |
|-------|----------------|----------|-------|
| Training (task $t$) | A-row fit + adaptive $\beta(n)$ | $\tau = 1.0$ | $B=0$ cold-start; β bù softmax dilution |
| Inference (mọi task) | SVD spectral affinity (đối xứng) | $\tau = 1.0$ | Mọi task dùng σ²-weighted Rayleigh quotient |

### C3 — Capacity-Aware Subspace Allocation

GPM threshold kiểm soát đánh đổi bảo vệ–capacity. Từ Định lý 1:
- $\varepsilon$ thấp hơn → bảo vệ & routing tốt hơn, nhưng null-space cạn nhanh hơn.
- $\varepsilon$ cao hơn → nhiều capacity hơn, nhưng đảm bảo routing yếu hơn.

**Dynamic threshold** (theo InfLoRA):

$$\varepsilon_t = (1 - \varepsilon_0) \cdot \frac{t}{T} + \varepsilon_0$$

trong đó $\varepsilon_0$ là base threshold. Phân bổ bảo vệ nghiêm ngặt dần khi task tích luỹ. Đánh đổi là *có nguyên tắc* qua Hệ quả 2: miễn là $\varepsilon_t$ vượt $(1 - d/(rT))$, capacity cho tất cả $T$ task được đảm bảo.

---

### C4 — Spectrally-Conditioned Gradient (Implementation Detail)

> **Lưu ý phân loại:** C4 là chi tiết triển khai, không phải đóng góp lý thuyết độc lập. Nó giải quyết một vấn đề kỹ thuật thuần túy: sau khi `get_reg_matrix()` chiếu $A_t$ vào null-space, column space của $A_t$ không còn trực giao, khiến gradient $\nabla_B \mathcal{L} = \nabla_{\Delta W} \mathcal{L} \cdot A^T$ bị biến dạng. Việc áp dụng preconditioner là một hiệu chỉnh kỹ thuật cần thiết, không phải một luận điểm học thuật mới.

Gradient $\nabla_B \mathcal{L}$ bị biến dạng bởi condition number của $A^T$. Chúng tôi áp dụng preconditioner một lần sau khi $A$ đóng băng:

$$\tilde{\nabla}_B = \nabla_B \mathcal{L} \cdot (AA^T + \epsilon I)^{-1/2}$$

Preconditioner được tính **một lần** sau `get_reg_matrix()` — không có overhead per-step.

> **Lưu ý:** Spectral entropy regularization (C4.2) được loại bỏ khỏi V7. Lý do: C5 (Data-Informed Init) khởi tạo $A_t$ sao cho $B_t$ học trong subspace task-relevant → singular values tự nhiên sẽ phân tán theo dữ liệu. Cưỡng bức entropy uniform qua regularization mâu thuẫn với triết lý của C5 (để dữ liệu dẫn dắt phân phối phổ, không phải regularizer). Preconditioner gradient (C4.1) vẫn giữ vì nó sửa điều kiện ma trận, không ảnh hưởng đến triết lý data-driven.

---

### C5 — Data-Informed Subspace Initialization (Đóng góp chính)

#### Động lực

Khi $A_t$ được khởi tạo ngẫu nhiên và chiếu vào null-space, nó chiếm một điểm *tùy ý* trên restricted Grassmannian $\mathrm{Gr}(r,\, d - N_{\text{protected}})$. Với $\dim\bigl(\mathrm{Gr}(8, 351)\bigr) = 8 \times 343 = 2744$, không gian lựa chọn rất lớn — random init gần như chắc chắn sub-optimal. Đặc biệt khi null-space thu hẹp, các hướng task-relevant ngày càng chiếm tỷ lệ nhỏ trong không gian còn lại, làm cho random init càng kém hiệu quả.

#### Bài toán tối ưu

Chúng tôi đặt vấn đề khởi tạo $A_t$ như bài toán tối ưu có ràng buộc:

$$\max_{A_t} \quad \text{tr}\!\bigl(A_t\, Q\, C_t\, Q\, A_t^T\bigr) \quad \text{s.t.} \quad A_t A_t^T = I_r$$

trong đó $Q = I - P_{\text{old}}$ là null-space projector (với $P_{\text{old}} = \mathcal{B}\mathcal{B}^T$ là GPM projection matrix), và:

$$C_t = \frac{1}{|\mathcal{X}_t|} \sum_{x \in \mathcal{X}_t} h(x)\, h(x)^T$$

là activation covariance của task $t$ được ước tính từ vài batch đầu của dữ liệu training.

**Ý nghĩa:** Maximize variance captured trong null-space theo phân phối dữ liệu task $t$ — tức là tìm subspace $r$-chiều trong null-space *phù hợp nhất* với dữ liệu task hiện tại.

#### Lời giải dạng đóng

Định nghĩa **projected covariance**: $\tilde{C}_t = Q\, C_t\, Q$.

Bài toán trở thành constrained PCA tiêu chuẩn trên $\tilde{C}_t$. Lời giải chính xác là:

$$A_t = \text{top-}r\text{ eigenvectors của } \tilde{C}_t$$

hay tương đương, các hàng của $A_t$ là $r$ eigenvectors ứng với eigenvalues lớn nhất của $\tilde{C}_t = Q C_t Q$.

**Thuật toán** (Constrained PCA trong null-space):

```
# Bước 1: Thu thập activation covariance (forward pass nhỏ, trước training)
C_t = ∑ h(x)h(x)^T / N_batch    # covariance input task t (N_batch ~200 batches)

# Bước 2: Project covariance vào null-space
Q = I - P_old                   # null-space projector (từ GPM bases đã lưu)
C_tilde = Q @ C_t @ Q           # projected covariance

# Bước 3: Eigenvector decomposition
eigvals, eigvecs = eigh(C_tilde) # đối xứng → eigh nhanh hơn SVD
top_r_idx = argsort(eigvals, descending=True)[:r]

# Bước 4: Set A_t
A_t = eigvecs[:, top_r_idx].T   # shape (r, d) — direction task-relevant nhất trong null-space
A_t = A_t / norm(A_t, dim=1, keepdim=True) * sqrt(3)  # normalize như InfLoRA gốc
```

#### Ý nghĩa Lý thuyết Thông tin

Theo Data Processing Inequality, với bất kỳ ma trận $A_t$ nào:
$$I(A_t h;\, y) \leq I(h;\, y)$$

Nhưng trong ràng buộc null-space, không phải mọi $A_t$ đều bằng nhau. Data-informed $A_t$ **maximize** $I(A_t h; y)$ trong lớp các $A_t$ thỏa mãn null-space constraint — trong khi random $A_t$ chỉ capture một phần ngẫu nhiên, không được tối ưu hoá.

Ngoài ra, khi $A_t$ được khởi tạo tốt hơn, $B_t$ huấn luyện trong subspace có liên quan đến task → $\sigma_{t,i}$ lớn hơn → spectral signature $\mathcal{S}_t$ mạnh hơn → routing margin $\kappa_{\min}(t)$ trong Định lý 1 **tăng**. Đây là kết nối trực tiếp từ C5 trở lại lý thuyết routing.

#### Tương thích Zero-Replay

$C_t$ được tính từ **dữ liệu training của task hiện tại** (task $t$ đang được huấn luyện). Đây không phải replay (replay = tái sử dụng dữ liệu *cũ*). Dữ liệu training của task hiện tại luôn sẵn có trong CL setting. $A_t$ (model parameter) chỉ encode *hướng* (không phải giá trị hay vị trí dữ liệu cụ thể), tương tự GPM bases cũng tính từ activation covariance và đã được chấp nhận trong InfLoRA, GainLoRA. ✅ zero-replay compliant.

#### Kết nối với Bài toán Gốc

| V6 failure mode | Root cause | C5 giải quyết |
|-----------------|-----------|---------------|
| IMDB/SST2 EM=0 (never-learning) | $A_t$ random bỏ lỡ task-relevant directions trong null-space | $A_t$ data-informed capture variance cao nhất trong null-space → $B_t$ CÓ THỂ học |
| Routing degradation (yelp 55→36) | Expert quality thấp → $\sigma \approx 0$ → signature = noise → routing ngẫu nhiên | Expert quality tăng → $\sigma > 0$ đáng kể → routing có phân biệt |

---

## 5. Những gì Loại bỏ từ GainLoRA

| Thành phần | GainLoRA | SpecRoute | Lý do |
|------------|----------|-----------|-------|
| MLP `trans_input` | Learned routing projection | ❌ Loại bỏ | Duality: spectral affinity là đủ |
| `prompt_key` | Learned per-task key | ❌ Loại bỏ | Thay bằng spectral signatures |
| `previous_trans_input` | Frozen MLP copies | ❌ Loại bỏ | Signatures bất biến theo cấu trúc |
| KL distillation | Replay-based routing loss | ❌ Loại bỏ | Không learned routing → không cần distill |
| GPM trên routing params | Subspace cho routing | ❌ Loại bỏ | Không có routing parameters để bảo vệ |

**Hiệu ứng tổng thể:** Toàn bộ subspace và compute budget mà GainLoRA dành cho routing infrastructure được *thu hồi* cho task learning.

---

## 6. Hai Đóng góp Cốt lõi

> **Nguyên tắc cấu trúc:** Hai đóng góp này tạo thành một vòng lặp logic khép kín — phần 1 xây dựng lý thuyết, phần 2 giải quyết rào cản thực tế để lý thuyết đó có thể hoạt động. Không có phần nào có thể đứng độc lập mà không cần phần kia.

---

### Đóng góp 1: Khung Định tuyến Phổ Phi tham số

> *Tổng hợp từ C1, C2, C3 — một lập luận thống nhất, không phải ba thủ thuật riêng biệt.*

**Vấn đề trung tâm:** Các phương pháp CL trước đây (GainLoRA, O-LoRA) coi routing và bảo vệ là hai bài toán độc lập, dẫn đến vòng lặp xấu: routing parameter trôi dạt → GPM phải bảo vệ routing → subspace của task learning bị thu hẹp → routing lại yếu hơn.

**Đóng góp:** Chúng tôi hình thức hóa và chứng minh rằng **bảo vệ không gian con trực giao và routing phân biệt là hai biểu hiện kép của cùng một cấu trúc phổ** (Định lý 1). Từ tính đối ngẫu này, chúng tôi dẫn xuất cơ chế routing hoàn toàn phi tham số: mỗi input được định tuyến đến expert có spectral affinity cao nhất với subspace đặc trưng của expert đó — không cần tham số học, không cần replay, không tốn GPM overhead cho routing infrastructure.

**Đảm bảo lý thuyết:**
- Routing margin $\geq \kappa_{\min}(t^*) - \varepsilon\, \kappa_{\max}$ — tỷ lệ thuận với chất lượng bảo vệ (Định lý 1).
- Routing weight $w_{t^*} \geq 1-\delta$ với nhiệt độ $\tau \leq m/\ln\!\bigl((T-1)/\delta\bigr)$ (Hệ quả 1).
- Capacity bound $T_{\max} \leq d/r(1-\varepsilon)$ qua lý thuyết Grassmannian packing (Hệ quả 2).
- Routing hoàn toàn bất biến qua thời gian: $h$ từ frozen embedding table, $\mathcal{S}_t$ đóng băng sau training (Mệnh đề 1).

**Tại sao không phải đơn giản hoá mà là tiến bộ lý thuyết:** Kết quả này cho thấy các kiến trúc như GainLoRA đang giải quyết một bài toán không tồn tại (routing parameter learning). Chúng tôi chứng minh rằng bảo vệ tốt ↔ routing tốt — hai bài toán hóa ra là *một*.

---

### Đóng góp 2: Tối ưu hóa Không gian con Dựa trên Dữ liệu

> *Tổng hợp từ C5 — giải quyết rào cản thực tế làm Đóng góp 1 sụp đổ lúc runtime.*

**Vấn đề trung tâm:** Đóng góp 1 yêu cầu $h \in \mathrm{span}(V_{t^*})$ — tức là expert $t^*$ phải học được các biến đổi có ý nghĩa từ dữ liệu task. Điều này phụ thuộc vào chất lượng của $A_t$. InfLoRA (và GainLoRA) khởi tạo $A_t$ ngẫu nhiên rồi chiếu vào null-space — một điểm *tùy ý* trên Grassmannian $\mathrm{Gr}(r, d - N_{\text{protected}})$ với dimension $r(d - N_{\text{protected}} - r)$ rất lớn. Khi null-space co lại theo tasks, xác suất ngẫu nhiên bắt đúng hướng task-relevant tiệm cận về 0.

**Đóng góp:** Chúng tôi phát biểu khởi tạo $A_t$ như **bài toán Constrained PCA trên Grassmannian bị giới hạn** và cung cấp lời giải dạng đóng:

$$\max_{A_t} \;\text{tr}(A_t\, Q C_t Q\, A_t^T) \quad \text{s.t.} \quad A_t A_t^T = I_r \;\Rightarrow\; A_t = \text{top-}r\text{ eigenvectors của } QC_tQ$$

$A_t$ này đảm bảo capture **variance task-relevant tối đa** trong null-space có sẵn — biến vấn đề ngẫu nhiên thành vấn đề tất định. Tuân thủ zero-replay vì $C_t$ tính từ dữ liệu task *hiện tại* (không phải cũ), cùng logic với GPM bases đã được chấp nhận trong InfLoRA.

**Vòng lặp khép kín với Đóng góp 1:** $A_t$ tốt hơn → $B_t$ học trong subspace task-relevant → $\sigma_{t,i}$ lớn hơn sau training → $\kappa_{\min}(t^*)$ trong Định lý 1 tăng → routing margin tăng → Đóng góp 1 hoạt động như lý thuyết dự đoán.

---

> **C4 (gradient preconditioning)** là chi tiết triển khai kỹ thuật cần thiết để sửa điều kiện ma trận sau projection, không phải đóng góp lý thuyết. Xem §4 để biết chi tiết.

---

## 7. Pipeline Huấn luyện

### Task 1 (`--run_single True`)
1. Load pretrained model + fresh LoRA ($A$: Kaiming, $B$: zeros).
2. Huấn luyện chuẩn (chỉ `lora_B`) — single expert, không routing.
3. Sau training: tính $\mathcal{S}_1$ (thin SVD) + GPM bases (ESA threshold).
4. Lưu: LoRA weights, spectral signatures, GPM reg files.

### Task $t \geq 2$
1. Load model + fresh LoRA; load LoRA weights và spectral signatures cũ.
2. **[C5 — MỚI]** Pre-task forward pass (200 batch, no grad):
   - Thu thập activation covariance $C_t$ của inputs task $t$
   - Tính projected covariance: $\tilde{C}_t = Q C_t Q$ ($Q = I - P_{\text{old}}$)
   - Eigenvectors của $\tilde{C}_t$ → khởi tạo $A_t$ (thay thế random Kaiming)
3. InfLoRA: chuẩn hoá $A_t$ (đã nằm trong null-space từ eigenvector decomposition).
4. Huấn luyện `lora_B` với spectral affinity routing + adaptive bias $\beta(n)$ + C4.
5. Sau training: tính $\mathcal{S}_t$ (cả inference routing và storage) + cập nhật GPM bases.
6. Lưu tất cả artifacts cho task tiếp theo.

---

## 8. Mapping Lý thuyết – Implementation

| Lý thuyết | Implementation | File |
|-----------|---------------|------|
| Spectral signature $\mathcal{S}_t$ | `compute_spectral_signatures()` (thin QR+SVD) | `t5_specroute.py` |
| Spectral affinity $\alpha_t(h)$ (inference) | σ²-weighted Rayleigh quotient | `compute_spectral_routing()` |
| A-row proxy $\alpha_t^{\mathrm{train}}$ (training) | A-row fit + adaptive bias $\beta(n)$ | `compute_spectral_routing()` |
| Symmetric inference SVD | `prepare_inference_routing()` → SVD của $B_t A_t$ | `t5_specroute.py` |
| Routing $w = \mathrm{softmax}(\alpha / \tau)$ | `torch.softmax(fit_scores / temp)` | `compute_spectral_routing()` |
| Drift-free input $h$ | `inputs_embeds = embed_tokens(input_ids)` → mean-pool | `T5Stack.forward()` |
| GPM + InfLoRA null-space | `get_reg_matrix()` | `cl_trainer_specroute.py` |
| Dynamic ESA threshold | `(1−ε₀)·t/T + ε₀` | `cl_trainer_specroute.py` |
| C4: Preconditioner | `precompute_preconditioners()` → eigendecomposition | `cl_trainer_specroute.py` |
| C4: Spectral entropy reg | `_compute_spectral_entropy_loss()` → QR trick | `cl_trainer_specroute.py` |
| **C5: Data-informed init** | **`pre_task_data_collection()` → `eigh(Q@C@Q)` → set `lora_A.data`** | **`cl_trainer_specroute.py`** |

---

## 9. Thiết lập Thí nghiệm

| Hạng mục | Giá trị |
|----------|---------|
| Mô hình | `google/flan-t5-small` (60M) / `flan-t5-large` (783M) |
| Benchmarks | SuperNI (15 tasks, 2 orderings), Long (15 tasks, 2 orderings) |
| Metrics | AP (Average Performance, ↑), FT (Forgetting, ↓) |
| LoRA | $r = 8$, target=Q+V, InfLoRA (chỉ B trained, A đóng băng) |
| Routing | $\tau = 1.0$, $\alpha_{\mathrm{target}} = 0.8$, adaptive $\beta(n)$ (train); SVD đối xứng (inference) |
| ESA | $\varepsilon_0 = 0.995$ (dynamic) |
| C4 | $\lambda_{\text{entropy}} = 0.01$, preconditioning on, $\epsilon = 10^{-6}$, warmup = 10% |
| **C5** | **N_batch_warmup = 200, dùng `torch.linalg.eigh` trên projected covariance** |
| Precision | fp32 + gradient checkpointing |
| So sánh | Batch size, LR, scheduler khớp chính xác ROOT (GainLoRA) |

---

## 10. File Map

| File | Vai trò |
|------|---------|
| `src/t5_specroute.py` | T5Stack + spectral routing + thin SVD |
| `src/t5_gainlora_inflora.py` | LoRALayer, T5Attention, T5Block (shared base) |
| `src/cl_trainer_specroute.py` | Trainer: GPM, InfLoRA, ESA, C4, C5, training_step |
| `src/run_t5.py` | Entry: model loading, parameter freezing |
| `gen_script_*_specroute*.sh` | Experiment scripts |
