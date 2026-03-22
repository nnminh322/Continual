# SpecRoute: Định tuyến Phổ Dẫn dắt bởi Dữ liệu trong Học Liên tục với LoRA

> **Tài liệu thiết kế chính thức — V8**
> Ràng buộc: Zero-replay nghiêm ngặt. Theory-first. Tổng hợp sau V2–V7.

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

### 2.3 GPM–Routing Paradox: Duality sụp đổ với A ngẫu nhiên

**Đây là phát hiện cốt lõi giải thích V2–V6 thất bại (AP ≈ 27–40).**

Routing đo $\alpha_t(h) \propto \|A_t h\|^2$. Để phân biệt task $t^*$ với $s$, cần $A_{t^*}$ align với input $h \sim p_{t^*}$.

**Nghịch lý:** InfLoRA khởi tạo $A_t$ **ngẫu nhiên** (Kaiming) rồi chiếu vào null-space. Kết quả:

1. $\text{rowspace}(B_tA_t) \subseteq \text{rowspace}(A_t)$ — rowspace không mở rộng qua phép nhân trái.
2. $\text{rowspace}(A_t) = r$ hướng **ngẫu nhiên** trong $d'$-chiều null-space (không encode task identity).
3. Với same-domain tasks (yelp/amazon/sst2/imdb): input $h$ có variance theo hướng CHUNG → signatures không phân biệt được.

**Empirical:** V6 IMDB (task 8): EM = 0.0 suốt 10 epochs dù training loss giảm — routing inference gán sai expert. V3 RTE, MNLI cũng tương tự. Root cause = không phải null-space exhaustion mà là **routing signal không có trong random $A_t$**.

### 2.4 Lối Thoát: Differential Projection via Data-Informed Init

**Lemma (sẽ chứng minh ở §3.5):** Do InfLoRA constraint $A_t P_{t-1} = 0$:
$$\|A_t h\|^2 \;=\; \|A_t Q_{t-1} h\|^2 \quad \forall h, \quad Q_{t-1} = I - P_{t-1}$$

Routing $\alpha_t$ **chỉ nhìn thành phần $Q_{t-1}h$** — phần của $h$ NGOÀI span tất cả task cũ. Với $h \sim p_s$ ($s < t$): GPM đã capture $\geq 99.5\%$ variance của $p_s$ → $\|Q_{t-1}h\|^2 \leq 0.005\,\mathrm{tr}(C_s)$ → $\alpha_t(h) \approx 0$ tự nhiên. **Đây là differential projection — task-discriminative theo thiết kế.**

**Vấn đề còn lại:** Random $A_t$ không align với hướng có variance cao nhất của task $t$ trong null-space → $\|A_t Q_{t-1}h\|^2$ nhỏ dù $\|Q_{t-1}h\|^2$ đáng kể.

**Giải pháp — C5:** Khởi tạo $A_t$ = top-$r$ eigenvectors của $\tilde{C}_t = Q_{t-1}C_tQ_{t-1}$. Khi đó:
$$E_{h \sim p_t}[\|A_t h\|^2] = \sum_{i=1}^r \lambda_i(\tilde{C}_t) \quad \textbf{— GIÁ TRỊ CỰC ĐẠI}$$

**C5 biến random routing key thành optimal routing key — giải quyết GPM–Routing Paradox.**

> **Kết nối C5 ↔ C2:** C5 không chỉ giúp learning (maximize captured variance trong null-space) mà **đồng thời** maximize routing signal. Một initialization, hai mục tiêu. Hai contribution C1/C2 và C5 là bất khả phân.

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

**Hệ quả 2** *(Capacity Bound — Kết nối Grassmannian).* Gọi $k_t$ là **GPM effective rank** của task $t$ — số eigenvectors thực tế được GPM giữ lại (threshold 99.5%). Số lượng tasks tối đa trước khi null-space sụp đổ:

$$T_{\max} \;\leq\; \frac{d}{\bar{k}\,(1 - \varepsilon)}, \qquad \bar{k} = \frac{1}{T}\sum_{t=1}^T k_t$$

**Lưu ý quan trọng:** $\bar{k}$ thực tế $\gg r$. Với T5-small ($d=512$) và NLP tasks phong phú, GPM giữ $k_t \approx 30\text{–}80$ dims/task để đạt 99.5% variance (không phải $r=8$). Ước tính thực tế: 15 tasks × 50 dims = 750 $> d$ → **null-space bão hòa là rủi ro thực**, không phải lý thuyết. Đây là lý do §3.10 thảo luận null-space collapse riêng. Bound Grassmannian vẫn đúng về *số lượng subspace $r$-chiều có thể pack*, nhưng capacity constraint của GPM là $\sum_t k_t \leq d$, chặt hơn $\sum_t r \leq d$.

### 3.4 Cam kết Trực giao từ Kiến trúc InfLoRA

> **Đây là phần đóng cửa lỗ hổng lý thuyết then chốt.** Reviewer thường lo ngại: "GPM gradient projection chỉ chiếu gradient, không đảm bảo các $\Delta W_t$ có subspace trực giao." Observation này *đúng* về GPM gradient projection nhưng *nhầm cơ chế* — tính trực giao đến từ bước khác: InfLoRA A-projection, cứng hơn nhiều.

**Mệnh đề 2** *(InfLoRA đảm bảo Điều kiện Định lý 1).* Với $P_{\text{old}} = \mathcal{B}\mathcal{B}^T$ là GPM projection matrix (built từ tasks $1,\ldots,t-1$), bước InfLoRA chiếu **tất cả hàng của $A_t$ vào null-space của $P_{\text{old}}$**:

$$A_t \leftarrow A_t(I - P_{\text{old}}) \quad\Rightarrow\quad \text{rowspace}(A_t) \subseteq \text{null}(P_{\text{old}})$$

Khi đó:

$$\text{span}(V_t) \;=\; \text{rowspace}(\Delta W_t) \;\subseteq\; \text{rowspace}(A_t) \;\subseteq\; \text{null}(P_{\text{old}})$$

**(Chứng minh từng bước.)**
- $\text{rowspace}(B_t A_t) \subseteq \text{rowspace}(A_t)$: đúng với mọi $B_t$ (phép nhân bên trái không mở rộng rowspace).
- $\text{rowspace}(A_t) \subseteq \text{null}(P_{\text{old}})$: bởi bước InfLoRA projection ở trên.
- GPM bases $\mathcal{B}$ span xấp xỉ $\text{rowspace}(A_s)$ cho các task $s < t$ (vì GPM tích lũy principal input directions, mà activation của task $s$ chủ yếu kích hoạt theo hướng $A_s$).
- Do đó: $\text{span}(V_t) \subseteq \text{null}(P_{\text{old}}) \approx \perp \text{span}(V_s)$ với mọi $s < t$. $\square$

**Chất lượng xấp xỉ:** Với GPM threshold $\varepsilon_0 = 0.995$ (capture ≥ 99.5% variance), $\delta_{t,s} \leq 0.005 \ll \kappa_{\min}(t^*)$ trong thực tế.


**Phân tích độ nhạy GPM (Davis–Kahan perturbation).** GPM trong thực tế chỉ capture *xấp xỉ* principal directions nên có sai số $\Delta P_{t-1}$ so với GPM lý tưởng. Theo Davis–Kahan theorem, perturbation trong subspace angle bị chặn bởi:
$$\sin\!\bigl(\Theta(\hat{V}_s, V_s)\bigr) \leq \frac{\|\hat{C}_s - C_s\|_2}{\delta_{\text{gap}}(C_s)}$$
trong đó $\hat{C}_s$ là sample covariance (200 batches), $\delta_{\text{gap}}$ là eigenvalue gap. Với batch size 200 và T5-small $d=512$, sai số này nhỏ khi $\delta_{\text{gap}}$ đủ lớn (task distributions phân kỳ). Kết quả thực tế: margin trong Định lý 1 bị giảm thêm $O(\|\Delta P\|_F / \delta_{\text{gap}})$ — nhỏ với tasks có spectrum phân kỳ, lớn hơn với same-domain tasks (expected). Xẩy ra same-domain routing failure vẫn là giới hạn của *mọi* zero-replay CL method, không phải đặc thù SpecRoute.

### 3.5 Lemma về Differential Projection

**Lemma 1** *(Differential Projection — Exact).* Với $A_t$ thoả mãn InfLoRA constraint $A_t P_{t-1} = 0$, với **mọi** $h \in \mathbb{R}^d$:
$$\|A_t h\|^2 \;=\; \|A_t Q_{t-1} h\|^2, \quad Q_{t-1} = I - P_{t-1}$$

**Chứng minh.** Viết $h = P_{t-1} h + Q_{t-1} h$. Vì $A_t P_{t-1} = 0$ (các hàng $A_t$ trực giao với colspace của $P_{t-1}$):
$$A_t h = A_t P_{t-1} h + A_t Q_{t-1} h = 0 + A_t Q_{t-1} h \qquad \square$$

**Hệ quả A (Current expert trên old data — from Lemma 1):** Với $h \sim p_s$ ($s < t$):
$$E_{h \sim p_s}\!\left[\alpha_t(h)\right] = \frac{E[\|A_t Q_{t-1} h\|^2]}{r\,\|h\|^2} \leq \frac{\mathrm{tr}(Q_{t-1} C_s)}{r} \leq \frac{(1-\tau_\text{GPM})\,\mathrm{tr}(C_s)}{r} \leq \frac{0.005\,\mathrm{tr}(C_s)}{r}$$

**Hệ quả B (Old expert trên new data — GPM-capture argument):** Với $h \sim p_t$ ($t > s$). Vì GPM sau task $s$ tích lũy principal directions của task $s$'s activations, trong đó $A_s$'s rowspace (= top-$r$ directions của $\tilde{C}_s$) được capture vào $P_s \subseteq P_{t-1}$. Do đó:
$$\text{rowspace}(A_s) \subseteq \text{range}(P_{t-1}) \quad\Rightarrow\quad A_s Q_{t-1} = 0$$
Suy ra $A_s h_t = A_s P_{t-1} h_t$ và:
$$E_{h \sim p_t}[\alpha_s(h)] = \frac{E[\|A_s P_{t-1} h_t\|^2]}{r\,\|h_t\|^2} \leq \frac{\mathrm{tr}(P_{t-1} C_t)}{r\,\mathrm{tr}(C_t)} \cdot \frac{\mathrm{tr}(C_t)}{r}$$
Với task $t$ có domain mới (khác task cũ): $\mathrm{tr}(P_{t-1} C_t) / \mathrm{tr}(C_t) = \text{PEV}_{t,\text{old}} \ll 1$ — fraction variance của task $t$ được giải thích bởi các cơ sở cũ. Với same-domain tasks: $\text{PEV}_{t,\text{old}}$ lớn hơn, đây là giới hạn cơ bản của mọi zero-replay CL method, không phải lỗ hổng đặc thù của SpecRoute.

---

### 3.6 Định lý C5 Routing Optimality (Đóng góp Lý thuyết Chính)

**Định nghĩa 4** *(Restricted Stiefel Manifold).*
$$\mathcal{A}_t = \{A \in \mathbb{R}^{r \times d} : A P_{t-1} = 0,\; A A^\top = I_r\}$$

**Định lý 2** *(C5 Routing Optimality).* Với $C_t = E_{h \sim p_t}[hh^\top]$ và $\tilde{C}_t = Q_{t-1} C_t Q_{t-1}$:
$$\operatorname{argmax}_{A_t \in \mathcal{A}_t} E_{h \sim p_t}[\alpha_t(h)] \;=\; \text{top-}r\text{ eigenvectors của } \tilde{C}_t$$
Giá trị cực đại: $\displaystyle\frac{1}{r}\sum_{i=1}^r \lambda_i(\tilde{C}_t)$

**Chứng minh.** Từ Lemma 1:
$$E_{h \sim p_t}[\alpha_t(h)] = \frac{E[\|A_t Q_{t-1} h\|^2]}{r\,E[\|h\|^2]} = \frac{\mathrm{tr}(A_t\,\tilde{C}_t\,A_t^\top)}{r}$$
Với ràng buộc $A_t A_t^\top = I_r$, đây là **Constrained PCA** tiêu chuẩn trên $\tilde{C}_t$: lời giải là eigenvectors ứng với eigenvalues lớn nhất. Đây chính xác là C5. $\square$

**Ý nghĩa:** C5 **đồng thời** tối ưu:
1. **Learning quality:** maximize $\mathrm{tr}(A_t \tilde{C}_t A_t^\top)$ = variance captured trong null-space → $B_t$ học được hiệu quả.
2. **Routing signal:** maximize $E[\alpha_t(h)]$ → routing phân biệt task $t$ tốt hơn mọi init khác.

*Đây là lý do C5 và C2 là bất khả phân: C5 biến random routing key thành optimal routing key.*

---

### 3.7 Định lý Routing Margin với GPM + C5

**Định lý 3** *(Explicit Routing Margin).* Gọi $\lambda_t^\min = \lambda_r(\tilde{C}_t)$ (r-th eigenvalue của projected covariance). Với C5 init và A-row routing ($\tau_\text{GPM} = 0.995$):
$$\boxed{E_{h \sim p_t}[\alpha_t(h)] - \max_{s < t}\,E_{h \sim p_s}[\alpha_t(h)] \;\geq\; \frac{\lambda_t^\min}{r} - \frac{0.005\,\bar{\sigma}^2}{r}}$$

với $\bar{\sigma}^2 = \max_s \mathrm{tr}(C_s)$.

**Hệ quả (GPM–Routing Paradox Formalized):** Với random $A_t$ (không có C5), routing signal:
$$E_{h \sim p_t}[\alpha_t^\text{rand}(h)] \approx \frac{\mathrm{tr}(\tilde{C}_t)}{d'}$$

Tỷ lệ lợi thế C5 over random:
$$\frac{E[\alpha_t^\text{C5}(h)]}{E[\alpha_t^\text{rand}(h)]} \;=\; \underbrace{\frac{d'}{r}}_{\text{null-space factor}} \cdot \underbrace{\text{PEV}_r(\tilde{C}_t)}_{\text{task concentration}}$$

**Quan sát quan trọng:** Factor $d'/r$ và $\text{PEV}_r$ đều **tăng ý nghĩa về mặt routing khi null-space shrinks** (later tasks). C5 quan trọng nhất chính khi routing khó nhất.

Với T5-small task 8 ($d' \approx 351$, $r=8$): tỷ lệ $\approx 44\times \cdot \text{PEV}_8 \gg 1$.

---

### 3.8 Training–Inference Symmetry

**Mệnh đề 2** *(Routing Symmetry).* V8 training và inference dùng cùng formula:

| Phase | Formula |
|-------|---------|
| Training (task $t$) | $\alpha_t^\text{train}(h) = \dfrac{\|A_t h\|^2}{r\|h\|^2} + \beta(n)$ |
| Inference | $\alpha_t^\text{inf}(h) = \dfrac{\|A_t h\|^2}{r\|h\|^2}$ |

**Không có metric asymmetry.** V2–V7 dùng A-row proxy khi train nhưng SVD Rayleigh quotient khi inference — đây là nguồn gốc mismatch vì hai formula sống trong *hai metric space khác nhau*. V8 dùng A-row cho cả hai: đây là "symmetry" về mặt metric formula.

$\beta(n) = \tau\ln\!\bigl(\alpha_\text{target} \cdot n / (1-\alpha_\text{target})\bigr)$ — **cold-start mechanism** cho gradient flow khi $B_t = 0$ lúc đầu training. Nó không phải "fake routing quality"; expert $B_t$ học để transform task $t$'s features trong null-space, và những features đó (C5 A-row directions) vẫn có tín hiệu tại inference dù không còn β. β giải quyết vấn đề *gradient starvation*, không phải routing quality. Loại bỏ β khi inference là *có chủ đích* — không có task ID, không thể biết n để tính β, và không cần: $A_t$ đã align với task $t$'s null-space variance.

> **Về asymmetry β tạo ra:** Expert $B_t$ được train với score tổng $\alpha_t + \beta$ nhưng deploy không có β. Điều này CÓ ảnh hưởng nếu $\alpha_t$ lúc inference quá nhỏ để win softmax. Đây là lý do Phản biện 1 (asymmetric capacity giữa old vs new experts) là concern thực — và lý do mục tiêu của C5 maximize $\alpha_t$ là cần thiết: đảm bảo $\|A_t Q_{t-1} h\|^2$ đủ lớn kể cả không có β.

**Điều kiện α-sufficiency (để inference không cần β):** Routing của expert $t$ thắng tại inference nếu:
$$E[\alpha_t(h) \mid h \sim p_t] - \max_{s \neq t} E[\alpha_s(h) \mid h \sim p_t] > 0$$
Điều kiện này được hụ bởi Định lý 3: margin là $\lambda_t^\min/r - 0.005\bar{\sigma}^2/r$, dương khi $\lambda_t^\min > 0.005\bar{\sigma}^2$. C5 maximize $\lambda_t^\min = \lambda_r(\tilde{C}_t)$ (r-th eigenvalue của projected covariance), đây cũng chính là largest feasible $\lambda_r$ trong $\mathcal{A}_t$. Nếu $\lambda_r(\tilde{C}_t)$ quá nhỏ (null-space collapse hoặc task tương đồng domain), margin bị nhiễu — đây là giới hạn của setting, được thảo luận trong §3.10.

---

### 3.9 Drift Invariance

**Mệnh đề 3** *(Drift-Free Routing).* Hàm routing $h \mapsto \alpha_t(h)$ bất biến qua tất cả tasks: $h$ từ frozen embedding table trước mọi attention layer; $A_t$ đóng băng sau C5 init. $\square$

**Làm rõ về layer routing:** Routing được tính *một lần* tại input (token embedding, trước block transformer đầu tiên) — không phải routing riêng biệt tại mỗi transformer layer. Vector $h$ = mean-pool của token embeddings (frozen `embed_tokens`), không thay đổi qua training. Điều này đảm bảo routing hoàn toàn frozen và không drift. C5 initialization per-layer (mỗi attention layer có $A_t^{(l)}$ riêng) phục vụ *learning quality* chứ không phải routing — routing chỉ dùng encoder layers để aggregate signal.

---

### 3.10 Vấn đề Null-Space Collapse (Còn tồn tại, được giải một phần)

Định lý 1 giả định $h \in \text{span}(V_{t^*})$. Điều kiện này:
- **(A) Expert phải học được:** $A_t$ phải align với task-relevant directions. C5 giải quyết bằng data-informed init.
- **(B) Input phải co projection:** Inputs thực phải có energy trên $\text{span}(V_t)$.

C5 giải (A). (B) được đảm bảo khi null-space ($d'$) còn đủ rộng để capture task-t variance. Với $d=512$, $r=8$, 15 tasks: null-space vẫn đủ theo Hệ quả 2.



Null-space (sau $t-1$ tasks) là một không gian $d - N_{\text{protected}}$ chiều. Kaiming random trong không gian này KHÔNG ĐẢM BẢO alignment với các hướng có liên quan đến task $t$. Khi null-space thu hẹp dần (Layer 7: 8/512 → 161/512 → 344/512 qua 13 tasks), xác suất random init bắt được đúng hướng task-relevant giảm theo.

**Hệ quả thực nghiệm (V6):** IMDB (task 8) — eval_loss dừng ở 6.37 sau 10 epoch, EM=0.0 suốt quá trình, expert thực sự không thể học bất cứ điều gì hữu ích.

---

## 4. Các Thành phần Framework

### C1 — Spectral Expert Signatures (V8: A_t as Signature)

**V8 thay đổi từ V7:** Signature là $\mathcal{S}_t = A_t$ (model parameter), **không cần thin SVD post-training**. Lý do từ Định lý 2: rowspace của $V_t$ (từ SVD của $B_tA_t$) = rowspace của $A_t$ (phép nhân trái không mở rộng rowspace) — SVD chỉ thêm $\sigma$-weighting gây noise. Với C5 init, $A_t$ rows **đã là** task-discriminative directions.

- **Không cần `prepare_inference_routing()`** — loại bỏ $O(dr^2)$ overhead per task per layer.
- **Không tham số bổ sung** — $A_t$ là model parameter đã có.
- **Bất biến** — $A_t$ đóng băng sau C5 init (Mệnh đề 3).

### C2 — Data-Informed Differential Routing (V8)

**V8 thay đổi từ V7:** Cả training lẫn inference đều dùng **A-row formula** — loại bỏ hoàn toàn `prepare_inference_routing()` và SVD inference mismatch. Lý do từ Lemma 1 + Định lý 2: $A_t$ rows với C5 init đã là routing optimal directions; SVD của $B_tA_t$ chỉ thêm $\sigma^2$-weighting từ B optimization artifact, không có đảm bảo lý thuyết.

**Routing formula (Training và Inference):**

$$w(h) = \mathrm{softmax}\!\left(\frac{[\alpha_1(h),\; \ldots,\; \alpha_T(h)]}{\tau}\right)$$

$$\alpha_t(h) = \frac{\|A_t\, h\|^2}{r\,\|h\|^2}, \qquad \text{(A-row affinity — cả train lẫn inference)}$$

**Training** (task $t$ thêm adaptive bias):

$$\alpha_t^{\mathrm{train}}(h) = \frac{\|A_t\, h\|^2}{r\,\|h\|^2} + \beta(n), \qquad \beta(n) = \tau \cdot \ln\!\left(\frac{\alpha_{\mathrm{target}} \cdot n}{1 - \alpha_{\mathrm{target}}}\right)$$

trong đó $n = |\{\text{task cũ}\}|$ và $\alpha_{\mathrm{target}} \in (0,1)$ là routing weight mục tiêu cho task hiện tại (mặc định 0.8).

**Lý giải A-row routing (từ Lemma 1 + Định lý 2):**

- **Exact decomposition (Lemma 1):** $\|A_t h\|^2 = \|A_t Q_{t-1} h\|^2$ — routing chỉ nhìn null-space component, không bị ảnh hưởng bởi task cũ.
- **Optimality với C5 (Định lý 2):** C5 init chọn $A_t$ = argmax $E[\|A_t h\|^2]$ trên tất cả $A_t \in \mathcal{A}_t$ — A-row affinity là tốt nhất có thể trong constraint.
- **Margin bound (Định lý 3):** $E[\alpha_t|h \sim p_t] - \max_s E[\alpha_t|h \sim p_s] \geq \lambda_t^\min/r - 0.005\bar{\sigma}^2/r > 0$.
- **Loại bỏ `prepare_inference_routing()`:** $A_t$ đóng băng sau C5 init — là signature không cần tái tính SVD của $B_tA_t$.

**Lý giải adaptive $\beta(n)$:** Giải $w_t = \alpha_{\mathrm{target}}$ trong softmax → closed-form $\beta(n) = \tau\ln(\alpha_\text{target} \cdot n/(1-\alpha_\text{target}))$. Tránh $O(1/n)$ softmax dilution khi số task tăng.

**Lợi thế C5 so với random A-row (Hệ quả Định lý 3):**
$$\frac{E[\alpha_t^\text{C5}(h)]}{E[\alpha_t^\text{rand}(h)]} = \frac{d'}{r} \cdot \text{PEV}_r(\tilde{C}_t) \approx 44\times \text{ tại task 8 (T5-small)}$$

| Phase | Cơ chế routing | Notes |
|-------|----------------|-------|
| Training (task $t$) | $\|A_t h\|^2/(r\|h\|^2) + \beta(n)$ | $\beta$ bù softmax dilution |
| **Inference (mọi task)** | **$\|A_t h\|^2/(r\|h\|^2)$** | **V8: không SVD, không $\sigma^2$-weighting** |
| Đảm bảo lý thuyết | Định lý 3 margin bound + C5 advantage $44\times$ | Cả hai phase đều dùng cùng metric space |

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
C_t = ∑ h(x)h(x)^T / N_batch    # covariance input task t (N_batch ~100 batches)

# Bước 2: Project covariance vào null-space
Q = I - P_old                   # null-space projector (từ GPM bases đã lưu)
C_tilde = Q @ C_t @ Q           # projected covariance

# Bước 3: Eigenvector decomposition
eigvals, eigvecs = eigh(C_tilde) # đối xứng → eigh nhanh hơn SVD

# Bước 4: Fallback nếu signal quá yếu (degenerate null-space)
if eigvals[-1] < 1e-6:
    # Null-space bị bão hoà hoặc task không có activation rõ ràng
    # Revert về Kaiming random init + InfLoRA projection như gốc
    continue

top_r_idx = argsort(eigvals, descending=True)[:r]

# Bước 5: Set A_t
A_t = eigvecs[:, top_r_idx].T   # shape (r, d) — direction task-relevant nhất trong null-space
A_t = A_t / norm(A_t, dim=1, keepdim=True) * sqrt(3)  # normalize như InfLoRA gốc
```

**Điều kiện fallback:** Nếu `max_eigenvalue(C_tilde) < 1e-6`, null-space quá hẹp hoặc activation không có signal đủ mạnh. Trong trường hợp này, C5 nhường cho Kaiming init + InfLoRA projection tiêu chuẩn — không làm tệ hơn V6, chỉ không cải thiện. Điều kiện này chỉ xảy ra khi null-space gần như bão hoà, tức là ESA đã tiêu thụ gần hết capacity.

**C5 per-layer:** Mỗi LoRA layer (encoder Q, V; decoder self/cross Q, V) có $C_t$ riêng thu thập từ activation tương ứng của layer đó. GPM cũng lưu $P_{\text{old}}^{(l)}$ riêng theo layer $l$. Do đó eigenvector decomposition được thực hiện độc lập cho từng layer — mỗi $A_t^{(l)}$ chỉ capture variance task-relevant trong null-space của layer $l$.

**Hệ kết với Routing:** Routing sử dụng input embedding (frozen embedding table output, trước tất cả transformer layers) và A-row của các encoder layers — không phải per-layer routing riêng biệt. C5 per-layer cải thiện **học hiệu quả** của $B_t$ tại mỗi layer, còn routing signal (§3.2) được aggregate qua các encoder layers.

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
| **`prepare_inference_routing()`** | **SVD của $B_tA_t$ post-training** | **❌ Loại bỏ (V8)** | **$A_t$ là signature, kh\u00f4ng c\u1ea7n t\u00e1i t\u00ednh SVD; lo\u1ea1i b\u1ecf $O(dr^2)$ overhead** |
| **SVD $\sigma^2$-weighted inference** | **Rayleigh quotient khác train formula** | **❌ Loại bỏ (V8)** | **Train-inference mismatch; A-row đủ từ Lemma 1 + Định lý 2** |

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
- Capacity bound $T_{\max} \leq d/(\bar{k}(1-\varepsilon))$ qua lý thuyết Grassmannian packing (Hệ quả 2), trong đó $\bar{k}$ là GPM effective rank ($\approx 30$–$80$ dims/task), không phải LoRA rank $r=8$.
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
3. Sau training: cập nhật GPM bases (ESA threshold). **Không cần tính SVD hay `prepare_inference_routing()`** — $A_t$ là signature.
4. Lưu: LoRA weights, GPM reg files.

### Task $t \geq 2$
1. Load model + fresh LoRA; load LoRA weights cũ.
2. **[C5 — MỚI]** Pre-task forward pass (200 batch, no grad):
   - Thu thập activation covariance $C_t$ của inputs task $t$
   - Tính projected covariance: $\tilde{C}_t = Q C_t Q$ ($Q = I - P_{\text{old}}$)
   - Eigenvectors của $\tilde{C}_t$ → khởi tạo $A_t$ (thay thế random Kaiming)
3. InfLoRA: chuẩn hoá $A_t$ (đã nằm trong null-space từ eigenvector decomposition).
4. Huấn luyện `lora_B` với A-row routing + adaptive bias $\beta(n)$ + gradient preconditioning (C4.1).
5. Sau training: cập nhật GPM bases (200 batches). **Không gọi `prepare_inference_routing()`** — $A_t$ đóng băng từ bước 2 là signature đủ dùng.
6. Lưu tất cả artifacts cho task tiếp theo.

---

## 8. Mapping Lý thuyết – Implementation

| Lý thuyết | Implementation | File |
|-----------|---------------|------|
| Spectral signature $\mathcal{S}_t = A_t$ | `lora_A` weights (frozen after C5 init) — **không cần SVD** | `t5_specroute.py` |
| Spectral affinity $\alpha_t(h)$ (inference) | A-row fit: `(proj**2).sum(-1) / (r * h_norm_sq)` | `compute_spectral_routing()` |
| A-row proxy $\alpha_t^{\mathrm{train}}$ (training) | A-row fit + adaptive bias $\beta(n)$ | `compute_spectral_routing()` |
| ~~Symmetric inference SVD~~ | ~~`prepare_inference_routing()` → SVD của $B_t A_t$~~ | **❌ Loại bỏ hoàn toàn (V8)** |
| Routing $w = \mathrm{softmax}(\alpha / \tau)$ | `torch.softmax(fit_scores / temp)` | `compute_spectral_routing()` |
| Drift-free input $h$ | `inputs_embeds = embed_tokens(input_ids)` → mean-pool | `T5Stack.forward()` |
| GPM + InfLoRA null-space | `get_reg_matrix()` | `cl_trainer_specroute.py` |
| Dynamic ESA threshold | `(1−ε₀)·t/T + ε₀` | `cl_trainer_specroute.py` |
| C4: Preconditioner | `precompute_preconditioners()` → eigendecomposition | `cl_trainer_specroute.py` |
| **C5: Data-informed init** | **`pre_task_data_collection()` → `eigh(Q@C@Q)` → set `lora_A.data`** | **`cl_trainer_specroute.py`** |
| C5: Fallback | max eigval < 1e-6 → skip C5, keep Kaiming + InfLoRA projection | `cl_trainer_specroute.py` |

---

## 9. Thiết lập Thí nghiệm

| Hạng mục | Giá trị |
|----------|---------|
| Mô hình | `google/flan-t5-small` (60M) / `flan-t5-large` (783M) |
| Benchmarks | SuperNI (15 tasks, 2 orderings), Long (15 tasks, 2 orderings) |
| Metrics | AP (Average Performance, ↑), FT (Forgetting, ↓) |
| LoRA | $r = 8$, target=Q+V, InfLoRA (chỉ B trained, A đóng băng) |
| Routing | $\tau = 1.0$, $\alpha_{\mathrm{target}} = 0.8$, adaptive $\beta(n)$ (train); **A-row formula (inference, V8 — không SVD)** |
| ESA | $\varepsilon_0 = 0.995$ (dynamic) |
| C4 | Gradient preconditioning bật (`--use_preconditioning True`), $\epsilon = 10^{-6}$; entropy reg đã loại bỏ V7 |
| **C5** | **N_batch = 100, `torch.linalg.eigh` trên projected covariance, fallback nếu max_eigval < 1e-6** |
| GPM repr. | 200 batches (giảm từ 1000 — SVD ổn định sau 200) |
| **Scalability note** | **C5 per-layer eigdecomp: $O(d^2)$ per layer per task. Với T5-small ($d=512$) tất cả layers: chấp nhận được. Với flan-t5-large ($d=1024$): 4× đắt hơn nhưng vẫn chỉ tuyến tính theo tasks. Với LLaMA-7B ($d=4096$): sử dụng randomized SVD hoặc Lanczos (top-$r$ eigenvecs không cần full decomp) giảm xuống $O(dr)$ per layer.** |
| Precision | fp32 + gradient_checkpointing (T5 + P100: fp16 có risk NaN overflow với large softmax) |
| P100 BSZ | BSZ=8, GA=4 (effective 32); T4: BSZ=2, GA=8 |
| Thời gian (P100 16GB) | SuperNI T5-Small ≈ 2-3h; Long benchmark ≈ 3-4h — thoải mái trong 12h Kaggle |
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
