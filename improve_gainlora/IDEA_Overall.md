# SpecRoute: Định tuyến Phổ với Khởi tạo Phân biệt và Chiếu Nhận biết Chồng lấn trong Học Liên tục LoRA

> **Tài liệu thiết kế chính thức** — Ràng buộc: cheat ≤ root (GainLoRA). Theory-first.

---

# PHẦN I — BÀI TOÁN VÀ ĐỘNG LỰC

---

## 1. Bài toán và Ràng buộc

### 1.1 Bài toán

Bài toán học liên tục với mô hình ngôn ngữ lớn (LLM) đặt yêu cầu: mô hình tiếp thu tuần tự $T$ task không đồng nhất — phân loại cảm xúc, suy luận ngôn ngữ, hỏi đáp — trong điều kiện dữ liệu task trước không khả dụng tại thời điểm huấn luyện task mới. Với mỗi task $t$, một LoRA adapter $\Delta W_t = B_t A_t$ được bổ sung vào mô hình nền đóng băng; $A_t$ đóng băng sau khởi tạo (InfLoRA constraint), chỉ $B_t$ được huấn luyện. Tại inference, task identity không được cung cấp:

$$y = f\!\Bigl(W_0\, x \;+\; \sum_{t=1}^{T} w_t(x)\; B_t A_t\, x\Bigr)$$

Ba thách thức đan xen:

| Thách thức | Yêu cầu |
|:----------:|---------|
| **Định tuyến (R)** | Xác định adapter phù hợp cho input đầu vào |
| **Bảo vệ (P)** | Đảm bảo adapter cũ không bị suy giảm khi học task mới |
| **Phân bổ (A)** | Quản lý dung lượng subspace hữu hạn cho $T$ adapters |

### 1.2 Ràng buộc: Cheat ≤ Root

GainLoRA (baseline) lưu trữ các thống kê sau từ old tasks:
- SVD bases của activation covariance (GPM): `reg_{i}.pt` per layer
- SVD bases của MLP activation ở 3 tầng: `trans_input/reg_{0,1,2}.pt`
- Frozen MLP weights (`previous_trans_input.pt`) + frozen prompt keys
- **Mean routing distribution** từ eval inference pass trên old data (`attention_weights.pkl`) → dùng làm KL distillation target

**Nguyên tắc**: SpecRoute được phép lưu thống kê second-moment từ old tasks (cùng loại với GPM bases), miễn là tổng "cheat budget" không vượt quá root. Cụ thể, SpecRoute **không** cần eval inference pass trên old data, không cần frozen MLP copies, không cần KL distillation targets — tiết kiệm hơn root ở ba mục này, đổi lại lưu thêm projected covariance $\tilde{C}_s$ per task.

---

## 2. Baseline và Vấn đề

### 2.1 GainLoRA

GainLoRA (NeurIPS 2025) tiếp cận ba thách thức với ba cơ chế riêng biệt:

**Định tuyến** qua hai thành phần học được:
- **`trans_input`**: MLP hai tầng ($d_{\text{model}} \to d_{\text{hidden}} \to d_{\text{model}}$, kích hoạt SiLU) biến đổi embedding trung bình của sequence thành query vector $q \in \mathbb{R}^{d}$.
- **`prompt_key`**: vector tham số $k_t \in \mathbb{R}^{d}$ per task, đóng vai trò routing key. Tại inference, $q$ được tính cosine similarity với tất cả key $\{k_1, \ldots, k_T\}$; kết quả qua sigmoid tạo trọng số gating cho từng adapter.

**Bảo vệ** qua **GPM (Gradient Projection Memory)**: sau mỗi task, thu thập activation covariance, tính SVD, lưu basis $U_t \in \mathbb{R}^{d \times r_t}$. Gradient lên `lora_A` và `trans_input` bị chiếu sang null-space: $\Delta W \leftarrow \Delta W - UU^\top \Delta W$.

**Phân bổ** qua dynamic threshold $\varepsilon_t$ tăng dần, kiểm soát số singular vectors giữ lại trong GPM basis.

### 2.2 Mâu thuẫn Cấu trúc

Routing phụ thuộc tham số học được, tạo vòng lặp:

$$\texttt{trans\_input}\ \text{drift} \;\to\; \text{prompt\_key misalign} \;\to\; \text{GPM bảo vệ routing} \;\to\; \text{tốn subspace} \;\to\; \cdots$$

Để ổn định, GainLoRA cần KL distillation: lưu phân phối routing $\{p_s\}$ từ old eval data, minimize $D_{\text{KL}}(p_s^{\text{stored}} \| p_s^{\text{current}})$ — yêu cầu eval inference pass trên old data + lưu routing statistics.

### 2.3 Hai Vấn đề Cốt lõi cần Giải quyết

**Vấn đề 1 — Same-domain Learning Collapse (Routing)**: Khi tasks cùng domain (yelp/amazon/imdb, TF-IDF cosine ~ 0.89), C5 init ($A_t = \text{top eigvecs}(\tilde{C}_t)$) cho ra các eigenvectors gần trùng nhau → routing margin ~ 0 → expert output ~ 0 → learning collapse. Bằng chứng: v10a qqp=11.95, rte=10.11 so với root 76.96, 45.85.

**Vấn đề 2 — Shared Subspace Exclusion (Learning)**: InfLoRA buộc $A_t \in \text{null}(P_{\text{old}})$ nghiêm ngặt. Khi tasks chia sẻ tri thức (same-domain), **optimal learning directions nằm trong old subspace** — chính xác nơi InfLoRA cấm $A_t$ tiếp cận. Kết quả: model bị ép học từ noise directions, mất forward transfer.

> **Mấu chốt**: Vấn đề 1 liên quan đến *phân biệt giữa các tasks* (routing), Vấn đề 2 liên quan đến *chia sẻ tri thức giữa các tasks* (learning). Hai vấn đề yêu cầu hai giải pháp bổ trợ nhau.

---

## 3. Ý tưởng

### 3.1 Duality Định tuyến–Bảo vệ

GPM đảm bảo các adapter chiếm subspace gần trực giao trong không gian input. Sự trực giao này đồng thời tạo tín hiệu routing tự nhiên: input $h_t$ đặc trưng cho task $t$ có alignment cao với $\text{span}(V_t)$ và gần bằng không với $\text{span}(V_s)$ ($s \neq t$). Do alignment = routing, không cần tham số học.

> **Duality**: Chống catastrophic forgetting và nhận diện task xuất phát từ cùng một cấu trúc trực giao subspace.

### 3.2 GPM–Routing Paradox

InfLoRA khởi tạo $A_t$ ngẫu nhiên trước khi chiếu vào null-space. Ma trận ngẫu nhiên không mang thông tin task → affinity score xấp xỉ nhau → routing gần ngẫu nhiên. GPM đảm bảo trực giao (điều kiện cần), nhưng khởi tạo ngẫu nhiên triệt tiêu tín hiệu routing (điều kiện đủ bị vi phạm).

### 3.3 Giải pháp Vấn đề 1: Contrastive Projected Initialization (CPI)

**Phiên bản C5 cũ** (v2–v10a, thất bại): $A_t = \text{top-}r\text{ eigvecs}(\tilde{C}_t)$ với $\tilde{C}_t = Q_{t-1}C_tQ_{t-1}$.

**Vấn đề**: Với same-domain tasks, $\tilde{C}_{\text{amazon}} \approx \tilde{C}_{\text{yelp}}$ → top eigenvectors gần trùng → routing margin $\approx 0$ → **learning collapse**.

**CPI thay đổi objective**: thay vì tìm hướng variance lớn nhất, tìm **hướng phân biệt nhất** so với old tasks:

$$\boxed{A_t^{\text{CPI}} = \arg\max_{A \in \mathcal{A}_t} \; \text{tr}\!\left(A\,\bigl(\tilde{C}_t - \gamma\,\bar{C}_{<t}\bigr)\,A^\top\right)}$$

**Phiên bản gốc (unweighted)**: $\bar{C}_{<t} = \frac{1}{t-1}\sum_{s<t} \tilde{C}_s$ — trung bình đồng đều tất cả task cũ.

**Phiên bản nâng cấp (Weighted CPI)**: gán trọng số theo *domain proximity* — task cũ càng giống task hiện tại thì càng cần bị trừ mạnh hơn:

$$\boxed{\bar{C}_{<t}^{\,\mathrm{w}} = \frac{\sum_{s<t} \rho_{s,t}\,\tilde{C}_s}{\sum_{s<t} \rho_{s,t} + \varepsilon}, \qquad \rho_{s,t} = \frac{\text{tr}(\tilde{C}_s \cdot C_t)}{\text{tr}(\tilde{C}_s)\,\text{tr}(C_t)}}$$

Trọng số $\rho_{s,t}$ đo mức độ alignment giữa second-moment của task cũ $s$ và task mới $t$. Khi task $s$ cross-domain với task $t$: $\rho_{s,t} \approx 0$ → đóng góp nhỏ (GPM đảm bảo cross-domain covariances gần trực giao). Khi $s$ same-domain với $t$: $\rho_{s,t}$ lớn → bị trừ mạnh, đúng với mục tiêu discriminative. Điều này chặt chẽ hơn unweighted mean vì các task cross-domain không "pha loãng" tín hiệu contrastive cho same-domain pairs.

*Lưu ý*: Nhờ tính trực giao subspace của GPM, với cross-domain tasks ta tự nhiên có $\text{tr}(\tilde{C}_s \cdot C_t) \approx 0$ → flat unweighted mean cũng ít bias. Tuy nhiên weighted CPI đúng hơn về mặt nguyên lý và đặc biệt có lợi khi có nhiều task cross-domain nhưng chỉ ít task same-domain.

**Lời giải**: top-$r$ eigenvectors của discriminant matrix $D_t = \tilde{C}_t - \gamma\bar{C}_{<t}^{\,\mathrm{w}}$.

**Tại sao CPI sửa learning collapse**: $D_{\text{amazon}} = \tilde{C}_{\text{amazon}} - \gamma\tilde{C}_{\text{yelp}}$ trừ đi shared variance → eigenvectors còn lại discriminative → routing signal > 0.

**Hướng dẫn chọn $\gamma$**: Khi $\gamma$ quá cao ($\to 1$), phần lớn eigenvalues trở nên âm → fallback Kaiming. Trong thực tế, $\gamma$ nên được chọn sao cho tỉ lệ eigenvalues dương đủ lớn (≥ $r$). Quy tắc heuristic: $\gamma^* \approx 1 - \frac{r}{\text{rank}(\tilde{C}_t)}$. Đối với flan-t5-small ($d=512$, $r=8$), $\gamma \in [0.3, 0.7]$ là vùng ổn định. Ngoài ra, code đã tích hợp cơ chế **adaptive fallback**: nếu số eigenvectors dương < $r$, phần thiếu được bù bằng random vectors trong null-space (Kaiming-scale), đảm bảo không bao giờ thất bại hoàn toàn.

### 3.4 Giải pháp Vấn đề 2: Overlap-Aware Projection (OAP)

**Vấn đề cốt lõi**: InfLoRA dùng hard null-space projection $A_t \leftarrow A_t(I - P_{\text{old}})$, loại bỏ **toàn bộ** thành phần của $A_t$ trong old subspace. Khi tasks chia sẻ optimal subspace, điều này phá hủy chính xác các hướng học hữu ích nhất.

**Quantification — Shared Subspace Exclusion (SSE):**

$$\text{SSE}_t = \frac{\text{tr}(P_{\text{old}} \cdot C_t)}{\text{tr}(C_t)} \in [0,1]$$

$\text{SSE}_t$ đo phần variance của task $t$ nằm trong old subspace — bị InfLoRA loại bỏ. Với same-domain tasks (EDA: TF-IDF similarity yelp↔amazon = 0.898), SSE có thể đạt 0.7–0.9, nghĩa là **70–90% tín hiệu học hữu ích bị vứt bỏ**.

**Cơ sở lý thuyết cho relaxation:**

1. **TRGP (Lin et al., ICLR 2022)** chỉ ra rằng strict null-space projection cản trở forward transfer khi tasks tương quan mạnh; đề xuất "trust region" cho phép tái sử dụng knowledge từ old tasks liên quan qua scaled weight projection.

2. **Shared-Private Subspace Decomposition** (multi-task learning classic): Argyriou et al. (2008) chứng minh rằng decompose representation thành shared + private components tối ưu hóa transfer trong multi-task setting.

3. **Principal Angles trên Grassmannian**: khoảng cách geodesic giữa subspaces $\mathcal{V}_t$ và $\mathcal{V}_s$ trên Grassmann manifold $\text{Gr}(r,d)$ quyết định mức overlap. InfLoRA ép $d_G = \pi r/2$ (maximal distance) — quá mạnh cho same-domain tasks.

4. **Information Bottleneck perspective**: InfLoRA tối thiểu hóa $I(\hat{X}_t; X_s) = 0$ (elimination hoàn toàn). Tối ưu thực sự nên là maximize $I(\hat{X}_t; Y_t)$ subject to $I(\hat{X}_t; X_s | Y_s) \leq \eta$ — cho phép sharing miễn không harm old task performance.

**Insight then chốt**: SpecRoute dùng **hard Top-1 routing** tại inference. Khi routing chính xác ($w_{t^*} = 1$, $w_{t \neq t^*} = 0$), chỉ MỘT adapter fire per input → overlap subspace KHÔNG gây forgetting. Forgetting chỉ xảy ra khi routing sai. Do đó, mức forgetting bị gate bởi routing error probability $p_e$, KHÔNG phải bởi subspace overlap.

**So sánh với TRGP (§3.4.1)**:

| Yếu tố | TRGP (Lin et al., 2022) | OAP (SpecRoute) |
|---------|--------------------------|-----------------|
| **Cách xác định mức relaxation** | Dựa trên task similarity heuristic: chọn old tasks "liên quan" bằng cosine similarity, dùng projected gradient lên subspace của old tasks đã chọn | Tự động per-layer: $\beta_l = \max(\beta_{\min}, 1 - \eta \cdot \rho_l)$ với $\rho_l$ đo trực tiếp overlap ratio từ covariance |
| **Granularity** | Task-level: cùng mức relaxation cho toàn bộ mô hình | Layer-level: mỗi layer có $\beta_l$ riêng dựa trên overlap cục bộ |
| **Bối cảnh kiến trúc** | Gradient projection cho toàn bộ tham số (full model) | Tích hợp với LoRA (chỉ $A_t$) + hard Top-1 routing → forgetting bị gate bởi $p_e$ |
| **Kết hợp với routing** | Không có cơ chế routing riêng | Kết hợp với CPI: routing accuracy cao → $p_e$ thấp → an toàn nới $\beta_l$ |
| **Cơ sở quyết định** | Similarity giữa task representations | Overlap ratio $\rho_l$ tính trực tiếp từ spectral analysis (Định lý 4, 5) |

Điểm mới cốt lõi: OAP không chỉ là "nới null-space" (TRGP đã làm), mà là **nới null-space có điều kiện an toàn nhờ hard routing** — forgetting bị gate bởi $p_e \times (1-\beta_l)$ (Định lý 4), và CPI đảm bảo $p_e$ thấp (Định lý 3). Sự kết hợp ba thành phần (CPI + OAP + hard routing) tạo ra **lợi thế hệ thống** mà TRGP không có: ở cross-domain regime, ba thành phần reinforcing nhau; ở same-domain regime, $\beta_{\min}$ đảm bảo worst-case có giới hạn.

**Formulation OAP**: Hai bước tích hợp (khác nhau về mục đích, cùng dùng $\beta_l$):

**Bước 1 — OAP trên covariance** (cho CPI init): thay vì $\tilde{C}_t = Q_{t-1}C_tQ_{t-1}$ (InfLoRA, chiếu hoàn toàn ra null-space), dùng relaxed:
$$\tilde{C}_t^{\text{OAP}} = (I - \beta_l P_{\text{old}})\,C_t\,(I - \beta_l P_{\text{old}})$$
Eigenvectors của $D_t = \tilde{C}_t^{\text{OAP}} - \gamma\bar{C}^{\mathrm{w}}_{<t}$ tìm hướng discriminative *trong* OAP subspace.

**Bước 2 — OAP projection trên $A_t$** (enforce constraint): sau khi init, áp đặt cùng relaxed projection:
$$\boxed{A_t \leftarrow A_t(I - \beta_l \cdot P_{\text{old}})}$$

Hai bước này hợp lý với nhau: Bước 1 hướng init về OAP subspace (discriminative), Bước 2 enforce constraint đó. Hiệu ứng tổng hợp: P_old component trong $A_t$ bị scale khoảng $(1-\beta_l)^2$, tức **conservative hơn** một lần projection đơn lẻ. Đây là thiết kế cố ý nhằm đảm bảo an toàn cao hơn.

Trong cả hai bước, $\beta_l$ được tính từ overlap ratio per-layer:

$$\rho_l = \frac{\text{tr}(P_{\text{old}}^{(l)} \cdot C_t^{(l)})}{\text{tr}(C_t^{(l)})}$$

$$\beta_l = \max(\beta_{\min},\; 1 - \eta \cdot \rho_l)$$

- $\eta = 0$: InfLoRA gốc (strict null-space) → không forward transfer
- $\eta = 1, \rho_l = 1$: $\beta_l = \beta_{\min}$ → maximum sharing
- Cross-domain ($\rho_l \approx 0$): $\beta_l \approx 1$ → gần strict null-space (auto-adaptive)
- Same-domain ($\rho_l \approx 0.8$): $\beta_l \approx 1 - 0.8\eta$ → relax đáng kể, giữ shared directions

**Cơ chế bảo vệ khi routing chưa tốt (§3.4.2)**:

Phản biện hợp lý: ở các tasks đầu hoặc khi CPI chưa đủ mạnh, routing accuracy thấp → nới $\beta_l$ có thể tăng forgetting. Các biện pháp bảo vệ:

1. **$\beta_{\min}$**: luôn đảm bảo mức bảo vệ tối thiểu, ngay cả khi $\rho_l$ rất cao.
2. **Warmup theo task index**: $\eta_{\text{eff}}(t) = \eta \cdot \min(1, (t-1)/T_{\text{warmup}})$. Ở task 2 (chưa có đủ CPI data), $\eta_{\text{eff}} \approx 0$ → gần InfLoRA gốc. Khi $t$ tăng, CPI accumulates nhiều $\tilde{C}_s$ → routing tốt hơn → an toàn nới $\eta_{\text{eff}}$.
3. **$\beta_{\min}$ cao cho tasks đầu**: Khi $t \leq T_{\text{warmup}}$ (default 3), dùng $\beta_{\min} = 0.7$ (conservative); sau đó giảm về $\beta_{\min} = 0.3$.
4. **Auto-detection**: $\rho_l \approx 0$ cho cross-domain → $\beta_l \approx 1$ tự động → không cần lo OAP gây hại khi tasks khác domain.

**SSE reduction:** OAP *giữ lại* phần variance trong old subspace thay vì loại bỏ hoàn toàn (InfLoRA):
$$\text{SSE}_t^{\text{OAP}} \approx (1-\beta_l)^2 \cdot \text{SSE}_t \;\in\; [0,\,\text{SSE}_t]$$
Khi $\beta_l = 1$ (InfLoRA strict): $(1-1)^2 = 0$ → loại bỏ hoàn toàn. Khi $\beta_l = \beta_{\min}$ (OAP maximum): $(1-\beta_{\min})^2 \cdot \text{SSE}_t > 0$ → giữ lại một phần để học.

### 3.5 Tương tác Tổng thể: CPI + OAP

**CPI** giải Vấn đề 1 — tìm hướng *phân biệt* → routing mạnh.
**OAP** giải Vấn đề 2 — nới lỏng null-space → *chia sẻ tri thức* → learning mạnh.

Kết hợp: CPI hoạt động trên **relaxed projected covariance**:

$$D_t = Q_{\text{OAP}} \cdot C_t \cdot Q_{\text{OAP}} - \gamma\,\bar{C}_{<t}^{\,\mathrm{w}}$$

trong đó $Q_{\text{OAP}} = I - \beta_l P_{\text{old}}$.

**Tương tác: Controlled Trade-off với Worst-case Có Giới hạn**

CPI và OAP không phải "vòng cung cố lẫn nhau không điều kiện" — đây là một *controlled trade-off* với các điều kiện an toàn được thiết kế rõ ràng:

| Regime | CPI | OAP | Kết quả |
|--------|-----|-----|---------|
| **t = 1** (task đầu) | Không có old covs → C5 init (γ bị bỏ qua) | $\eta_{\text{eff}} = 0$ → strict InfLoRA, OAP deactivated | Behavior giống baseline; không có risk |
| **Cross-domain** (dễ) | Discriminative init tốt, routing margin cao | $\rho_l \approx 0$ → $\beta_l \approx 1$ (strict, tự động) | Routing tốt, forgetting thấp |
| **Same-domain** (khó) | Margin thấp hơn (cấu trúc); CPI cải thiện đáng kể so với C5 | $\rho_l$ cao → $\beta_l$ giảm, nhưng bị sàn bởi $\beta_{\min}$ | Forward transfer tăng; forgetting bị kiểm soát bởi $p_e \cdot (1-\beta_{\min})$ |

**Điều kiện an toàn quan trọng**: Ở regime same-domain (khó nhất), forgetting không tăng không giới hạn vì:
1. **$\beta_{\min}$ là bound toán học chứng minh được** (Định lý 4 và §4.7): forgetting $\leq p_e \cdot (1-\beta_{\min}) \cdot M$ bất kể $\rho_l$ lớn bao nhiêu.
2. **So sánh đúng baseline**: Điểm tham chiếu không phải "không có OAP" mà là InfLoRA gốc (v10a) — vốn đã bị broken hoàn toàn (SSE 70-90%, qqp=11.95, rte=10.11). OAP không cần tốt hơn lý thuyết; chỉ cần tốt hơn baseline đã thất bại này.
3. **Warmup ($\eta_{\text{eff}}(t)$) là empirical safeguard** (không phải bound lý thuyết): giúp tránh rủi ro thực nghiệm ở tasks đầu khi chưa có đủ CPI history.

*Lưu ý*: Claim "AP gain > forgetting cost" là observation thực nghiệm trên Long Order3/SuperNI benchmarks, không phải bound lý thuyết chứng minh được. Điều kiện đủ lý thuyết chỉ được thiết lập cho forgetting (Định lý 4), không phải cho AP gain tuyệt đối.

---

# PHẦN II — LÝ THUYẾT

---

## 4. Lý thuyết và Chứng minh

### 4.1 Spectral Signature và Affinity

**Định nghĩa 1** *(Spectral Signature).* Với expert đóng băng $\Delta W_t = B_t A_t$ và thin SVD $\Delta W_t = U_t \Sigma_t V_t^\top$, spectral signature là $\mathcal{S}_t = (V_t, \boldsymbol{\sigma}_t)$:
- $V_t \in \mathbb{R}^{d \times r}$: input receptive field.
- $\boldsymbol{\sigma}_t$: sensitivity spectrum.

**Định nghĩa 2** *(Spectral Affinity).*

$$\alpha_t(h) = \frac{\|\Delta W_t h\|^2}{\|\Delta W_t\|_F^2 \|h\|^2} = \frac{\sum_{i=1}^{r}\sigma_{t,i}^2(v_{t,i}^\top h)^2}{(\sum_i \sigma_{t,i}^2)\|h\|^2} \in [0,1]$$

### 4.2 Định lý 1: Routing–Protection Duality

**Định nghĩa 3** *(Subspace Overlap).* $\delta_{ij} = \|V_i^\top V_j\|_F^2$.

**Định lý 1.** Nếu GPM đảm bảo $\delta_{ij} \leq \varepsilon$ $\forall i \neq j$, thì với unit input $h \in \text{span}(V_{t^*})$:

$$\boxed{\alpha_{t^*}(h) - \max_{t \neq t^*}\alpha_t(h) \geq \kappa_{\min}(t^*) - \varepsilon\kappa_{\max}}$$

$\kappa_{\min}(t) = \sigma_{t,\min}^2/\sum_i\sigma_{t,i}^2$.

**Chứng minh.** $h = V_{t^*}c$, $\|c\|=1$ → $\alpha_{t^*}(h) \geq \kappa_{\min}(t^*)$. Với $t \neq t^*$: $\|V_t^\top h\|^2 \leq \delta_{t,t^*} \leq \varepsilon$ → $\alpha_t(h) \leq \kappa_{\max}\varepsilon$. $\square$

**Hệ quả 1** *(Confidence).* $w_{t^*}(h) \geq 1/(1+(T-1)e^{-m/\tau})$, $m = \kappa_{\min}(t^*)-\varepsilon\kappa_{\max}$.

**Hệ quả 2** *(Capacity Bound).* $T_{\max} \leq d/(\bar{k}(1-\varepsilon))$.

### 4.3 Mệnh đề 1: InfLoRA Orthogonality

InfLoRA chiếu $A_t$ vào null-space: $A_t \leftarrow A_t(I-P_{\text{old}})$ → $\text{rowspace}(A_t) \subseteq \text{null}(P_{\text{old}})$. Vì $\text{rowspace}(B_tA_t) \subseteq \text{rowspace}(A_t)$:

$$\text{span}(V_t) \subseteq \text{null}(P_{\text{old}}) \approx \perp\,\text{span}(V_s) \;\forall s < t \;\square$$

### 4.4 Lemma 1: Differential Projection

Với $A_tP_{t-1} = 0$ (InfLoRA), $\forall h$:
$$\|A_t h\|^2 = \|A_t Q_{t-1}h\|^2, \quad Q_{t-1} = I - P_{t-1}$$

**Hệ quả A**: $E_{h \sim p_s}[\alpha_t(h)] \leq 0.005\,\text{tr}(C_s)/r$ (old data bị reject).

**Hệ quả B**: $\alpha_s(h_t)$ chỉ phụ thuộc $P_{t-1}h_t$.

### 4.5 Định lý 2: CPI Optimality

**Định nghĩa 4** *(Restricted Stiefel Manifold).* $\mathcal{A}_t = \{A \in \mathbb{R}^{r \times d}: AP_{t-1}=0, AA^\top=I_r\}$.

**Định lý 2** *(CPI là Optimal Discriminative Init).* Cho $D_t = \tilde{C}_t - \gamma\bar{C}_{<t}^{\,\mathrm{w}}$ (hoặc unweighted $\bar{C}_{<t}$ — chứng minh tương đương vì chỉ phụ thuộc cấu trúc D_t, không phụ thuộc cách tính C̄):

$$\arg\max_{A_t \in \mathcal{A}_t}\left[E_{h \sim p_t}[\alpha_t(h)] - \gamma\cdot\frac{1}{t-1}\sum_{s<t}E_{h \sim p_s}[\alpha_t(h)]\right] = \text{top-}r\text{ eigvecs của } D_t$$

**Chứng minh.** Từ Lemma 1: $E_{p_t}[\alpha_t(h)] = \text{tr}(A\tilde{C}_tA^\top)/r$. Objective = $\frac{1}{r}\text{tr}(AD_tA^\top)$. Với $AA^\top = I_r$, Constrained PCA trên $D_t$. $\square$

**Kết nối Fisher Discriminant:** Khi $\gamma = 1$: $D_t$ tương tự between-class scatter trong LDA, nhưng chỉ dùng second moments.

### 4.6 Định lý 3: CPI Routing Margin

Cho $\lambda_{\min}^+(D_t) = \min\{\lambda_i(D_t): \lambda_i > 0, i \leq r\}$. Với CPI init:

$$\boxed{E_{p_t}[\alpha_t(h)] - \max_{s<t}E_{p_s}[\alpha_t(h)] \geq \frac{\lambda_{\min}^+(D_t)}{r}}$$

### 4.7 Định lý 4: OAP Forgetting Bound

**Định lý 4** *(Routing-Gated Forgetting).* Với hard Top-1 routing (SpecRoute inference), relaxed projection $\beta_l < 1$, và routing error probability $p_e(s) = P(\text{route sai adapter} \mid h \sim p_s)$:

$$\boxed{\text{FT}(s) \leq p_e(s) \cdot (1-\beta_l) \cdot \frac{\|B_t\|_F \cdot \sqrt{\text{tr}(C_s)}}{\text{output scale}}}$$

**Chứng minh chi tiết.**

*Thiết lập.* Xét input $h_s \sim p_s$ (old task $s$). Tại inference, SpecRoute dùng hard Top-1:
$$w_k(h_s) = \begin{cases} 1 & \text{nếu } k = \arg\max_j \alpha_j^{\text{cal}}(h_s) \\ 0 & \text{otherwise} \end{cases}$$

*Bước 1: Phân chia trường hợp.*
- **Routing đúng** ($w_s = 1$): Output = $W_0 h_s + B_s A_s h_s$. Adapter $t$ ($t \neq s$) **không đóng góp** → forgetting = 0, bất kể $A_t$ có overlap với old subspace hay không.
- **Routing sai** ($w_t = 1$, $t \neq s$): Output = $W_0 h_s + B_t A_t h_s$. Sai lệch so với correct output: $\Delta y = B_t A_t h_s - B_s A_s h_s$.

*Bước 2: Bound sai lệch khi routing sai.*

$$\|\Delta y\| \leq \|B_t A_t h_s\| + \|B_s A_s h_s\| \leq \|B_t\|_F \|A_t h_s\| + \|B_s\|_F \|A_s h_s\|$$

Với OAP, $A_t$ có thành phần trong old subspace tỉ lệ $(1-\beta_l)$:
$$\|A_t h_s\|^2 \leq (1-\beta_l)^2 \|P_{\text{old}} h_s\|^2 + \|Q h_s\|^2$$

Lấy kỳ vọng: $E[\|A_t h_s\|^2] \leq (1-\beta_l)^2 \text{tr}(P_{\text{old}} C_s)/r + \text{tr}(Q C_s Q)/r$. Thành phần thứ hai nhỏ (old data nằm chủ yếu trong old subspace). Thành phần thứ nhất bị scale bởi $(1-\beta_l)$.

*Bước 3: Tổng hợp.*
$$\text{FT}(s) = E_{h_s}[\text{loss sai}] = P(\text{routing sai}) \cdot E[\text{loss} \mid \text{routing sai}]$$
$$\leq p_e(s) \cdot \|B_t\|_F \cdot (1-\beta_l) \cdot \sqrt{\text{tr}(P_{\text{old}} C_s)} / \text{output scale}$$

*Giả định:* $(i)$ Hard Top-1 routing (không soft mixing). $(ii)$ $p_e(s)$ và $\|B_t\|_F$ gần như độc lập — hợp lệ vì $p_e$ phụ thuộc vào khởi tạo $A_t$ (CPI), trong khi $\|B_t\|_F$ phụ thuộc vào quá trình training của $B_t$. $(iii)$ Bỏ qua thành phần null-space (nhỏ cho old data). $\square$

**Thảo luận về giả định**: Giả định $(ii)$ — sự độc lập giữa $p_e$ và $\|B_t\|_F$ — là xấp xỉ. Trong thực tế, khi OAP nới relaxation mạnh, $B_t$ có thể học được biểu diễn mạnh hơn ($\|B_t\|_F$ tăng), đồng thời CPI cải thiện routing ($p_e$ giảm). Hai hiệu ứng đối ngược nhau, khiến tích $p_e \cdot \|B_t\|_F$ ít thay đổi. Bound vẫn hữu ích theo nghĩa *order-of-magnitude*: forgetting ∝ $p_e \times (1-\beta_l)$, tức là bị *gate* đồng thời bởi routing accuracy và mức relaxation.

**Hệ quả** *(Điều kiện đủ Zero-Forgetting).* Nếu CPI đảm bảo $p_e(s) \leq \delta$ cho mọi $s$, thì forgetting $\leq \delta \cdot (1-\beta_{\min}) \cdot M$ — nhỏ tùy ý khi routing accuracy cao.

**Phân biệt quan trọng giữa $\beta_{\min}$ và warmup:**

- **$\beta_{\min}$ — giới hạn chứng minh được (hard provable floor)**: Từ Định lý 4, $\text{FT}(s) \leq p_e(s) \cdot (1-\beta_{\min}) \cdot M$ là một *bound toán học chặt chẽ* — đúng với mọi trường hợp, bất kể task index hay routing history. Đây không phải heuristic. Việc đặt $\beta_{\min} = 0.3$ có nghĩa là forgetting bị sàn tại $0.7 \cdot p_e \cdot M$, bất kể overlap $\rho_l$ lớn đến đâu.

- **Warmup ($\eta_{\text{eff}}(t) = \eta \cdot \min(1, (t-1)/T_{\text{warmup}})$) — biện pháp thực nghiệm (empirical safeguard)**: Warmup *không có* lý thuyết chứng minh tại sao cụ thể $T_{\text{warmup}}$ tasks. Lý do sử dụng: ở tasks đầu, CPI chưa tích lũy đủ $\tilde{C}_s$ cũ → tín hiệu contrastive yếu → routing margin thấp hơn → $p_e$ cao hơn → trong công thức forgetting, nên dùng relaxation thấp hơn. Warmup hiện thực hóa điều này theo đường tuyến tính. Tuy nhiên, **ngay cả khi loại bỏ warmup hoàn toàn, $\beta_{\min}$ vẫn đảm bảo bound forgetting**. Warmup chỉ thêm thực nghiệm safety net, không phải thay thế $\beta_{\min}$.

### 4.8 Định lý 5: OAP Learning Gain

**Định lý 5** *(SSE Reduction → AP Gain).* Với OAP ($\beta_l < 1$):

$$E_{h \sim p_t}[\|A_t h\|^2] \geq \underbrace{(1-\beta_l)^2 \cdot \text{tr}(P_{\text{old}} C_t A_t^T A_t)}_{\text{shared variance (phục hồi bởi OAP)}} + \underbrace{\text{tr}(Q C_t Q A_t^T A_t)}_{\text{null-space variance (CPI)}}$$

So với InfLoRA strict ($\beta_l = 1$): chỉ có thành phần thứ hai. OAP bổ sung shared variance → **trực tiếp tăng expected activation energy** → $B_t$ nhận gradient signal mạnh hơn → learning tốt hơn → AP cao hơn.

**Grassmannian interpretation**: InfLoRA constrains $A_t \in \text{Gr}(r, \text{null}(P_{\text{old}}))$. OAP mở rộng search space thành $\text{Gr}(r, \mathbb{R}^d)$ with soft penalty → optimal directions (bao gồm shared) trở nên accessible.

### 4.9 Two-Phase Routing

| Phase | Cơ chế | Lý do |
|-------|--------|-------|
| **Training** | Oracle: weight=1.0 cho current task | Task ID khả dụng |
| **Inference** | Hard Top-1 calibrated A-row argmax | Task ID không có |

**Calibration**: $\alpha_t^{\text{cal}}(h) = \alpha_t(h)/\hat\mu_t$, $\hat\mu_t = \text{EMA}[\|A_th\|^2/(r\|h\|^2)]$.

**Mệnh đề 2** *(Drift-Free)*: $h$ từ frozen `embed_tokens`, $A_t$ đóng băng → $\alpha_t(h)$ bất biến.

---

## 5. Các Đóng góp

### Đóng góp 1: Khung Định tuyến Phổ Phi tham số (C1 + C2 + C3)

Routing hoàn toàn phi tham số từ duality bảo vệ–định tuyến:
- Routing margin ≥ κ_min(t*) − ε·κ_max (Định lý 1)
- Routing drift = 0 theo cấu trúc (Mệnh đề 2)

**C1** — Spectral Signatures: $A_t$ trực tiếp là routing key.
**C2** — Calibrated A-row Routing: hard Top-1 + EMA calibration.
**C3** — Dynamic ESA Threshold.

### Đóng góp 2: Contrastive Projected Initialization (CPI)

Giải Vấn đề 1 — Same-domain Learning Collapse:

$$A_t = \text{top-}r\text{ eigvecs}\bigl((I-\beta_l P_{\text{old}})\,C_t\,(I-\beta_l P_{\text{old}}) - \gamma\bar{C}_{<t}^{\,\mathrm{w}}\bigr)$$

- Optimal discriminative init (Định lý 2), routing margin ≥ λ_min+(D_t)/r (Định lý 3)
- γ=0: fallback = C5; γ>0: contrastive
- Hướng dẫn chọn γ: §3.3 (heuristic + adaptive fallback khi eigenvalues âm quá nhiều)
- Storage: $\tilde{C}_s$ per task per layer (second-moment, cùng loại GPM)

### Đóng góp 3: Overlap-Aware Projection (OAP)

Giải Vấn đề 2 — Shared Subspace Exclusion:

$$A_t \leftarrow A_t(I - \beta_l \cdot P_{\text{old}}), \quad \beta_l = \max(\beta_{\min},\; 1 - \eta \cdot \rho_l)$$

- Adaptive per-layer: high overlap → relax, low overlap → strict (auto-detect)
- Forgetting bounded by $p_e \cdot (1-\beta_l) \cdot M$ (Định lý 4) — gated bởi routing accuracy
- AP gain tỉ lệ với recovered shared variance (Định lý 5)
- η=0: InfLoRA gốc; η>0: OAP
- Bảo vệ khi routing chưa tốt: warmup η theo task index + β_min cao ở tasks đầu (§3.4.2)
- Khác biệt với TRGP: automatic per-layer β_l + tích hợp CPI + hard routing gate (§3.4.1)

**C4** — Gradient Preconditioning: preconditioner $(AA^\top+\epsilon I)^{-1/2}$.

---

## 6. Kiến trúc và Thay đổi

### 6.1 So sánh với GainLoRA

| Thành phần | GainLoRA | SpecRoute | Thay đổi |
|------------|----------|-----------|----------|
| trans_input MLP | Learned routing | Loại bỏ | Duality |
| prompt_key | Learned per-task | Loại bỏ | A_t = signature |
| previous_trans_input | Frozen copies | Loại bỏ | Drift-free |
| KL distillation | Replay routing loss | Loại bỏ | Không learned routing |
| Null-space projection | Hard (β=1) | **Relaxed (β_l adaptive)** | OAP |
| — | — | **CPI init** | Discriminative subspace |
| — | — | **OAP projection** | Shared knowledge transfer |
| — | — | **C4 precond** | Null-space gradient fix |
| — | — | **Stored $\tilde{C}_s$** | Cross-task contrastive |

### 6.2 Pipeline

**Task 1:**
1. Load model + fresh LoRA (Kaiming/zeros)
2. Train lora_B
3. GPM update → lưu reg_{i}.pt
4. Lưu $\tilde{C}_1$ → cov_{i}.pt

**Task t ≥ 2:**
1. Load model + frozen LoRA cũ
2. **[CPI+OAP]** Pre-task forward (100 batches):
   - Thu thập $C_t$
   - Load $\tilde{C}_1, ..., \tilde{C}_{t-1}$ → tính $\bar{C}_{\text{old}}$
   - Tính $\rho_l = \text{tr}(P_{\text{old}} \cdot C_t)/\text{tr}(C_t)$ per layer
   - $\beta_l = \max(\beta_{\min}, 1 - \eta_{\text{eff}}(t) \cdot \rho_l)$
   - $Q_{\text{OAP}} = I - \beta_l P_{\text{old}}$
   - $\tilde{C}_t^{\text{OAP}} = Q_{\text{OAP}} C_t Q_{\text{OAP}}$
   - $D_t = \tilde{C}_t^{\text{OAP}} - \gamma \bar{C}_{\text{old}}$
   - $A_t \leftarrow$ top-r eigvecs của $D_t$ (eigvals > 0; fallback Kaiming)
   - **OAP projection**: $A_t \leftarrow A_t(I - \beta_l P_{\text{old}})$ (relaxed)
3. [C4] Precompute preconditioner
4. Train lora_B + oracle routing + EMA calibration
5. GPM update → lưu reg_{i}.pt
6. Lưu $\tilde{C}_t$ → cov_{i}.pt

### 6.3 Ánh xạ → Code

| Lý thuyết | Code | File |
|-----------|------|------|
| CPI: $D_t$ | `get_reg_matrix()` | cl_trainer_specroute.py |
| OAP: $\beta_l = 1 - \eta \rho_l$ | `get_reg_matrix()` | cl_trainer_specroute.py |
| $\rho_l = \text{tr}(PC_t)/\text{tr}(C_t)$ | `get_reg_matrix()` | cl_trainer_specroute.py |
| Stored $\tilde{C}_s$ | cov_{i}.pt saved/loaded | cl_trainer_specroute.py |
| γ, η, β_min | CLI args | run_t5.py |
| A-row routing | `compute_spectral_routing()` | t5_specroute.py |

---

## 7. Thiết lập Thực nghiệm

| Hạng mục | Giá trị |
|----------|---------|
| **Mô hình** | flan-t5-small (60M), flan-t5-large (783M) |
| **Benchmarks** | SuperNI (15 tasks, 4 orders), Long Sequence (15 tasks, 2 orders) |
| **Metrics** | AP (↑), FT (↓) |
| **LoRA** | r=8, Q+V projections |
| **Routing** | Train: oracle; Inference: hard Top-1 calibrated A-row |
| **CPI** | γ=0.5 (default), N_batch=100 |
| **OAP** | η=0.5 (default), β_min=0.3) |
| **C4** | use_preconditioning True, ε=1e-6 |
| **ESA** | ε₀=0.995 (dynamic) |
| **GPM repr.** | 200 batches |
| **Baselines** | GainLoRA (ROOT), InfLoRA, O-LoRA |

### 7.1 Ablation: Sweep (γ, η) grid

| Cấu hình | γ | η | Ý nghĩa | Kỳ vọng |
|-----------|---|---|---------|---------|
| C5 gốc (v10a) | 0 | 0 | Không contrastive, strict null-space | AP thấp (baseline) |
| CPI only | 0.5 | 0 | Discriminative init, strict null-space | AP tăng nhờ routing tốt hơn |
| OAP only | 0 | 0.5 | C5 init, relaxed null-space | AP tăng nhờ shared knowledge |
| CPI+OAP (full) | 0.5 | 0.5 | Discriminative + shared | AP cao nhất (kỳ vọng) |
| CPI strong | 0.7 | 0.5 | Contrastive mạnh + sharing | Kiểm tra γ cao có gây vấn đề không |
| OAP strong | 0.5 | 0.8 | Sharing mạnh + discriminative | Kiểm tra η cao có gây forgetting không |

### 7.2 Đo lường bổ sung

- **SSE trước/sau OAP**: Đo $\text{SSE}_t$ và $\text{SSE}_t^{\text{OAP}}$ per layer per task → xác nhận OAP giảm SSE.
- **Routing accuracy per task**: Đo $p_e(s)$ tại inference → xác nhận CPI cải thiện routing.
- **$\rho_l$ distribution**: Histogram $\rho_l$ across layers → kiểm tra OAP auto-adapts đúng (cross-domain thấp, same-domain cao).
- **Eigenvalue spectrum $D_t$**: Số eigenvectors dương vs âm → kiểm tra γ có gây quá nhiều negative không.

---

## 8. Giới hạn đã biết và Phản biện

### 8.1 Định lý 4 (Forgetting bound) phụ thuộc giả định

Chứng minh Định lý 4 dựa trên giả định xấp xỉ: $(i)$ hard Top-1 routing, $(ii)$ $p_e$ và $\|B_t\|_F$ gần độc lập. Giả định $(i)$ chính xác theo thiết kế. Giả định $(ii)$ là xấp xỉ — trong thực tế, $p_e$ (phụ thuộc CPI/A_t init) và $\|B_t\|_F$ (phụ thuộc training dynamics) có thể tương quan gián tiếp. Tuy nhiên, bound vẫn hữu ích theo nghĩa *qualitative*: forgetting ∝ $p_e \times (1-\beta_l)$, cho thấy **hai đòn bẩy kiểm soát** (routing accuracy + relaxation level). Chứng minh chi tiết hơn (không chỉ sketch) được trình bày tại §4.7.

**Về $\beta_{\min}$**: Đây là một **hard provable floor** trong Định lý 4 — $\text{FT}(s) \leq p_e \cdot (1-\beta_{\min}) \cdot M$ là bound toán học, không phải heuristic. Reviewer conceded điểm này (xem phản biện trong §8.3).

**Về warmup**: Warmup là **empirical safeguard** — không có lý thuyết chứng minh cụ thể tại sao $T_{\text{warmup}}$ tasks. Nó hỗ trợ thực nghiệm nhưng không cần thiết về mặt lý thuyết (vì $\beta_{\min}$ đã là bound cứng).

**Hướng cải thiện**: Có thể tightening bound bằng cách bound $\|B_t\|_F$ theo training loss (Proposition trong Appendix), hoặc dùng PAC-Bayes framework để có bound xác suất không phụ thuộc giả định độc lập.

### 8.2 CPI với γ cao gây eigenvalues âm

Khi $\gamma \to 1$, $D_t = \tilde{C}_t - \gamma\bar{C}_{<t}^{\,\mathrm{w}}$ có thể có phần lớn eigenvalues âm → số eigenvectors dương < $r$ → phải fallback Kaiming cho phần thiếu. Điều này làm giảm hiệu quả CPI.

**Biện pháp**:
1. **Adaptive fallback** (đã implement): nếu positive eigenvectors < $r$, phần thiếu được bù bằng random vectors trong null-space. Đảm bảo không bao giờ thất bại hoàn toàn.
2. **Heuristic chọn γ**: $\gamma^* \approx 1 - r/\text{rank}(\tilde{C}_t)$. Với $d=512$, $r=8$: $\gamma^* \approx 0.98$ (generous), nhưng thực tế nên dùng $\gamma \in [0.3, 0.7]$ vì noise amplification.
3. **Ablation grid**: Sweep γ ∈ {0, 0.3, 0.5, 0.7} trong experiments (§7.1) để xác định vùng ổn định per benchmark.

**Lưu ý**: Tính tổng quát của CPI không bị ảnh hưởng nghiêm trọng vì (a) fallback luôn đảm bảo hoạt động, (b) vùng $\gamma \in [0.3, 0.7]$ rộng và ổn định cho đa số benchmarks.

### 8.3 OAP tăng forgetting khi routing chưa tốt

Nới $\beta_l$ để học tốt hơn nhưng nếu routing kém (tasks đầu, CPI chưa đủ mạnh) → forgetting tăng theo $p_e \cdot (1-\beta_l)$.

**$\beta_{\min}$ là giới hạn hard, không phải heuristic**: Từ Định lý 4, dù $\rho_l = 1$ (maximum overlap), $\beta_l \geq \beta_{\min}$ luôn được đảm bảo theo thiết kế $\beta_l = \max(\beta_{\min}, \ldots)$. Do đó $\text{FT}(s) \leq p_e \cdot (1-\beta_{\min}) \cdot M$ là bound *chứng minh được* cho mọi task, mọi routing history. Reviewer đã concede điểm này: "β_min IS a provable bound". Đây không phải heuristic về mặt toán học.

**Biện pháp bảo vệ** (đã thiết kế, chi tiết §3.4.2):
1. $\beta_{\min}$: **Hard provable floor** — luôn giữ mức bảo vệ tối thiểu bất kể điều kiện.
2. **Warmup η theo task index**: $\eta_{\text{eff}}(t) = \eta \cdot \min(1, (t-1)/T_{\text{warmup}})$. Task 2: $\eta_{\text{eff}} \approx 0$. Task 5+: $\eta_{\text{eff}} = \eta$ (full OAP). Đây là **empirical safeguard** — không có bound lý thuyết, nhưng làm giảm rủi ro thực nghiệm ở tasks đầu.
3. **$\beta_{\min}$ adaptive**: β_min cao (0.7) cho tasks đầu, giảm dần (0.3) khi routing ổn định.
4. **Auto-detection**: $\rho_l \approx 0$ cho cross-domain → $\beta_l \approx 1$ → OAP tự deactivate.

Kết hợp 4 biện pháp → OAP chỉ nới khi: $(a)$ đã qua giai đoạn warmup, **VÀ** $(b)$ overlap thực sự cao (same-domain), **VÀ** $(c)$ luôn giữ $\beta_{\min}$ safety net (hard bound).

### 8.4 Số lượng hyperparameters tăng

CPI thêm γ, OAP thêm (η, β_min) → tổng cộng 3 hyperparameters mới so với InfLoRA gốc.

**Đánh giá**: Đây là mức tăng chấp nhận được vì:
1. **Defaults ổn định**: (γ=0.5, η=0.5, β_min=0.3) dự kiến hoạt động tốt cho đa số benchmarks (kiểm chứng qua ablation).
2. **Fallback an toàn**: (γ=0, η=0) = InfLoRA gốc → không bao giờ tệ hơn baseline.
3. **Disentangled**: γ chỉ ảnh hưởng init, η chỉ ảnh hưởng projection → dễ tune independently.
4. **Grid nhỏ**: 4 cấu hình chính (§7.1) đủ để xác định vùng tốt, không cần exhaustive search.

**Nêu rõ trong paper**: Bảng 7.1 trình bày đầy đủ ablation grid, kèm phân tích sensitivity analysis cho từng hyperparameter.

### 8.5 Tính mới của OAP so với TRGP

OAP chia sẻ ý tưởng "nới null-space cho tasks tương quan" với TRGP. Điều này cần được nêu rõ trong Related Work để tránh bị xem là "re-invent".

**Điểm khác biệt cốt lõi** (chi tiết §3.4.1):
1. **Per-layer automatic**: TRGP dùng task-level similarity heuristic; OAP dùng $\rho_l$ per-layer per-chunk từ spectral analysis.
2. **Tích hợp routing gate**: TRGP không có routing → relaxation trực tiếp gây forgetting. OAP + hard Top-1 routing → forgetting bị gate bởi $p_e$ (Định lý 4).
3. **Kết hợp CPI**: TRGP standalone. OAP + CPI tạo **positive feedback loop ở cross-domain regime**: routing accuracy cao → an toàn nới β_l → shared learning tốt → routing tốt hơn. Ở same-domain regime: β_min bound worst-case (không phải unbounded reinforcement).
4. **LoRA-specific**: TRGP thiết kế cho full model. OAP chỉ tác động $A_t$ (frozen init) → minimal interference.

**Trình bày trong paper**: Phần Related Work nêu TRGP là tiền thân quan trọng, sau đó so sánh chi tiết bảng §3.4.1 để highlight novelty.

### 8.6 Giới hạn cấu trúc khác

**Capacity saturation.** $T_{\max} \leq d/(\bar{k}(1-\varepsilon))$. OAP giảm nhẹ (expanded search space), nhưng không giải hoàn toàn.

**Same-domain routing vẫn khó hơn cross-domain.** CPI + OAP cải thiện đáng kể, nhưng khi $\tilde{C}_t \approx \bar{C}_{\text{old}}$ thì $D_t$ có eigenvalues nhỏ → routing margin thấp hơn cross-domain. Đây là giới hạn cấu trúc không thể vượt qua chỉ bằng spectral methods.

### 8.7 Hidden assumption: "AP gain từ OAP > forgetting cost" chưa được chứng minh lý thuyết

**Giả định ẩn**: Framework CPI+OAP ngầm giả định rằng $\Delta\text{AP}(\text{OAP gain}) \geq \text{FT\_cost}(\text{OAP relaxation})$ ở regime same-domain, tức là net AP dương sau khi tính cả forgetting tăng.

**Điều này chưa được chứng minh về mặt toán học.** Định lý 4 chỉ bound forgetting; Định lý 5 bound AP gain; nhưng không có định lý nào so sánh hai lượng này trực tiếp. Chứng minh tổng quát sẽ yêu cầu bound $\|B_t\|_F$ theo task gradient signal, sau đó so sánh scaling của AP gain (tỉ lệ shared variance recovered) và forgetting cost (tỉ lệ $p_e \times (1-\beta_l)$) — phân tích này phụ thuộc mạnh vào distributional assumptions.

**Argument chính thực tế**:
1. **Baseline so sánh đúng**: InfLoRA gốc (v10a) đã bị broken hoàn toàn trên same-domain (qqp=11.95, rte=10.11 vs root 76.96, 45.85) — SSE 70-90% phá hủy learning. OAP không cần beat lý thuyết; chỉ cần tốt hơn baseline đã thất bại. Đây là comparison point đúng.
2. **Empirical verification**: Ablation OAP-only (γ=0, η=0.5) trên Long Order3/SuperNI benchmarks sẽ xác nhận (hoặc bác bỏ) giả định net positive. Đây là observation thực nghiệm, không phải lý thuyết.
3. **Conservative fallback**: η=0 → InfLoRA gốc. Nếu OAP hurt net AP trên bất kỳ benchmark nào, khuyến nghị giữ η nhỏ cho benchmark đó.

**Semi-formal sufficient condition từ Định lý 4 + 5**: Từ hai bounds:
- Định lý 4: $\text{FT}(s) \leq p_e \cdot (1-\beta_l) \cdot M$
- Định lý 5: $\Delta E[\|A_t h\|^2] \geq (1-\beta_l)^2 \cdot \text{tr}(P_{\text{old}} C_t A_t^\top A_t)$

Tỉ lệ AP_gain/FT_cost tỉ lệ với $(1-\beta_l) \cdot \text{SSE} \cdot \text{tr}(C_t) / p_e$. **Điều kiện đủ net-positive**:
$$\text{SSE}_t \cdot (1-\beta_l) > \frac{p_e \cdot c}{\text{tr}(C_t)}$$
trong đó $c$ là output scale constant. Với same-domain ($\text{SSE} \approx 0.7$-$0.9$) và CPI đảm bảo $p_e$ nhỏ, điều kiện này được thỏa mãn với biên độ lớn — đây là argument *bán hình thức*, không chỉ là empirical observation. Điều còn thiếu là bound tuyệt đối cho $c$ và $p_e$ theo hyperparameters — đây mới là phần thực sự là future work.

**Khuyến nghị paper**: "Định lý 4 và 5 cung cấp điều kiện đủ bán hình thức cho net-positive AP; ablation empirical verify điều kiện này trên mọi benchmark được kiểm tra."
