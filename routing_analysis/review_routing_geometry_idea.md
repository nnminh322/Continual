# Nhận xét: ROUTING_GEOMETRY_IDEA.md

> Đánh giá thẳng thắn về các giả thiết, idea, và kịch bản. Không nịnh, không dìm.

---

## 1. Giả thiết nền tảng: "Backbone frozen → embedding bất biến → Routing = bài toán phân loại phân phối tĩnh"

**Đây là giả thiết mạnh nhất và đúng nhất trong toàn bộ tài liệu.** Nó cho phép tách hoàn toàn routing ra khỏi training loop — một bước khái niệm rất sạch.

**Nhưng có gap quan trọng chưa được nói rõ:**

- Giả thiết này đúng khi routing dựa trên `h(x) = Pool(Embed(x))` — tức raw frozen embedding. Nhưng trong pipeline thực tế (GainLoRA / SpecRoute), LoRA adapter nằm **bên trong** encoder stack (Q, V projections). Nghĩa là output cuối cùng của encoder **CÓ** bị ảnh hưởng bởi LoRA weights, dù embedding layer ban đầu thì frozen.
- Tài liệu ngầm giả định rằng routing chỉ cần nhìn vào input embedding (trước khi đi qua encoder layers + LoRA). Điều này hợp lý nếu bạn dùng **embedding layer output** (word embeddings trung bình). Nhưng nếu bạn dùng **encoder output** (sau attention layers), thì embedding KHÔNG còn bất biến nữa — nó phụ thuộc vào LoRA weights hiện tại.
- Cần làm rõ: PSR router nhận `h(x)` từ **layer nào**? Nếu là word embedding layer → đúng bất biến nhưng có thể thiếu semantic information. Nếu là encoder output → giàu semantic nhưng mất tính bất biến.

> [!IMPORTANT]
> Đây là điểm mấu chốt cần clarify sớm. Nếu `h(x)` là embedding layer (trước attention), bạn đang route dựa trên **token-level statistics** chứ không phải **semantic representation**. Liệu token distribution đủ phân biệt 15 tasks không? Phase A sẽ trả lời, nhưng phải test đúng layer.

---

## 2. Mô hình PPCA (Low-rank Gaussian) cho task distributions

### Giả thiết: Mỗi task ~ Gaussian, có thể xấp xỉ bằng low-rank covariance

**Điểm mạnh:**
- Argument CLT cho average pooling là hợp lý — trung bình nhiều token embeddings thường sẽ có phân phối gần Gaussian.
- PPCA có nền tảng lý thuyết rõ ràng (Tipping & Bishop 1999), có closed-form, và storage hiệu quả.
- Kế hoạch test Gaussianity (Phase A2: Henze-Zirkler, Anderson-Darling, QQ-plots) là đúng hướng và đầy đủ.

**Điểm cần cảnh giác:**

1. **Multi-modality trong task.** Nhiều task trong benchmark KHÔNG đơn giản là single Gaussian:
   - `dbpedia` có 14 categories rất khác nhau (Company, Artist, Building, ...). Embeddings của chúng **rất có thể** tạo thành multi-modal distribution (nhiều cụm con).
   - `mnli` có 3 labels (entailment, contradiction, neutral) với ngữ nghĩa rất khác nhau.
   - `yahoo` có 10 topic categories.
   - Nếu task distributions là **mixture of Gaussians** thay vì single Gaussian, thì PPCA sẽ underfit — centroid $\mu_t$ không đại diện cho cluster nào cả, và eigenvalues sẽ inflate.

2. **Anisotropy bug quen thuộc của Transformer embeddings.** BERT-family (và T5) embeddings nổi tiếng bị anisotropic — phần lớn variance tập trung vào vài hướng chung (common directions) mà **tất cả tasks đều share**. Điều này có nghĩa:
   - Top eigenvectors $V_t$ có thể **giống nhau** giữa các tasks (vì chúng capture common direction, không phải task-specific direction).
   - PSR dựa vào $V_t$ để phân biệt same-domain tasks → nếu $V_t \approx V_s$ do anisotropy, PSR không tốt hơn nearest centroid.
   - **Mitigation:** Cần **center-and-whiten** global trước khi tính per-task PCA. Tài liệu chưa đề cập bước này.

3. **Rank $k$ sensitivity.** Tài liệu đề xuất sweep $k \in \{2,4,8,16,32,64\}$ nhưng lựa chọn $k$ phụ thuộc nặng vào task. Với CB (250 samples, d=512), $k > 1$ đã có thể overfit. Với dbpedia (14000 samples), $k=32$ có thể hợp lý. **PSR cần per-task adaptive $k$**, nhưng tài liệu dùng fixed $k$ cho mọi task.

---

## 3. PSR Framework — "Unified, subsumes tất cả"

### Giả thiết: PSR subsumes Nearest Centroid, Spectral Affinity, QDA, LDA

**Về mặt toán, claim này đúng.** Bảng special cases (Section 2.4) chính xác — khi bạn set $k=0$ thì được centroid, khi $\mu_t=0, \sigma_t^2 \to 0$ thì được spectral, v.v.

**Nhưng "subsumes" ≠ "tốt hơn" trong thực tế:**

1. **Overfitting risk.** PSR có nhiều thành phần hơn (mean + subspace + spectrum + penalty). Khi data ít, mô hình phức tạp hơn **không nhất thiết** route chính xác hơn mô hình đơn giản. Nearest centroid với 200 samples có thể **thắng** PSR-full vì ít bị estimation noise.
   - Đây chính xác là vấn đề bias-variance tradeoff — PSR có variance cao hơn do ước lượng nhiều tham số hơn.
   - Tài liệu có nhận thức vấn đề (RMT section, shrinkage), nhưng chưa đặt câu hỏi: **"Liệu Nearest Centroid + shrinkage đã đủ tốt rồi?"**

2. **Novelty claim N1 ("chưa có work nào phân rã routing error trong CL thành mean + subspace + spectral"):**
   - Trong CL community, claim này có thể đúng vì CL papers hiếm khi phân tích routing ở mức này.
   - Nhưng trong statistical classification / Bayesian decision theory, công thức KL decomposition giữa hai Gaussian là **textbook material** (Bishop 2006, Murphy 2012). Contribution nằm ở **áp dụng** nó vào CL routing, không phải ở bản thân công thức.
   - Cần cẩn thận khi viết paper: trình bày là "application of well-known theory to a new problem" chứ không phải "chúng tôi phát minh decomposition mới".

3. **Regime-Adaptive (N2) — tự chuyển giữa centroid-dominant và subspace-dominant:**
   - Claim này đúng **nếu** các tham số ($\mu_t, V_t, \Lambda_t, \sigma_t^2$) được ước lượng chính xác.
   - Trong thực tế, nếu $\sigma_t^2$ bị ước lượng sai (quá nhỏ hoặc quá lớn), cân bằng giữa các terms sẽ lệch → regime detection sai.
   - Cần sensitivity analysis cho $\sigma_t^2$ estimation method.

---

## 4. Contribution 2: RIAP — Routing Confidence → Null-space Relaxation

### Giả thiết: Routing confident → safe to relax null-space

**Ý tưởng coupling routing ↔ protection là thú vị**, nhưng có vấn đề logic:

1. **Hướng nhân quả ngược.** RIAP nói: "Nếu routing confident, relax projection." Nhưng thực tế:
   - Routing confident = PSR phân biệt được task = $D_{\text{KL}}$ lớn.
   - $D_{\text{KL}}$ lớn giữa tasks = embedding distributions **đã rất khác nhau** = GPM null-space CHƯA bị chật (vì tasks chiếm các vùng riêng biệt).
   - Khi GPM chưa bị chật, **không cần** relax. Relaxation chỉ cần thiết khi null-space exhaustion xảy ra — tức đúng lúc $D_{\text{KL}}$ giữa task mới và task cũ **nhỏ** (same-domain) → routing KHÔNG confident.
   - **Nghĩa là: RIAP muốn relax đúng lúc routing nói "đừng relax".** Đây là mâu thuẫn cấu trúc.

2. **Reformulation cần thiết.** Có thể RIAP nên được đặt lại là:
   - "Khi routing **uncertain** (low margin) → task mới giống task cũ → **cho phép share** subspace thay vì ép orthogonal"
   - Tức relax khi $D_{\text{KL}}$ **nhỏ**, không phải khi lớn. Ngược lại với công thức hiện tại $\beta \propto (1 - D_{\text{KL}}/\tau)$.
   - Hoặc: Cần phân biệt giữa "routing confident" và "task distant". Khi task cũ và mới gần nhau (low $D_{\text{KL}}$), ta vừa cần routing cẩn thận hơn, vừa có thể relax protection vì share subspace là hợp lý.

> [!WARNING]
> Công thức RIAP hiện tại ($\beta$ giảm khi $D_{\text{KL}}$ tăng) có vẻ ngược logic. Khi $D_{\text{KL}}$ lớn (task rất khác), null-space chưa chật, protection strict là OK và chi phí thấp. Khi $D_{\text{KL}}$ nhỏ (task giống), null-space bị chật, cần relax — nhưng lúc đó routing cũng kém nhất. Cần suy nghĩ lại hướng coupling.

---

## 5. Kế hoạch thí nghiệm (Phase A–E)

### Điểm mạnh

- **Rất có tổ chức.** Chia phase rõ ràng, mỗi phase trả lời một câu hỏi nghiên cứu cụ thể.
- **Theory-first.** Phase A (geometric EDA) chạy trước để validate giả thiết PPCA, rồi mới so sánh methods. Đây là cách tiếp cận đúng đắn.
- **Ablation đầy đủ.** Phase D1 (PSR component ablation) sẽ cho biết đóng góp thực sự của từng thành phần.

### Điểm cần lưu ý

1. **Phase C quá rộng.** Liệt kê ~20 classifiers (SVM, kNN, Random Forest, XGBoost, Matching Network, MAML...) tạo cảm giác "shotgun approach" — thử tất cả rồi xem cái nào win. Điều này mâu thuẫn với claim "theory-first, toán trước implement sau".
   - **Đề xuất:** Giữ Phase C1 (generative: LDA/QDA/PSR/Naive Bayes) + C2 (Linear SVM, Ridge/RLS) + C6 (Grassmann). Bỏ kNN, Random Forest, XGBoost, Matching Network, MAML — chúng không liên quan đến PSR theory và biến paper thành benchmark comparison.
   - Nếu PSR đã từ Bayes-optimal theory, thì beating Random Forest **không phải** contribution — và nếu Random Forest thắng PSR thì mới là vấn đề.

2. **Thiếu baseline quan trọng nhất: RLS (V11) trên cùng embeddings.** Bạn đã implement RLSRouter trong `t5_specroute.py`. Một so sánh trực tiếp PSR vs RLS trên **cùng** frozen embeddings (offline, không qua training loop) sẽ cực kỳ informative. RLS là discriminative, PSR là generative — so sánh này trả lời câu hỏi "generative vs discriminative" cho routing CL.

3. **Phase D5 (Task Order Sensitivity): "PSR bất biến theo task order".** Claim này đúng cho **offline** routing (tất cả signatures đã có). Nhưng trong **incremental** CL (Phase D6), task order ảnh hưởng đến **protection** (GPM null-space tích lũy), và protection ảnh hưởng gián tiếp đến routing accuracy qua expert quality. Cần phân biệt rõ: PSR bất biến, nhưng downstream effect (expert quality) không bất biến.

4. **Bures-Wasserstein (Phase B4) có vẻ thừa.** Cho point-vs-distribution, $W_2^2(\delta_h, \mathcal{P}_t) = \|h - \mu_t\|^2 + \text{tr}(\Sigma_t)$. Term $\text{tr}(\Sigma_t)$ là hằng số per task, nên $W_2$ degenerate thành $\ell_2$ distance to centroid (+ task-dependent constant). Không có thêm thông tin so với nearest centroid. BW chỉ thú vị khi so sánh **distribution vs distribution** (batch routing). Cần ghi rõ limitation này.

---

## 6. Câu hỏi meta quan trọng chưa được đặt

1. **PSR route chính xác hơn → nhưng liệu routing accuracy là bottleneck?** Từ experiment_versions.md, V5 (prototype routing) đã đạt AP ≈ ROOT với forgetting rất thấp (-0.85). Vấn đề còn lại không phải routing accuracy mà là **single-task learning quality** (sst2, yahoo, amazon kém hơn ROOT dù routing đúng). PSR có thể đạt 99% routing accuracy nhưng AP vẫn không cải thiện nếu bottleneck nằm ở expert quality.

2. **PSR lưu $\mu_t$ từ training data — vi phạm zero-replay?** V6 đã loại prototype routing vì lưu mean embeddings = lưu data statistics = vi phạm zero-replay. PSR lưu $(\mu_t, V_t, \Lambda_t, \sigma_t^2)$ — tất cả đều là data statistics. Nếu $\mu_t$ vi phạm zero-replay, thì PSR cũng vi phạm. Nếu **cho phép** lưu statistics, thì V5 prototype routing đã giải quyết xong rồi (AP ≈ ROOT).
   - Cần xác định rõ: **zero-replay = không lưu raw samples**, hay **zero-replay = không lưu bất kỳ data statistics nào**? Phần lớn CL literature cho phép lưu statistics (running mean, covariance), chỉ cấm raw sample replay. Cần cite rõ constraint definition.

3. **Complexity penalty term $\ln|\Sigma_t|$ có thể dominate sai.** Cho task có nhiều data (dbpedia: 14000 samples), variance tự nhiên lớn hơn → $\ln|\Sigma_t|$ lớn hơn → PSR "phạt" task đó. Còn task ít data (CB: 250 samples), variance ước lượng thấp hơn (underestimate do few samples) → $\ln|\Sigma_t|$ nhỏ → PSR "ưu ái" CB. Điều này có thể gây bias hệ thống.

---

## Tổng kết

| Khía cạnh | Đánh giá |
|-----------|----------|
| **Giả thiết frozen embedding** | Đúng và mạnh, nhưng cần clarify dùng embedding layer hay encoder output |
| **PPCA model** | Hợp lý cho first approximation, cần test multi-modality và anisotropy correction |
| **PSR "subsumes all"** | Đúng toán học, nhưng subsume ≠ superior trong practice (bias-variance) |
| **Novelty claim** | Contribution ở application, không phải ở công thức mới — cần framing cẩn thận |
| **RIAP coupling** | Ý tưởng thú vị nhưng hướng nhân quả có vẻ ngược — cần reformulate |
| **Kế hoạch thí nghiệm** | Tổ chức tốt, nhưng Phase C quá rộng và thiếu RLS baseline |
| **Bottleneck question** | Chưa chứng minh routing accuracy là bottleneck thực sự |
| **Zero-replay definition** | Cần clarify — nếu statistics OK thì V5 prototype đã đủ |
