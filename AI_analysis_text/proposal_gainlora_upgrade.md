# Proposal: OT-SIGN — Statistical Signatures + Optimal Transport Routing for GainLoRA

---

## PHẦN 0: XÁC MINH KHẢO SÁT (Survey Verification)

**Kết quả: ✅ Toàn bộ thông tin khảo sát chính xác. Không cần sửa.**

| Paper | arXiv ID | Xác minh | Mô tả trong survey |
|-------|----------|---------|-------------------|
| Grassmannian MoE | 2602.17798 | ✅ Tồn tại | "Bingham distribution trên Grassmannian để control routing entropy" → ĐÚNG. Không phải CL. |
| Selective Sinkhorn Routing (SSR) | 2511.08972 | ✅ Tồn tại | "OT cho load-balancing token-to-expert" → ĐÚNG. Không phải distribution-matching. |
| Continual Pre-training of MoEs | 2503.05029 | ✅ Tồn tại | "Sinkhorn-balanced routing trong CPT context" → ĐÚNG. Nghiên cứu robustness của router, không phải CL với signature. |
| SCDEM | 2504.10561 | ✅ Tồn tại | "OT cho feature alignment (FDC), không phải routing" → ĐÚNG. Tên đầy đủ: Self-Controlled Dynamic Expansion Model. |

**Kết luận**: Bốn novelty gaps trong `novelty_search_report.md` vẫn giữ nguyên giá trị. Không có paper nào combine statistical signatures + OT distribution-matching routing + backbone anti-drift trong CL.

---

## PHẦN 1: VẤN ĐỀ CỦA GAINLORA HIỆN TẠI

### 1.1 Kiến trúc Gating Hiện Tại (từ `t5_gainlora_inflora.py`)

GainLoRA dùng cơ chế routing **key-query cosine attention**:

```
Bước 1: avg_inputs_embeds = weighted_mean(token_embeddings)  # shape (B, 1, d)
Bước 2: x = trans_input(avg_inputs_embeds)                   # 2-layer MLP → (B, 1, d)
         x = normalize(x)                                     # unit sphere
Bước 3: score_t = cosine_sim(x, prompt_key_t)                # scalar per task
         weight_t = |sigmoid(4 * score_t) * 2 - 1|
Bước 4: agg_lora = Σ_t  weight_t * lora_t(hidden_states)    # weighted sum
```

Với:
- `prompt_key_t ∈ R^d`: vector học được cho task t (learnable)
- `trans_input`: MLP 2 lớp (d → mlp_hidden → d, activation SiLU)

### 1.2 Ba Vấn Đề Cốt Lõi

**Vấn đề 1 — Routing không có nền tảng phân phối (Non-distributional routing)**

`prompt_key_t` là một **điểm trong không gian** (point estimate), không phải một **phân phối** trên không gian kiến thức của task t. Điều này có nghĩa:
- Routing chỉ đo khoảng cách đến một điểm đặc trưng duy nhất
- Không capture được độ rải hay hình dạng của không gian kiến thức (có task có features trải rộng, có task tập trung)
- Inputs ở boundary giữa hai tasks không được phân bổ một cách có nguyên tắc

**Vấn đề 2 — Gating weights không đảm bảo global optimality**

`weight_t = |sigmoid(4 * cos_sim) * 2 - 1|` là một hàm monotone **local** trên mỗi cặp (input, task). Không có ràng buộc global nào đảm bảo assignment là optimal trên toàn bộ batch hay toàn bộ expert set. Điều này dẫn đến:
- Expert utilization không balanced (một số LoRA experts bị underused)
- Không có theoretical guarantee về assignment quality

**Vấn đề 3 — Backbone drift không được kiểm soát tường minh**

Trong quá trình huấn luyện sequential, `trans_input` (MLP xử lý input) bị update cho task hiện tại nhưng không có cơ chế bảo vệ. Sau khi học $K$ tasks:
- `trans_input` có thể drift xa khỏi input features của các tasks cũ
- `prompt_key` của các tasks cũ được học cùng với `trans_input` cũ → bị misaligned với `trans_input` mới
- Kết quả: routing của tasks cũ kém chính xác dù LoRA weights vẫn được preserve

**Vấn đề 4 — Các experts không ngang hàng (Non-parallel feature spaces)**

Đây là vấn đề kiến trúc sâu hơn, ẩn trong cách GainLoRA xây dựng `past_x` (line 1305 của `t5_gainlora_inflora.py`):

```python
past_x = torch.cat([x, self.previous_trans_input(avg_inputs_embeds)], dim=1)
#                   ↑current task           ↑ N frozen snapshots (task_0, task_1, ...)
key_attention_weights = self.cal_attention(past_prompt_key, past_x)
```

`previous_trans_input` là một module chứa $t-1$ MLP riêng biệt, mỗi cái là **snapshot frozen tại thời điểm task đó được train**. Kết quả:

| Expert | Feature extractor | Feature space |
|--------|-----------------|--------------|
| Task 0 | `trans_input_frozen_at_t=0` | $\mathcal{F}_0$ |
| Task 1 | `trans_input_frozen_at_t=1` | $\mathcal{F}_1$ |
| Task $t$ (current) | `trans_input` (đang update) | $\mathcal{F}_t$ |

Routing tính **cosine similarity** giữa các vectors từ $N$ không gian khác nhau $\mathcal{F}_0, \mathcal{F}_1, \ldots, \mathcal{F}_t$ — so sánh này không có ý nghĩa hình học nhất quán. `prompt_key_i` được học trong $\mathcal{F}_i$ nhưng được dùng trong routing tại $\mathcal{F}_t$ → experts được đánh giá không công bằng, không phải do knowledge match mà do feature space mismatch. Thêm vào đó, memory overhead tăng tuyến tính: 15 tasks → 15 bản sao MLP.

---

## PHẦN 2: ĐỀ XUẤT CẢI TIẾN (GainLoRA → OT-SIGN)

### 2.1 Tổng Quan

Thay thế ba điểm yếu trên bằng ba thành phần tương ứng:

| Vấn đề | GainLoRA Hiện Tại | OT-SIGN Đề Xuất |
|--------|------------------|-----------------|
| Point routing | `prompt_key_t ∈ R^d` | vMF signature `(μ_t, κ_t)` |
| Local scoring | cosine sim → sigmoid | OT cost = vMF log-likelihood → Sinkhorn |
| No backbone protection | Không có | Anti-drift + Anti-invasion loss |
| Non-parallel experts | $N$ frozen `previous_trans_input` snapshots | 1 `trans_input` chung + signatures cùng không gian |

### 2.2 Component 1 — vMF Knowledge Signatures

**Thay thế `prompt_key_t ∈ R^d` bằng von Mises-Fisher signature `(μ_t, κ_t)`**

Sau khi huấn luyện xong task $t$, chạy một lần qua training data để collect:

$$\mu_t = \frac{\bar{x}_t}{\|\bar{x}_t\|}, \qquad \kappa_t = \frac{\bar{r}(d-1) - \bar{r}^3}{1 - \bar{r}^2}$$

với $\bar{x}_t = \mathbb{E}[\text{trans\_input}(x)]$ (mean direction sau MLP) và $\bar{r} = \|\bar{x}_t\|$ (mean resultant length). Đây là ước lượng MLE chuẩn của vMF (Banerjee et al., 2005).

**Tại sao vMF?**
- Features sau `normalize(trans_input(x))` nằm trên đơn vị hypersphere $\mathcal{S}^{d-1}$ → đúng domain của vMF
- vMF capture cả **hướng** (μ: trung tâm kiến thức) và **độ tập trung** (κ: task có diverse inputs có κ nhỏ, task tập trung có κ lớn)
- Chỉ lưu thêm $d + 1$ scalars so với $d$ scalars hiện tại (minimal overhead)

**Code integration** — thêm vào end-of-task hook trong `cl_trainer_gainlora_inflora.py`:

```python
def compute_vmf_signature(self, dataloader, model, task_id):
    """Chạy sau training mỗi task để fit vMF signature."""
    model.eval()
    all_x = []
    with torch.no_grad():
        for batch in dataloader:
            avg_emb = (batch['attention_mask'].unsqueeze(-1) * 
                       model.encoder.embed_tokens(batch['input_ids'])).mean(dim=1, keepdim=True)
            medium = model.encoder.trans_input[1](model.encoder.trans_input[0](avg_emb))
            x = model.encoder.trans_input[3](model.encoder.trans_input[2](medium))
            x = F.normalize(x.squeeze(1), dim=-1)  # (B, d)
            all_x.append(x)
    all_x = torch.cat(all_x, dim=0)
    
    x_bar = all_x.mean(0)                                    # (d,)
    r_bar = x_bar.norm()                                     # scalar
    mu_t = F.normalize(x_bar, dim=-1)                        # mean direction
    kappa_t = r_bar * (model.config.d_model - 1 - r_bar**2) / (1 - r_bar**2)
    
    model.encoder.vmf_signatures[task_id] = (mu_t.detach(), kappa_t.detach())
```

### 2.3 Component 2 — OT Distribution-Matching Routing

**Thay thế `cal_attention` (cosine sim) bằng Sinkhorn-OT với cost = vMF log-likelihood**

Với input feature $x_b$ (sau `trans_input`, normalized) và $N$ task signatures, tính cost matrix:

$$C_{bt} = -\kappa_t \cdot (\mu_t \cdot x_b) \quad \in \mathbb{R}^{B \times N}$$

(negative log-likelihood của vMF, bỏ constant term)

Sau đó chạy Sinkhorn OT (entropic regularization, $\varepsilon = 0.05$, 10 iterations):

$$\Pi^* = \text{Sinkhorn}(C, \varepsilon), \quad \Pi^* \in \mathbb{R}^{B \times N}, \quad \Pi^* \mathbf{1} = \mathbf{1}/B$$

`key_attention_weights` = $\Pi^* \in \mathbb{R}^{B \times 1 \times N}$ → đưa vào `agg_lora_states` y chang hiện tại.

**Code integration** — thay hàm `cal_attention` trong `T5Stack`:

```python
def cal_attention_ot(self, x, task_id=None):
    """
    x: (B, 1, d) — normalized input features
    Returns OT transport weights: (B, N_tasks, 1)
    """
    x = x.squeeze(1)  # (B, d)
    N = len(self.vmf_signatures)
    
    # Build cost matrix via vMF log-likelihood
    # C[b,t] = -kappa_t * (mu_t · x_b)
    mu_stack = torch.stack([sig[0] for sig in self.vmf_signatures.values()], dim=0)   # (N, d)
    kappa_stack = torch.tensor([sig[1] for sig in self.vmf_signatures.values()])       # (N,)
    kappa_stack = kappa_stack.to(x.device, dtype=x.dtype)
    
    dot_products = x @ mu_stack.T      # (B, N)
    C = -kappa_stack.unsqueeze(0) * dot_products   # (B, N)  — cost matrix
    
    # Sinkhorn iterations (log-domain for stability)
    weights = sinkhorn_log(C, epsilon=0.05, n_iter=10)  # (B, N)
    
    return weights.unsqueeze(2)  # (B, N, 1)  — same shape as current key_attention_weights

def sinkhorn_log(C, epsilon=0.05, n_iter=10):
    """Log-domain Sinkhorn — numerically stable."""
    log_a = torch.zeros(C.shape[0], device=C.device)  # uniform source (log 1/B)
    log_b = torch.zeros(C.shape[1], device=C.device)  # uniform target (log 1/N)
    log_K = -C / epsilon
    u = torch.zeros_like(log_a)
    for _ in range(n_iter):
        u = log_a - torch.logsumexp(log_K + u.unsqueeze(1), dim=1)
    v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)
    log_pi = log_K + u.unsqueeze(1) + v.unsqueeze(0)
    return log_pi.exp() * C.shape[1]  # normalize to sum=1 per row (B, N)
```

**Tại sao OT tốt hơn cosine sim?**
- Cost matrix encode "khoảng cách phân phối" — inputs gần vùng kiến thức task nào thì được route nhiều hơn đến task đó
- Sinkhorn constraints đảm bảo **global optimal assignment** trên cả batch
- OT weights tự nhiên sum to 1 → không cần normalization ad-hoc như `|sigmoid(...)*2-1|`
- Differentiable → gradients vẫn flow qua weights đến `trans_input` MLP

### 2.4 Component 3 — Backbone Anti-Drift Loss

**Thêm hai penalty terms vào training loop của mỗi task mới**

**Anti-drift loss** — bảo vệ `trans_input` khỏi drift trên replay data:

$$\mathcal{L}_{\text{drift}} = \frac{1}{|\mathcal{B}_{\text{replay}}|} \sum_{x \in \mathcal{B}_{\text{replay}}} \left\| \text{trans\_input}(x) - \text{trans\_input}_{\text{ref}}(x) \right\|^2$$

với `trans_input_ref` là frozen snapshot của `trans_input` sau nhiệm vụ $t-1$.

**Anti-invasion loss** — ngăn features của task mới "xâm chiếm" vùng của task cũ trong feature space:

$$\mathcal{L}_{\text{inv}} = \sum_{s < t} \max\left(0,\ \kappa_s \cdot (\mu_s \cdot x_{\text{new}}) - \tau \right)$$

với $x_{\text{new}}$ là features của task hiện tại, $(\mu_s, \kappa_s)$ là signature của task cũ $s$, và $\tau$ là threshold (VD: $\tau = -\log(0.1)$). Hàm này phạt khi features task mới có high likelihood dưới signature của task cũ.

**Tổng loss function:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{drift}} \mathcal{L}_{\text{drift}} + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}}$$

($\mathcal{L}_{\text{KL}}$ là replay loss đã có trong GainLoRA)

**Code integration** — trong `compute_loss` của `cl_trainer_gainlora_inflora.py`:

```python
# Anti-drift (thêm sau replay KL loss)
if self.args.anti_drift and self.ref_trans_input is not None:
    replay_avg = (replay_mask.unsqueeze(-1) * self.model.encoder.embed_tokens(replay_ids)).mean(1)
    x_curr = self.model.encoder.trans_input(replay_avg)  # F.normalize inside
    with torch.no_grad():
        x_ref = self.ref_trans_input(replay_avg)
    drift_loss = self.args.lambda_drift * F.mse_loss(x_curr, x_ref)
    loss = loss + drift_loss

# Anti-invasion (thêm với current task batch)
if self.args.anti_invasion and hasattr(self.model.encoder, 'vmf_signatures'):
    x_new = F.normalize(self.model.encoder.trans_input(avg_emb_curr), dim=-1)
    invasion_loss = 0.0
    for t_id, (mu_s, kappa_s) in self.model.encoder.vmf_signatures.items():
        if t_id < self.current_task_id:
            log_lik = kappa_s * (mu_s @ x_new.T).mean()
            invasion_loss += F.relu(log_lik - self.args.invasion_threshold)
    loss = loss + self.args.lambda_inv * invasion_loss
```

---

## PHẦN 3: ĐÁNH GIÁ KHẢ THI (Feasibility Assessment)

### 3.1 Tại Sao GainLoRA Là Candidate Tốt Nhất

Dựa vào code phân tích (`t5_gainlora_inflora.py`, `cal_attention`, `agg_lora_states`):

| Yếu tố | Đánh giá | Chi tiết |
|--------|---------|---------|
| Feature space đã normalized | ✅ Hoàn hảo | `x = x/x.norm()` ở line 1210 → trực tiếp trên $\mathcal{S}^{d-1}$ → vMF domain |
| Gating có weights scalar | ✅ Dễ thay | `key_attention_weights (B, N, 1)` feed vào `agg_lora_states` → chỉ cần output cùng shape |
| Multi-task keys structure | ✅ Sẵn có | `previous_prompts_keys` (N, d) → thay bằng `vmf_signatures dict` |
| Sequential training loop | ✅ Rõ ràng | End-of-task hook có thể thêm vào `cl_trainer` sau `save_model()` |
| lora_r=4 nhỏ | ✅ Không ảnh hưởng | Signature fit trên `trans_input` output (d=1024), không phải trên r=4 space |
| Memory overhead | ✅ Giảm đáng kể | Loại bỏ `previous_trans_input` (~15 × MLP size), thay bằng 15 × (d+1) floats cho signatures |
| Non-parallel expert problem | ✅ Giải quyết hoàn toàn | Loại bỏ `previous_trans_input`: tất cả experts dùng cùng `trans_input` → cùng feature space $\mathcal{S}^{d-1}$ |
| Sinkhorn on T4 | ✅ Khả thi | k=15 tasks, B=8, 10 iterations → <1ms/forward pass |
| Differentiable | ✅ | Log-domain Sinkhorn có gradients → không cần thay optimizer |

### 3.2 Thay Đổi Tối Thiểu Cần Làm

Chỉ cần modify **3 chỗ** trong codebase GainLoRA:

1. **`t5_gainlora_inflora.py → T5Stack.__init__`**: Thay `self.prompt_key` bằng `self.vmf_signatures = {}` + thêm `cal_attention_ot()` + `sinkhorn_log()`
2. **`t5_gainlora_inflora.py → T5Stack.forward`**: Thay `self.cal_attention(...)` bằng `self.cal_attention_ot(x)` sau khi signatures được loaded
3. **`cl_trainer_gainlora_inflora.py`**: Thêm `compute_vmf_signature()` call cuối mỗi task + thêm drift/invasion losses trong `compute_loss()`

Giữ nguyên hoàn toàn:
- `LoRALayer`, `agg_lora_states`, InfLoRA SVD projection
- KL distillation loss (replay)
- `trans_input` MLP architecture
- `previous_lora_weights_*` mechanism
- DeepSpeed / training infrastructure

### 3.3 Rủi Ro Thực Thi

| Rủi ro | Mức độ | Giải pháp |
|--------|--------|----------|
| κ estimation unstable (κ → 0 hoặc ∞) | Medium | Clip κ ∈ [0.1, 50]; fallback to cosine routing khi κ < 0.5 |
| Sinkhorn không converge với ε quá nhỏ | Low | Dùng ε = 0.05–0.1; log-domain stable |
| Anti-drift quá mạnh → catastrophic underfitting | Medium | Schedule λ_drift decreasing, bắt đầu từ 0.01 |
| vMF fit trên lora_r=4 features (nếu fit ở wrong level) | Low | **Fit trên trans_input output (d=1024), không phải LoRA factors** |
| T5-Large + 15 tasks + signatures + Sinkhorn OOM | Low | Signatures chỉ 15×1025 floats ≈ 60KB; Sinkhorn là matrix ops không grow model size |

---

## PHẦN 4: TÓM TẮT ĐÓNG GÓP

### Điểm Khác Biệt So Với Các Paper Liên Quan

| Paper gần nhất | Điểm khác biệt |
|-----------|----------------|
| GrMoE (2602.17798) | GrMoE: Bingham kiểm soát **routing entropy** (sparsity). OT-SIGN: vMF mô tả **knowledge region** của expert. GrMoE không phải CL, không có anti-invasion. |
| SSR (2511.08972) | SSR: OT cho **load balancing** (cost = learned linear score). OT-SIGN: OT cho **distribution matching** (cost = vMF log-likelihood). Semantics hoàn toàn khác. |
| SCDEM (2504.10561) | SCDEM: OT cho **feature alignment** giữa epochs (FDC). OT-SIGN: OT như **routing mechanism** để chọn expert. |
| PASs-MoE (2601.13020) | PASs-MoE: subspace methods cho router alignment. OT-SIGN: statistical signatures + global OT assignment. |

### Contribution Claim

> *OT-SIGN là framework đầu tiên sử dụng von Mises-Fisher distributions như fingerprint của knowledge region của từng expert module trong modular continual learning, đồng thời thay thế heuristic gating bằng Optimal Transport với semantic cost matrix (vMF log-likelihood), kết hợp với anti-drift và anti-invasion losses để bảo vệ shared representation space.*

---

*Analysis date: based on GainLoRA codebase + survey verification against arXiv 2024-2026*
