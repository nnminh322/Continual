# Methodology Analysis: GainLoRA → SRT → SGWI+DualFisher

## Tóm tắt bức tranh toàn cảnh

### 1. GainLoRA gốc (root_gainlora) — Cơ chế GPM có vấn đề

**Kiến trúc gốc**: GainLoRA = Multi-LoRA adapters + Parametric Router (cal_attention) + GPM (Gradient Projection Memory)

**Vấn đề cốt lõi với GPM routing:**

1. **Parametric router drift**: Router `cal_attention` sử dụng sigmoid-based weights qua `trans_input` network. Network này được **cập nhật mỗi task** → thay đổi routing weights cho ALL previous tasks → **catastrophic routing drift**.

2. **GPM-learn coupling**: GPM vừa phải bảo vệ gradient subspace (anti-forgetting) VÀ đồng thời phải train router parameters. Hai mục tiêu này **mâu thuẫn nhau**:
   - Bảo vệ subspace → hạn chế không gian gradient
   - Train router chính xác → cần gradient tự do
   
3. **Replay dependency**: Cần replay labels (attention_weights.pkl) từ task trước → thêm memory overhead + replay labels bị outdated sau khi router drift.

4. **Thực nghiệm**: Routing accuracy giảm khi số task tăng (xem routing_analysis). Đặc biệt từ task 5+ trở đi, router thường route sai → adapter sai → forgetting tăng.

### 2. SRT (Statistical Routing Theory) — Contribution 1

**Phát hiện chính**: Routing là bài toán **phân loại thống kê**, không phải bài toán tối ưu parametric.

**Cơ chế SRT:**
```
Sau mỗi task t:
  1. Extract embeddings h_i từ frozen backbone (không train)
  2. Tính signature: μ_t = mean(h_i), Σ_t = cov(h_i)
  3. Lưu vào SRT Router (non-parametric)

Tại inference, cho sample x:
  1. Extract h_x từ frozen backbone
  2. Tính d(x, t) = (h_x - μ_t)^T Σ_t^{-1} (h_x - μ_t)  [Mahalanobis]
  3. Route sang adapter t* = argmin_t d(x, t)  [hard one-hot]
```

**Tại sao SRT triệt để hơn GPM routing:**

| Thuộc tính | GPM Routing | SRT |
|---|---|---|
| Parameter drift | ✗ Router parameters thay đổi | ✓ Signatures FROZEN sau khi tính |
| Zero-rehearsal | ✗ Cần replay labels | ✓ Chỉ cần {μ, Σ} |
| Accuracy vs #tasks | Giảm (drift tích lũy) | Ổn định (signatures không đổi) |
| Training cost | Phải train router | Zero cost (compute μ,Σ 1 lần) |

**Thực nghiệm xác nhận:**
- Routing accuracy ~100% trên cả Long_Sequence và SuperNI
- Full pipeline: forgetting giảm so với GainLoRA gốc
- Đặc biệt: task 5+ (nhiều tasks) → SRT vẫn route chính xác, GainLoRA gốc route sai

### 3. Trade-off: Anti-forgetting ↑ nhưng Knowledge Transfer ↓

**Vấn đề quan trọng bạn chỉ ra:**

SRT sử dụng **hard one-hot routing**:
- Training: w = [1, 0, 0, ...] → chỉ train current adapter
- Inference: w = one-hot(argmin distance) → chỉ dùng 1 adapter

→ **Mỗi task là một "silo" hoàn toàn độc lập.**

Điều này:
- ✓ **Maximizes anti-forgetting**: task cũ không bị ảnh hưởng bởi task mới (silos)
- ✗ **Minimizes knowledge transfer**: task mới không tận dụng được tri thức từ task cũ
  - Không có soft blending giữa adapters
  - LoRA init là random (Kaiming) → mỗi task bắt đầu từ scratch
  - Không có shared representation learning

**Đây là trade-off cơ bản trong Continual Learning:**
```
Stability (anti-forgetting) ←→ Plasticity (knowledge transfer)
```

SRT đẩy cực stability nhưng sacrifices plasticity.

### 4. SGWI + Dual Fisher — Contribution 2: Khôi phục Knowledge Transfer

**Insight cốt lõi**: SRT giải phóng GPM khỏi routing → GPM giờ chỉ cần bảo vệ gradient subspace → **còn rất nhiều "dư địa" (headroom) để đóng góp cơ chế transfer mới**.

#### 4a. SGWI (SRT-Guided Warm Initialization)

**Ý tưởng**: Thay vì init LoRA A/B random, dùng SRT distances để warm-init từ past tasks.

```
Cho task t mới:
  1. Extract embeddings cho task t
  2. Tính khoảng cách SRT d(t, t') cho mỗi past task t'
  3. Softmax weights: w(t') = exp(-d(t,t')/τ) / Σ exp(-d/τ)
  4. Weighted fusion: ΔW_init = Σ w(t') * ΔW(t')  [ΔW = B @ A]
  5. SVD decomposition: ΔW_init = USV^T
  6. Init: A_new = sqrt(S[:r]) @ V[:r], B_new = U[:,:r] @ sqrt(S[:r])
```

**Tại sao hoạt động:**
- Tasks tương tự nhau (gần trong SRT space) → ΔW có correlation cao → warm init giúp hội tụ nhanh hơn + performance tốt hơn
- Tasks khác nhau (xa trong SRT space) → weights ≈ uniform → warm init ≈ average → không gây hại
- SVD đảm bảo rank constraint của LoRA

**Knowledge transfer mechanism**: Transfer xảy ra ở **initialization**, không phải ở routing. → Không ảnh hưởng đến SRT hard one-hot routing (stability).

#### 4b. Dual Fisher Regularization

**Ý tưởng**: Embedding layer (embed_tokens) là shared across ALL tasks. Nếu nó drift quá nhiều → ảnh hưởng representations cho past tasks → hidden forgetting.

```
L_total = L_task + λ_emb * ||W_emb - W_emb^anchor||^2
```

Trong đó `W_emb^anchor` được snapshot trước khi train task mới.

**Tại sao cần:**
- LoRA chỉ modify attention Q/V → frozen
- Nhưng embed_tokens có thể drift (nếu nó trainable)
- Dual Fisher ngăn drift quá mức

**Kết hợp với SRT**: SRT signatures dựa trên embeddings. Nếu embed_tokens drift → signatures stale → routing sai. Dual Fisher giữ embeddings ổn định → SRT signatures vẫn valid.

### 5. Bức tranh tổng thể: 3-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Layer 3: ROUTING (SRT - zero drift)                      │
│   {μ_t, Σ_t} → Mahalanobis → hard one-hot              │
│   → Anti-forgetting: MAXIMAL                             │
├─────────────────────────────────────────────────────────┤
│ Layer 2: INITIALIZATION (SGWI - transfer via init)       │
│   SRT distances → softmax weights → SVD warm-init LoRA  │
│   → Knowledge transfer: via initialization               │
├─────────────────────────────────────────────────────────┤
│ Layer 1: REGULARIZATION (GPM + Dual Fisher)              │
│   GPM: gradient projection for trans_input, prompt_key   │
│   Dual Fisher: L2 penalty on embed_tokens                │
│   → Stability: gradient subspace protection              │
└─────────────────────────────────────────────────────────┘
```

**Mỗi layer giải quyết một vấn đề khác nhau:**
- SRT: "route đúng adapter" (Contribution 1)
- SGWI: "init adapter mới từ tri thức cũ" (Contribution 2a)
- Dual Fisher: "giữ ổn định shared representations" (Contribution 2b)

### 6. Điểm mạnh lý thuyết

1. **Orthogonal contributions**: SRT (routing) và SGWI (init) hoạt động ở 2 giai đoạn khác nhau → không mâu thuẫn
2. **Zero rehearsal**: Không cần replay data hay replay labels
3. **Non-parametric routing**: Routing không có learnable parameters → zero drift
4. **Principled transfer**: SRT distances là metric có ý nghĩa thống kê → transfer weights có cơ sở toán học

### 7. Rủi ro và hạn chế cần thừa nhận

1. **SGWI cold-start**: Task 0 và 1 không có past tasks → SGWI không hoạt động → cần 2+ tasks để thấy benefit
2. **SRT capacity**: Với rất nhiều tasks (>50), covariance matrices lớn → memory + compute tăng
3. **Hard one-hot giới hạn**: Không thể blend adapters cho samples nằm ở boundary giữa 2 task distributions
4. **Embed_tokens thường frozen**: Trong nhiều setup, embed_tokens không trainable → Dual Fisher vô nghĩa → cần kiểm tra setup cụ thể
5. **SVD overhead**: SGWI cần SVD cho mỗi LoRA layer → thêm ~5-10s mỗi task (acceptable)

### 8. Hướng nghiên cứu tiếp theo

1. **Soft SRT routing**: Thay vì hard one-hot, dùng softmax over SRT distances → blend adapters cho borderline samples → tăng transfer nhưng có thể giảm stability
2. **Adaptive τ**: Temperature τ trong SGWI softmax có thể learned hoặc adaptive per-task
3. **Fisher-weighted SGWI**: Dùng Fisher information để weight past tasks thay vì chỉ SRT distances → transfer quan trọng hơn ở "important" parameters
4. **SRT signature refinement**: Update signatures periodically mà không gây drift (exponential moving average)
