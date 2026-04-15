# SRT-Guided Warm Initialization (SGWI)
## Giải Quyết Single-Task Performance Trong Hard Routing

---

## 1. Bản Chất Vấn Đề

### Kiến trúc SRT+GainLoRA

```
SRT hard routing → mỗi task tại inference CHỈ dùng 1 adapter duy nhất
→ adapter PHẢI đủ mạnh standalone
→ không có cross-task blending để "cứu" adapter yếu
```

### Hiện trạng init LoRA

```python
# run_t5.py, lines 546-555 — CURRENT INITIALIZATION:
nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))   # Random A
# lora_B = zeros (standard LoRA convention)
# → ΔW = B*A = 0 → mỗi task bắt đầu từ ZERO, không có knowledge transfer
```

### Hệ quả

| Task | Samples | Init | Kết quả | Lý do |
|------|---------|------|---------|-------|
| CB | ~250 | Random | **3.57%** 🔴 | Quá ít data + zero init → adapter yếu |
| wic | ~1000 | Random | **57.99%** | Medium data + zero init → adapter trung bình |
| dbpedia | ~5000 | Random | **99.03%** ✅ | Nhiều data + clear signal → adapter mạnh |

**Insight cốt lõi**: Trong soft gating (baseline), CB lợi dụng adapter từ rte, mnli (cùng NLI). Trong hard routing, CB đứng một mình → cần init tốt hơn.

---

## 2. Ý Tưởng: SRT Dual-Purpose Framework

SRT đã compute signatures {μ_t, Σ_t} cho routing. **Cùng một hệ thống signatures này** có thể phục vụ hai mục đích:

```
SRT Signatures {μ_t, Σ_t}
    ├── PURPOSE 1: Routing (inference)     → hard one-hot → zero forgetting
    └── PURPOSE 2: Initialization (train)  → warm start  → better single-task
```

**Đây là tính thống nhất (unification) của framework**: cùng một không gian metric vừa định tuyến, vừa chuyển giao tri thức.

---

## 3. Phương Pháp: SGWI (SRT-Guided Warm Initialization)

### 3.1 Formulation

Khi task t đến, trước khi train:

**Bước 1**: Extract embedding signature từ frozen backbone (đã có sẵn từ SRT)
$$\mu_t = \frac{1}{n_t} \sum_{x \in \mathcal{D}_t} h_{\text{frozen}}(x)$$

**Bước 2**: Compute SRT distance tới mọi task trước đó
$$d(t, s) = \|\mu_t - \mu_s\|_{\Sigma_{\text{pool}}^{-1}} \quad \forall s < t$$

(Chính xác là Mahalanobis distance trong whitened space — metric đã chứng minh 99.99% accuracy)

**Bước 3**: Construct initialization weights
$$w_s = \frac{\exp(-d(t,s) / \tau)}{\sum_{s'<t} \exp(-d(t,s') / \tau)} \quad \forall s < t$$

trong đó τ là temperature (set từ data, xem Section 3.3).

**Bước 4**: SVD Fusion — tạo initial LoRA từ weighted combination
$$\Delta W_{\text{init}} = \sum_{s<t} w_s \cdot B_s A_s$$
$$U, \Sigma, V^T = \text{rank-}r\text{ SVD}(\Delta W_{\text{init}})$$
$$B_t^{(0)} = U\Sigma^{1/2}, \quad A_t^{(0)} = \Sigma^{1/2} V^T$$

**Bước 5**: Train LoRA_t trên data task t bắt đầu từ $(B_t^{(0)}, A_t^{(0)})$

### 3.2 Tại Sao SVD Fusion?

Không thể simply average B và A vì:
- Các LoRA adapter sống trong **khác subspace** (B_s, A_s có orientations khác nhau)
- Average(B_s) * Average(A_s) ≠ Average(B_s * A_s)

SVD Fusion giải quyết bằng cách:
1. Tổng hợp ở **weight space** (ΔW = BA), không phải parameter space
2. SVD tìm **principal subspace** tối ưu cho rank-r approximation
3. Đảm bảo rank constraint: B_t ∈ ℝ^{d×r}, A_t ∈ ℝ^{r×d}

### 3.3 Chọn Temperature τ

**Option A — Data-driven (recommended):**
$$\tau = \text{median}\{d(s, s') : s \neq s', s,s' \leq t-1\}$$

Lý do: median pairwise distance là scale tự nhiên của không gian task similarity.

**Option B — Nearest-only (simplest, τ → 0):**
$$s^* = \arg\min_{s<t} d(t,s)$$
$$B_t^{(0)} = B_{s^*}, \quad A_t^{(0)} = A_{s^*}$$

Không cần SVD, không cần τ. Chỉ copy LoRA từ task gần nhất.

**Option C — Uniform (τ → ∞):**
$$\Delta W_{\text{init}} = \frac{1}{t-1}\sum_{s<t} B_s A_s$$

Bình quân mọi task trước đó. Tốt khi không rõ task nào liên quan.

### 3.4 Edge Case: Task Đầu Tiên

Khi t = 1 (chưa có task trước):
- Fallback về random init hiện tại: kaiming(A), zeros(B)
- Đây chính xác là behavior hiện tại

---

## 4. Lý Thuyết: Tại Sao SGWI Hoạt Động

### 4.1 Task Similarity → LoRA Subspace Similarity

**Proposition (informal)**: Cho frozen backbone $f_\theta$, nếu hai tasks s, t có distributions gần nhau trong embedding space ($d(s,t) \approx 0$), thì optimal LoRA weights gần nhau.

**Intuition**: LoRA học $\Delta W$ sao cho $(W_0 + \Delta W) h \approx y$ cho samples $(h, y) \sim \mathcal{D}_t$. Nếu $\mathcal{D}_t \approx \mathcal{D}_s$ (distributions gần), thì optimal $\Delta W_t \approx \Delta W_s$.

**Formal sketch**: Giả sử loss $L_t(\Delta W) = \mathbb{E}_{(x,y)\sim\mathcal{D}_t}[\ell(f_{\theta+\Delta W}(x), y)]$. Bởi vì backbone frozen, $h = f_\theta(x)$ chỉ phụ thuộc vào input. Nếu $d(s,t) \approx 0$:
- Task distributions on $h$ gần nhau: $\mathcal{P}_t^h \approx \mathcal{P}_s^h$
- → Loss landscapes gần nhau: $L_t(\Delta W) \approx L_s(\Delta W) + O(d(s,t))$
- → Optimal points gần nhau: $\Delta W_t^* \approx \Delta W_s^* + O(d(s,t))$

### 4.2 Optimization Landscape Argument

LoRA training là non-convex optimization. Init point matters:
- Random init → có thể rơi vào bad local minimum
- Warm init từ related task → bắt đầu **gần** good minimum
- Gradient descent hoàn thiện từ đó → converge nhanh hơn + better optimum

Đặc biệt quan trọng cho **low-data tasks** (CB: 250 samples):
- Ít data → loss landscape noisy → dễ rơi vào bad minimum
- Warm init → bắt đầu từ vùng tốt → ít bị ảnh hưởng bởi noise

### 4.3 Tính Thống Nhất Với SRT

```
SRT Framework:
  ┌─── {μ_t, Σ_t} ───┐
  │                     │
  ├→ Routing: d(x,μ_t)  → which adapter to use (inference)
  │
  └→ Init:   d(μ_t,μ_s) → how to initialize adapter (training)
  
  CÙNG metric space, CÙNG signatures, CÙNG distance function
```

Đây là điểm đẹp nhất: không cần cơ chế mới, chỉ **tái sử dụng** SRT infrastructure cho mục đích khác.

---

## 5. So Sánh Approaches

### 5.1 Simplest: Nearest-Task Init (NTI)

```
Khi task t đến:
  s* = argmin_{s<t} d_SRT(t, s)
  LoRA_t.A ← LoRA_{s*}.A    # Copy from nearest
  LoRA_t.B ← LoRA_{s*}.B
  Train LoRA_t on D_t
```

**Pros**: Cực kỳ đơn giản, ~5 dòng code
**Cons**: Chỉ dùng 1 task, bỏ qua multi-source info

### 5.2 Balanced: SVD Fusion Init (SFI)

```
Khi task t đến:
  w_s = softmax(-d_SRT(t,s) / τ) cho mọi s < t
  ΔW_init = Σ_s w_s * B_s * A_s
  U, Σ, V^T = rank-r SVD(ΔW_init)
  LoRA_t.B ← U * Σ^{1/2}
  LoRA_t.A ← Σ^{1/2} * V^T
  Train LoRA_t on D_t
```

**Pros**: Multi-source, principled (SVD finds optimal subspace)
**Cons**: Cần compute SVD (nhưng chỉ 1 lần per task, negligible cost)

### 5.3 Comparison Table

| Method | Init | Routing | Knowledge Transfer | Forgetting |
|--------|------|---------|-------------------|------------|
| root_gainlora | Random | Soft gating | Implicit (at inference via blending) | ~0.77 |
| new_gainlora (current) | Random | Hard SRT | None | ~0.34 ✅ |
| **new_gainlora + NTI** | Nearest-task copy | Hard SRT | Explicit (at init from nearest) | ~0.34 ✅ |
| **new_gainlora + SFI** | SVD fusion | Hard SRT | Explicit (at init from all, weighted) | ~0.34 ✅ |

---

## 6. Implementation Sketch

### 6.1 Nơi Can Thiệp: run_t5.py

Sau block re-init LoRA (line 555) và trước khi training:

```python
# ========== SGWI: SRT-Guided Warm Initialization ==========
if cur_task_id > 0 and training_args.use_srt_router:
    print("=" * 50)
    print("[SGWI] SRT-Guided Warm Initialization")
    
    # 1. Load SRT signatures từ previous checkpoint
    srt_sig_path = os.path.join(prev_checkpoint, 'srt_signatures.npz')
    srt_router = SRTRouter(...)
    srt_router.load(srt_sig_path)
    
    # 2. Extract current task embedding stats
    #    (reuse frozen encoder already attached)
    mu_t = extract_current_task_centroid(model, train_dataloader)
    
    # 3. Compute distances to all previous tasks
    distances = {}
    for task_id, sig in srt_router.signatures.items():
        distances[task_id] = np.linalg.norm(mu_t - sig.mu)  # whitened L2
    
    # 4. Compute weights
    dist_array = np.array(list(distances.values()))
    tau = np.median(dist_array)  # data-driven temperature
    weights = softmax(-dist_array / tau)
    
    # 5. SVD Fusion Init per layer
    for j in range(num_layers):
        # Collect previous ΔW = B*A for this layer
        delta_W = sum(
            w * (B_s[j] @ A_s[j])  # B: (d, r), A: (r, d) → (d, d)
            for w, (B_s, A_s) in zip(weights, previous_lora_params[j])
        )
        # SVD → rank-r approximation
        U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
        U_r = U[:, :r]      # (d, r)
        S_r = S[:r]          # (r,)
        Vt_r = Vt[:r, :]    # (r, d)
        
        # Set initial LoRA weights
        model.encoder.block[j]...lora_B.data.copy_(U_r * S_r.sqrt())
        model.encoder.block[j]...lora_A.data.copy_(S_r.sqrt().unsqueeze(1) * Vt_r)
    
    print(f"[SGWI] Initialized from {len(distances)} tasks, "
          f"nearest={min(distances, key=distances.get)}, τ={tau:.2f}")
    print("=" * 50)
```

### 6.2 Dự Kiến Kết Quả Cho CB

**Trước (Random Init + Hard Routing)**:
- CB init = random → train 250 samples → 3.57%

**Sau (SGWI + Hard Routing)**:
- SRT distance: d(CB, rte) ≈ small, d(CB, mnli) ≈ small
- SGWI: init CB's LoRA ≈ 0.4*LoRA_rte + 0.35*LoRA_mnli + ...
- Train 250 samples → expected >> 3.57% (NLI knowledge already embedded)

---

## 7. Tại Sao Đây Là Phương Pháp Đúng

### 7.1 Tách Biệt Rõ Ràng

```
ROUTING (inference):  SRT hard routing  → task isolation → zero forgetting
TRAINING (per-task):  SGWI warm init   → knowledge transfer → better single-task
```

Hai cơ chế **hoàn toàn độc lập** — không ảnh hưởng lẫn nhau:
- SGWI chỉ thay đổi init point, KHÔNG thay đổi training process
- SRT routing chỉ hoạt động ở inference, KHÔNG ảnh hưởng training

### 7.2 Không Thêm Hyperparameter Mới

- τ = median(pairwise distances) — hoàn toàn data-driven
- Hoặc dùng NTI (nearest-task) → 0 hyperparameters

### 7.3 Zero Overhead At Inference

- SGWI chỉ chạy 1 lần per task trước training
- Inference path giống hệt: SRT route → LoRA forward
- Không thêm computation, memory, hay latency

### 7.4 Unified Framework

SRT cung cấp:
1. **Routing metric** → which adapter to activate
2. **Similarity metric** → how to initialize adapter
3. **Task geometry** → understanding why some tasks are hard

Tất cả từ **cùng một bộ signatures** {μ_t, Σ_t} — zero overhead.

---

## 8. Tóm Tắt

```
Vấn đề:
  Hard SRT routing → perfect isolation → zero forgetting
  Nhưng: mỗi adapter phải đứng 1 mình → cần init tốt
  Hiện tại: random init → adapter yếu cho low-data tasks (CB=3.57%)

Giải pháp — SGWI:
  Dùng SRT distances (đã có) → xác định related tasks
  → Warm init LoRA từ related task adapters (SVD fusion)
  → Train trên task data → adapter mạnh hơn
  
Kết quả kỳ vọng:
  AP: 78+ (vượt baseline) nhờ init tốt hơn
  Forgetting: ~0.34 (giữ nguyên, hard routing không đổi)
  CB: >>3.57% (NLI knowledge từ rte/mnli qua init)
```

---

## Appendix: Pseudo-code Đầy Đủ

```python
def sgwi_initialize(model, srt_router, cur_task_id, task_order, 
                     previous_lora_paths, lora_rank):
    """
    SRT-Guided Warm Initialization.
    
    Replaces random init with warm start from nearest previous tasks,
    using SRT distance metric for similarity computation.
    """
    if cur_task_id == 0:
        return  # First task: keep random init
    
    # 1. Get current task's centroid (from SRT, already computed)
    cur_task = task_order[cur_task_id]
    
    # 2. Compute distances to all previous tasks
    distances = []
    for s in range(cur_task_id):
        prev_task = task_order[s]
        d = srt_router.distance(cur_task, prev_task)  # whitened L2
        distances.append(d)
    
    distances = np.array(distances)
    
    # 3. Compute similarity weights (data-driven τ)
    tau = np.median(distances) if len(distances) > 1 else 1.0
    weights = scipy.special.softmax(-distances / max(tau, 1e-8))
    
    # 4. Load previous LoRA weights
    prev_loras = []  # list of (lora_A_dict, lora_B_dict)
    for s in range(cur_task_id):
        path = previous_lora_paths[s]
        A = torch.load(os.path.join(path, "lora_weights_A.pt"))
        B = torch.load(os.path.join(path, "lora_weights_B.pt"))
        prev_loras.append((A, B))
    
    # 5. SVD Fusion per layer
    for layer_name in get_lora_layer_names(model):
        # Weighted sum in weight space
        delta_W = sum(
            w * (B[layer_name + '.lora_B'] @ A[layer_name + '.lora_A'])
            for w, (A, B) in zip(weights, prev_loras)
        )
        
        # Rank-r SVD
        U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
        sqrt_S = torch.sqrt(S[:lora_rank])
        
        # Set init weights
        set_lora_weight(model, layer_name + '.lora_B', U[:, :lora_rank] * sqrt_S)
        set_lora_weight(model, layer_name + '.lora_A', sqrt_S.unsqueeze(1) * Vt[:lora_rank])
    
    nearest = task_order[np.argmin(distances)]
    print(f"[SGWI] Task '{cur_task}' initialized from {cur_task_id} tasks "
          f"(nearest='{nearest}', d={distances.min():.2f}, τ={tau:.2f})")
```
