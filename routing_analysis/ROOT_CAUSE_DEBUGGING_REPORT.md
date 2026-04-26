# ROOT CAUSE: Tại sao SRT Router có ~50% accuracy

## Tóm tắt

**Không có bug trong code `srt_router.py`** — implementation đúng về mặt toán học.
Root cause là **n/d ratio = 0.04** (160 samples / 4096 dims) làm ZCA whitening
trở nên **uninformative** (gần như identity matrix).

**Kết quả reference 97% đạt được nhờ `--whiten` flag** — fit ZCA một lần trên
tất cả 15 tasks (n/d = 0.59), KHÔNG phải từ `ShrinkageWhitenedRouter` incremental.

---

## Bằng chứng cụ thể

### 1. Reference files không nhất quán

| File | Có `Shrinkage_ReWhiten`? | `Shrinkage_ReWhiten` accuracy | `--whiten` flag? |
|------|--------------------------|-------------------------------|------------------|
| `result_llama_2_7b_hf_superNI.txt` | **KHÔNG** | N/A | Không có |
| `result_llama_2_7b_hf_whiten_superNI.txt` | **CÓ** | 97.09% | Có |

→ `ShrinkageWhitenedRouter` KHÔNG được test trong reference mà KHÔNG có `--whiten`.
→ 97% = đến từ `--whiten` preprocessing, không phải từ `ShrinkageWhitenedRouter` incremental.

### 2. Simulation: n/d = 0.04 vs n/d = 0.59

```
d=4096, n/task=160, n_total=2400
n/d per task = 0.0391
n/d total    = 0.5859
```

**Incremental ZCA (n/d=0.04, shrink=0.1):**
```
Eigenvalue range: [0.025, 8.26]
Condition number:  330

||mu1_whitened|| = 1.05
||mu2_whitened|| = 1.05   ← CÙNG NORM! Whitened centroids collapse
L2(whitened_centroid1, whitened_centroid2): 2.10
L2(raw_centroid1, raw_centroid2): 8.20
→ Whitened distance chỉ bằng 26% raw distance
```

**Global ZCA (n/d=0.59, shrink=0.1):**
```
Eigenvalue range: [0.026, 5.72]
Condition number:  218

||mu_w|| = [3.89, 3.89, 3.90, ... 3.92]  ← KHÁC NHAU
Min pairwise distance: 5.69
Max pairwise distance: 5.72
→ Khoảng cách discriminative
```

### 3. Ảnh hưởng của shrink_factor

| Shrink | Cond# (n/d=0.04) | L2 whitened dist |
|--------|-------------------|------------------|
| 0.1    | 330               | 2.10             |
| 0.5    | 37.6              | 2.80             |
| 0.9    | 5.06              | 5.91             |

Với shrink=0.9, condition number = 5 → ZCA gần identity → whitened distances
≈ raw distances → discriminative.

---

## Giải thích toán học

ZCA whitening transform: `W = V @ diag(1/√λ) @ Vᵀ`

- **Với n/d = 0.04:** Sample covariance Σ̂ = Σ_true + noise. Noise eigenvalues
  lớn bất thường. Ngay cả với shrink=0.1:
  - `λ_min` bị inflate bởi noise → `1/√λ_min` rất lớn
  - `λ_max` cũng bị ảnh hưởng
  - `W` khuếch đại các hướng noise → whitened space ≈ isotropic

- **Với n/d = 0.59:** Σ̂ gần Σ_true hơn. Noise eigenvalues tương đối nhỏ.
  `W` preserve được discriminative structure.

- **Đặc biệt:** Với n/d cực thấp, sau khi ZCA, tất cả centroids có norm
  gần bằng nhau (||mu_w|| ≈ cùng giá trị). Lý do: whitened norm =
  √(d) với isotropic noise, và discriminative signal quá yếu so với noise.

---

## Root Cause Chain

```
srt_router.py add_task() với n=160, d=4096
    → Σ̂_t có condition# ~100-1000
    → Ledoit-Wolf(Σ̂_t, 0.1) vẫn còn ill-conditioned
    → ZCA whitening W ≈ noise amplifier
    → s.mu = (mu_t - mu_pool) @ W.T
    → Whitened centroids collapse về cùng origin
    → metric_l2() trả về khoảng cách gần như bằng nhau
    → argmin ≈ random → ~50% accuracy
```

---

## Fix Options

### Option 1: Tăng shrink_factor lên 0.5–0.9 (ĐƠN GIẢN NHẤT)

```python
# Trong llama_gainlora.py hoặc run_llama.py:
SRTRouter(srt_metric_mode='hard', use_shrink=True, shrink_factor=0.9)
```

Với shrink=0.9:
- Condition number giảm từ 330 → 5
- Whitened space gần identity → preserve raw geometry
- Không cần thay đổi logic khác
- Trade-off: ít lợi ích từ whitening, nhưng vẫn tốt hơn random

### Option 2: Global ZCA fit-once (CHÍNH XÁC NHƯ REFERENCE 97%)

```python
# Trong SRTRouter.__init__ hoặc add_task() đầu tiên:
# Fit ZCA một lần trên tất cả tasks hiện có
def fit_global_zca_once(self, all_task_embs: dict):
    """Fit ZCA trên pooled covariance của TẤT CẢ tasks."""
    all_embs = np.vstack(list(all_task_embs.values()))
    mu = all_embs.mean(0)
    Xc = all_embs - mu
    cov = np.cov(Xc, rowvar=False, ddof=1)
    # Shrink
    d = cov.shape[0]
    trace = np.trace(cov)
    target = (trace / d) * np.eye(d)
    cov_shrunk = (1 - self.shrink_factor) * cov + self.shrink_factor * target
    # ZCA
    eigvals, eigvecs = np.linalg.eigh(cov_shrunk)
    eigvals = np.maximum(eigvals, 1e-8)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    self._W_zca = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    self._mu_global = mu

# Sau khi fit global, chỉ apply, KHÔNG refit
# (thay đổi logic add_task hard mode: bỏ dòng refit ZCA)
```

**Lưu ý:** Option 2 vi phạm zero-rehearsal principle vì cần tất cả embeddings
trước. Option 1 hoàn toàn incremental và zero-rehearsal compliant.

### Option 3: Kết hợp — Fit global ZCA cho task 1 (samsum)

Vì task 1 (samsum) có n/d=0.04 ngay từ đầu, có thể:
1. Sau khi train tất cả tasks, save tất cả embeddings
2. Hoặc dùng hidden states từ checkpoint sau khi train xong task đầu
3. Fit global ZCA một lần

---

## Recommendation

**Đề xuất: Tăng shrink_factor từ 0.1 lên 0.9**

```bash
# Trong run script:
SRTRouter(srt_metric_mode='hard', shrink_factor=0.9)
```

Đây là fix đơn giản nhất, không cần thay đổi kiến trúc, và vẫn
zero-rehearsal compliant. Với shrink=0.9, ZCA whitening gần như identity
transform → whitened distances ≈ raw L2 distances giữa centroids.

Hoặc nếu muốn chính xác như reference 97%:
1. Trích embeddings cho tất cả tasks từ frozen backbone
2. Fit global ZCA trên tất cả embeddings
3. Save mu_global và W_zca
4. Load vào SRTRouter thay vì fit incremental
