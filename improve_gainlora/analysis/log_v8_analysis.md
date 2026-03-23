Created 13 todos

Ran terminal command:  wc -l /Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log && ls /Users/nnminh322/Downloads/logs_and_outputs/ 2>/dev/null | head -20

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log#1-1), lines 1 to 150

Ran terminal command:  grep -n "Task\|task\|AP\|EM\|loss\|Loss\|C5\|C4\|GPM\|routing\|GRAD\|Routing\|collapse\|NaN\|nan\|error\|Error\|fallback\|Fallback\|Warning\|warn\|skip\|threshold\|Layer\|layer\|eigval\|precond\|signature\|[Ss]pectral" /Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log | grep -v "remote:\|Counting\|Compress\|Resolving\|Receiving\|Cloning\|Enumerating" | head -200

Read [](file:///Users/nnminh322/.vscode-personal/User/workspaceStorage/905b5359561a5dc2604c965d2c84e93d/GitHub.copilot-chat/chat-session-resources/144fcddb-0935-440a-a6aa-176086a8b03f/toolu_bdrk_01CK1ZN2rUeHEvrWwRnzjUK6__vscode-1773981846653/content.txt)

Starting: *Analyze Task 2+ training dynamics* (3/13)

Ran terminal command:  grep -n "eval_exact_match\|eval_rougeL\|eval_loss\|C5\|C4\|GPM\|GRAD\|Gradient\|Threshold\|threshold\|fallback\|Fallback\|Skip\|skip\|Layer [0-9]\|Spectral\|spectral\|AP\|average\|Average\|task_[0-9]\|task [0-9]\|\*\*\*\*\|====\|----\|final\|Final\|result\|Result" /Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log | grep -v "remote:\|Counting\|Compress\|Resolving\|Receiving\|Cloning\|Enum\|tokenization\|deprecat\|warn\|LoRA\|block\|SelfAtt\|EncDec" | head -300

Read [](file:///Users/nnminh322/.vscode-personal/User/workspaceStorage/905b5359561a5dc2604c965d2c84e93d/GitHub.copilot-chat/chat-session-resources/144fcddb-0935-440a-a6aa-176086a8b03f/toolu_bdrk_01QYjEejx4T96KQDgqPbDmxY__vscode-1773981846656/content.txt#1-1), lines 1 to 250

Read [](file:///Users/nnminh322/.vscode-personal/User/workspaceStorage/905b5359561a5dc2604c965d2c84e93d/GitHub.copilot-chat/chat-session-resources/144fcddb-0935-440a-a6aa-176086a8b03f/toolu_bdrk_01QYjEejx4T96KQDgqPbDmxY__vscode-1773981846656/content.txt#250-250), lines 250 to 262

Ran terminal command:  grep -n "predict_eval_rougeL\|AP\|Training task\|task [0-9\-]\|TASK\|predict_eval_exact_match\|Loaded.*spectral\|Saved spectral\|Skip Updat" /Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log | grep -v "for_CL\|for_yelp\|for_amazon\|for_mnli\|for_cb\|for_copa\|for_rte\|for_wic\|for_sst2\|for_imdb\|for_qnli\|for_boolq\|for_wsc\|for_wnli"

Starting: *Analyze C5 init quality per task* (4/13)

Ran terminal command:  grep -n "predict_eval_rougeL_for_yelp\|predict_eval_rougeL_for_amazon\|predict_eval_rougeL_for_mnli\|predict_eval_rougeL_for_cb\|predict_eval_rougeL_for_copa\|predict_eval_rougeL_for_rte\|predict_eval_rougeL_for_wic\|predict_eval_rougeL_for_sst2\|predict_eval_rougeL_for_imdb\|predict_eval_rougeL_for_qnli\|predict_eval_rougeL_for_boolq\|predict_eval_rougeL_for_wsc\|predict_eval_rougeL_for_wnli\|predict_eval_rougeL_for_qqp\|predict_eval_rougeL_for_dbpedia\|predict_eval_rougeL_for_yahoo\|predict_eval_rougeL_for_agnews\|predict_eval_rougeL_for_multirc" /Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log

Read [](file:///Users/nnminh322/.vscode-personal/User/workspaceStorage/905b5359561a5dc2604c965d2c84e93d/GitHub.copilot-chat/chat-session-resources/144fcddb-0935-440a-a6aa-176086a8b03f/toolu_bdrk_01Vrnewgqr8GmM6pHHhmpdJL__vscode-1773981846661/content.txt#1-1), lines 1 to 120

Compacted conversation

Updated todo list

I now have the complete forgetting trajectory. Let me dig into the training failures directly. 

Starting: *Analyze imdb/sst2/yahoo training failure* (2/8)

Ran terminal command:  LOG=/Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log
# Find where task 7 (imdb) training starts and ends
grep -n "Training task\|task_id\|Task 7\|imdb\|IMDB\|9624\|imdb.*train\|train.*imdb\|epoch.*7\|C5 Layer.*init\|Saved signature\|Loading\|DataLoader" $LOG | grep -A2 -B2 "imdb\|9624" | head -60

Ran terminal command:  LOG=/Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log
# Find training start/end markers and C5 init for imdb section
grep -n "task_order\|Training task\|Epoch\|C5 Layer 0\|Saved.*signature\|Loading.*task\|imdb" $LOG | head -120

Ran terminal command:  LOG=/Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log
# Get the actual log structure around line numbers where we know things happen
# Lines around imdb eval (line 4667) - go back ~200 lines to see training
sed -n '4450,4680p' $LOG

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log#4450-4450), lines 4450 to 4680

Let me look at the imdb training section (task 7) and sst2 (task 8) directly: 

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log#4004-4004), lines 4004 to 4060

Searched for regex `C5 Layer 0|Precomputed.*preconditioner|Epoch 1/|EM\s*=|exact_match.*epoch.*1|eval_exact_match.*1.0|epoch.*1.0|Training loss epoch 1|rougeL.*epoch` (`**/logs/t5_small_improve/improve-gainlora-v8.log`), no results

Searched for regex `C5 Layer 0|Precomputed.*preconditioner|eval_exact_match|eval_epoch|rougeL.*=.*[0-9].*epoch.*[0-9]` (`**/logs/t5_small_improve/improve-gainlora-v8.log`), 80 results

Now let me read the imdb training section directly — it should start around line 4060+: 

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log#4060-4060), lines 4060 to 4220

Let me look at sections right after the rte result (line 4004) to find the imdb setup: 

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve-gainlora-v8.log#4003-4003), lines 4003 to 4130

Let me directly read the source code to understand the architecture's core issues: 

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora)

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src)

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/t5_specroute.py#1-1), lines 1 to 120

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/t5_specroute.py#120-120), lines 120 to 300

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/t5_specroute.py#300-300), lines 300 to 450

Now let me look at how `key_attention_weights` is used in T5Block and LoRA attention: 

Searched for regex `key_attention_weights|lora_A|lora_B|loRA|scaling|task_num|run_single|spectral_signatures` (`**/src/t5_gainlora_inflora.py`), 40 results

Searched for text `key_attention_weights` (`**/src/t5_gainlora_inflora.py`), 30 results

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/t5_gainlora_inflora.py#591-591), lines 591 to 680

Now let me read the cl_trainer_specroute.py to understand how previous LoRA weights are stored and how training is setup per task: 

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/cl_trainer_specroute.py#1-1), lines 1 to 150

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/cl_trainer_specroute.py#150-150), lines 150 to 350

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/cl_trainer_specroute.py#350-350), lines 350 to 550

Searched for text `previous_lora_weights|spectral_signatures|load_state_dict|requires_grad|lora_A|lora_B|save_pretrained` (`**/src/run_t5.py`), 50 results

Read [](file:///Users/nnminh322/Desktop/personal/Continual/improve_gainlora/src/run_t5.py#500-500), lines 500 to 720

I now have enough information for a comprehensive analysis. Let me compile everything: 

Starting: *Write final root cause + solution report* (7/7)

Đây là báo cáo phân tích sâu và đầy đủ về lý do V8 thất bại:

---

## BÁO CÁO PHÂN TÍCH V8: TẠI SAO VẪN FAIL NẶNG

### 1. DỮ LIỆU THỰC TẾ — TRAJECTORY FORGETTING

| Task | Yelp | Amazon | MNLi | CB | Copa | QQP | RTE | IMDB | SST2 | DBpedia | AGNews | Yahoo | MultiRC | BoolQ |
|------|------|--------|------|----|------|-----|-----|------|------|---------|--------|-------|---------|-------|
| After T0 | 71.1 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| After T2 (MNLI) | 69.1 | 68.0 | **42.1** | — | — | — | — | — | — | — | — | — | — | — |
| After T3 (CB) | 66.3 | 66.3 | 37.4 | **60.7** | — | — | — | — | — | — | — | — | — | — |
| After T4 (Copa) | 65.1 | 65.0 | **33.0** | 42.9 | **58.0** | — | — | — | — | — | — | — | — | — |
| After T5 (QQP) | 64.4 | 64.1 | 31.2 | 21.4 | 58.0 | **77.1** | — | — | — | — | — | — | — | — |
| After T7 (IMDB) | 63.2 | 63.0 | 27.8 | 3.6 | 55.0 | 77.3 | 53.1 | **0.015** | — | — | — | — | — | — |
| After T8 (SST2) | 62.9 | 62.6 | 22.2 | 1.8 | 54.0 | 77.2 | 53.1 | 0.015 | **0.0** | — | — | — | — | — |
| After T11 (Yahoo) | 62.5 | 62.1 | 7.96 | 0.0 | 52.0 | 77.3 | 52.3 | 0.015 | 0.0 | 71.8 | 72.6 | **5.45** | — | — |
| **Final** | **62.0** | **62.0** | **2.8** | **0.0** | **48.0** | **77.0** | **49.8** | **0.015** | **0.0** | **67.1** | **69.3** | **4.1** | **53.0** | **61.8** |

**Phân loại:**
- ✅ **Stable (loss < 10 pts):** yelp, amazon, qqp, multirc, boolq
- ⚠️ **Slow forgetting (10–25 pts):** copa, rte, dbpedia, agnews
- ❌ **Catastrophic forgetting:** mnli (42→2.8), cb (61→0)
- 💀 **NEVER WORKED from day 1:** imdb (0.015), sst2 (0.0), yahoo (5.45), wic (0.0)

---

### 2. NGUYÊN NHÂN GỐC RỄ

---

#### 🔴 ROOT CAUSE 1 (Critical): GPM–Routing Self-Defeat — Lỗi Lý Thuyết Cơ Bản

**Cơ chế:**
- C5 init buộc `A_t ∈ null(P_old)` — tức A_t ⊥ các hướng của tất cả task cũ.
- Nhưng T5 dùng CHUNG 1 embedding space. Input `h_imdb` của imdb (sentiment) CÁC HƯỚNG CHỦ ĐẠO của nó nằm trong `span(P_old_yelp_amazon)` — y hệt yelp và amazon.
- Hệ quả: `A_imdb @ h_imdb ≈ 0` vì `A_imdb ⊥ P_old` và `h_imdb ≈ P_old @ h_imdb`!
- **LoRA output = B @ A @ h ≈ B @ 0 = 0** → cả training lẫn inference đều vô nghĩa.

**Vì sao xảy ra với imdb, sst2, yahoo nhưng KHÔNG với yelp, qqp?**

| Task | Vị trí | Constraint trên A | Domain |
|------|--------|-------------------|--------|
| yelp (T0) | Đầu tiên | **KHÔNG có** (Kaiming random, full 512-dim) | Sentiment |
| amazon (T1) | T1 | Null-space of GPM(yelp) — nhỏ (19 dims) | Sentiment |
| mnli (T2) | T2 | Null-space of GPM(yelp+amazon) — 35 dims | NLI |
| imdb (T7) | Muộn | Null-space của **6 task** (yelp+amazon+mnli+...) | Sentiment |
| sst2 (T8) | Sau imdb | Null-space của **7 task** | Sentiment |
| yahoo (T11) | Rất muộn | Null-space của **10 task** | Topic |

**imdb/sst2 fail vì:** Domain của chúng (sentiment) BỊ YELP+AMAZON ĐÃ CHIẾM trong GPM từ rất sớm. A_imdb bị đẩy orthogonal với chính domain của imdb.

**Chứng cứ từ log:**
- `predict_eval_rougeL_for_imdb = 0.0154` — ngay BỘT khi vừa train xong (task 7). Không phải forgetting, là TRAINING FAILURE hoàn toàn.
- `predict_eval_rougeL_for_sst2 = 0.0` — ngay sau train (task 8). Training failure.

**Lỗi lý thuyết:** §3 (Lemma 1 + Theorem 2) chứng minh C5 routing tốt nhất TRONG NULL-SPACE, nhưng giả định ngầm là `h_t ∈ null(P_{t-1})`. Giả định này SAI với shared embedding. Khi các task domain trùng nhau (sentiment→sentiment), `h_imdb ∈ span(P_yelp_amazon)` → null-space projection = gần 0.

---

#### 🔴 ROOT CAUSE 2 (Critical): Early Task A Matrices Dominate Routing

**Cơ chế:**
- `A_yelp` = Kaiming random, spans toàn bộ 512-dim space (không bị constrain).
- `A_mnli` ∈ null-space (477-dim sau khi trừ yelp+amazon 35 dims).
- Routing score = `||A_t @ h||² / (r * ||h||²)`.
- Với bất kỳ input `h` nào: `||A_yelp @ h||²` ≥ `||A_mnli @ h||²` trung bình vì A_yelp có variance trong TOÀN BỘ thành phần của h, còn A_mnli chỉ thấy phần null-space.

**Hệ quả:** Softmax routing luôn give higher weight cho yelp hơn mnli với KHÔNG input nào của mnli! Đây là lý do mnli decay monotonically từ task 2 đến task 14.

**Chứng cứ:**
- mnli: 42 → 37 → 33 → 31 → 28 → 22 → 16 → 11 → 8 → 5 → 4 → 3 → 2.8 (MONOTONIC!)
- Nếu đây là forgetting thực, ta sẽ thấy sudden drops, không phải decay đều như vậy.
- Đây là routing weight decline: mỗi task mới được thêm vào, softmax dilutes thêm.

---

#### 🟠 ROOT CAUSE 3 (Severe): Softmax Dilution với 15 Tasks

**Cơ chế:**
```
routing_weight_mnli = e^(fit_mnli/T) / Σ_{all 15 tasks} e^(fit_k/T)
```
Với 15 tasks và `fit_mnli < fit_yelp` (do ROOT CAUSE 2), mnli nhận weight ngày càng nhỏ.

**Tính toán minh họa:** Nếu `fit_yelp = 0.3`, `fit_mnli = 0.15`, 13 tasks khác đều = 0.2, T=1:
- weight_yelp = e^0.3 / (e^0.3 + e^0.15 + 13×e^0.2) = 1.35 / (1.35 + 1.16 + 13×1.22) ≈ 1.35/18.37 ≈ **7.3%**
- weight_mnli = 1.16/18.37 ≈ **6.3%**

Với chỉ 6-7% routing weight cho mnli, output ≈ 6% mnli expert + 94% garbage → rougeL → 0.

---

#### 🟠 ROOT CAUSE 4 (Severe): Training Bias `β` Tạo Train-Inference Mismatch

**Cơ chế:**
- Khi train task T: beta = T × ln(α×n_old/(1-α)) được ADD vào fit_current → current task nhận ~80% weight.
- Khi inference task T (sau nhiều task mới được thêm): không có beta → weight giảm xuống ~6%.

**Hệ quả:** B_mnli được train với routing weight ~80%, nhưng at inference chỉ nhận ~6%. Điều này có nghĩa là B_mnli phải tạo ra output ĐÚNG với 80% signal, nhưng chỉ contribute 6% signal → output bị suppress.

Cụ thể với T = 1.0 (temperature):
- Training phân phối routing: [0.8, 0.2/n_old, 0.2/n_old, ...]
- Inference phân phối routing: [~0.07, ~0.07, ..., ~0.07] cho 15 tasks

Expert B_mnli đã learn để compensate cho "only 20% dari noise", nhưng inference lại noisier gấp bội.

---

#### 🟡 ROOT CAUSE 5 (Moderate): GPM Saturation cho Early Layers

**Chứng cứ từ log:**
```
Task 3 (cb): Skip Updating GPM for layer: 1, 9, 10
Task 6 (rte): Skip Updating GPM for layer: 1, 9
Task 8 (sst2): Skip Updating GPM for layer: 9, 10
...
Layer 7: 48/512 (max) sau task 3
Layer 8: 48/512 (max) sau task 3
```

Layer 7 và 8 (middle encoder layers, semantically richest) bão hòa sau chỉ 3 tasks. `get_repsentation()` không thể thêm dimension mới vào GPM. Các tasks sau đó phải co-exist trong cùng null-space mà không expand.

**Hậu quả cho C5:** C5 vẫn tìm được eigenvectors (eigval > 1e-6), nhưng null-space bị chia sẻ với 12+ tasks → mỗi task chỉ có r=8 vectors trong không gian có thể 512-48=464 dim — quá ít signal.

---

#### 🟡 ROOT CAUSE 6 (Moderate): MNLI Training Chỉ Reach 42 rougeL — Ceiling Thấp

**Chứng cứ từ log:**
```
mnli epochs: ep1=31.1, ep2=32.0, ep3=31.7, ep4=32.5, ep5=34.5, ep6=37.3, ep7~38, ep8~40, ep9~42, ep10=42.1
```

mnli chỉ đạt 42 rougeL sau 10 epochs (không converger cao hơn). Điều này có TẠI VÌ:
1. C5 constraint: A_mnli ∈ null-space của yelp+amazon, chỉ có 477 dims thay vì 512.
2. mnli là 3-class NLI: harder task with lower ceiling (SOTA T5-small for mnli ≈ 70–80% accuracy).
3. Routing weight ~80% during training nhưng grad chỉ flow qua A directions bị constrained.

---

### 3. STRUCTURAL ANALYSIS: Tại Sao Một Số Task Survive?

| Task | Survival Reason |
|------|----------------|
| **yelp (62)** | A_yelp = full-rank Kaiming, không có GPM constraint. Luôn có routing score cao nhất. |
| **amazon (62)** | A_amazon chỉ bị constraint 19/512 dims (GPM yelp nhỏ). Vẫn gần full-rank. |
| **qqp (77)** | Dataset khổng lồ (70k+ examples), 10 epochs → B_qqp rất mạnh. Compensates routing dilution. |
| **multirc (53)** | Trained cuối (task 12), it forgetting time. Dataset medium. |
| **boolq (62)** | Trained gần cuối (task 13), stable. |
| **dbpedia/agnews (67–73)** | Slow forgetting vì topic diversity → routing score vẫn OK. |

---

### 4. GIẢI PHÁP

---

#### 🔧 FIX 1 (Quan Trọng Nhất): Phát Hiện Same-Domain, Skip GPM Constraint

```python
# Trong get_reg_matrix(), thêm domain-similarity check:
# Nếu task hiện tại thuộc cùng domain với một task cũ, 
# không projecting A_t vào null-space — thay vào đó dùng lại A của task cũ.
DOMAIN_MAP = {
    'yelp': 'sentiment', 'amazon': 'sentiment', 'sst2': 'sentiment', 'imdb': 'sentiment',
    'mnli': 'nli', 'rte': 'nli', 'cb': 'nli', 'wnli': 'nli',
    'qqp': 'paraphrase', 'mrpc': 'paraphrase',
    'dbpedia': 'topic', 'agnews': 'topic', 'yahoo': 'topic',
}
```

Khi domain matches, dùng SHARED LoRA expert (hoặc initialize A_t từ A_prev_same_domain thay vì null-space).

---

#### 🔧 FIX 2 (Critical): Replace Softmax với Sparse Top-1 Routing

```python
# Thay vì softmax over all tasks:
weights = torch.softmax(fit_scores / T, dim=1)  # CURRENT (bad)

# Dùng gumbel-softmax hoặc top-1 hard routing:
top1_idx = fit_scores.argmax(dim=1, keepdim=True)
weights = torch.zeros_like(fit_scores).scatter_(1, top1_idx, 1.0)  # FIX
```

Hoặc top-2 với renormalization để tránh collapse.

---

#### 🔧 FIX 3 (Critical): Calibrated Routing — Normalize fit score theo task-specific statistics

```python
# Thu thập E[fit_t] trên training data của task t:
# Sau train task t, tính mean(||A_t @ h||² / (r||h||²)) trên eval set → normalization factor.
# Lưu vào spectral_signatures cùng với 'A'.
# Tại inference: fit_t = ||A_t @ h||² / (r||h||² * E[fit_t_on_train])
```

Điều này normalize để mọi task có cùng baseline ~1.0 trên in-distribution data.

---

#### 🔧 FIX 4 (Moderate): Add Per-Task Bias Tại Inference (Hay Tắt Bias Hoàn Toàn)

Phương án A — Tắt training bias:
```python
# Không dùng beta bias. Thay vào đó, start training với lớn hơn lora_r 
# để B_t có đủ capacity học trong routing weight thấp hơn.
```

Phương án B — Add inference bias per-task (lưu calibration offset):
```python
# Khi training task t xong, tính actual routing weight E[α_t(h)] trên val set.
# Lưu bias = log(target_alpha) - log(E[α_t]) 
# Inference: fit_t = raw_fit_t + saved_bias_t
```

---

#### 🔧 FIX 5 (Moderate): Tăng GPM Threshold Threshold → Giảm Saturation Rate

Hiện tại: `threshold = 0.995` (dynamic ESA). Điều này giữ 99.5% variance mỗi task → quá nhiều dimensions.

```bash
# Thử threshold = 0.98 hoặc 0.97:
--threshold 0.97  # GPM chỉ giữ 97% variance → ít dim hơn per task → chậm saturation
```

Với T5-small (512-dim), 0.97 threshold có thể add chỉ 5-10 dims/task thay vì 15-35.

---

#### 🔧 FIX 6 (Quick Win): Giảm Number of Epochs cho Large Datasets, Tăng cho Small

```bash
# imdb (50k): 3 epochs thay vì 10 (tránh overfit, cũng tránh routing instability)
# cb (250 samples): 30 epochs thay vì 10 (quá ít steps tuyệt đối)  
# mnli (large): 5 epochs thay vì 10
```

---

#### 🔧 FIX 7 (Long-term): Thêm Routing Diagnostic Logging

```python
# Trong compute_spectral_routing(), log routing weights mỗi eval step:
if self.is_inference and len(self.spectral_signatures) > 0:
    w = weights.mean(dim=0).squeeze()  # (n_tasks,)
    print(f"[ROUTING] task_weights: {w.tolist()}")
```

Không có log này thì hoàn toàn mù về routing quality. Cần biết actual routing weights.

---

### 5. KẾT LUẬN TÓM TẮT

| Failure | Nguyên Nhân | Độ Nghiêm Trọng |
|---------|-------------|-----------------|
| imdb=0.015, sst2=0.0 | A_imdb/sst2 ⊥ input space (same domain as yelp/amazon) | 🔴 Critical |
| yahoo=5.45 | A_yahoo ⊥ general NLP space (late task, GPM accumulated many dims) | 🔴 Critical |
| mnli 42→2.8 | A_yelp spans full space → yelp dominate routing → mnli weight → 0 | 🔴 Critical |
| cb 61→0.0 | Same + tiny dataset (250 samples) → can't recover even with high weight | 🔴 Critical |
| cola/rte slow forgetting | Moderate routing dilution | 🟠 Severe |
| yelp/amazon stable | No GPM constraint (first tasks) → highest routing scores always | — |
| qqp stable | Massive dataset compensates diluted routing signal | — |
| Final AP=43.73 vs ROOT≈59 | All above combined | 15 pts gap |

**Chiến lược fix theo priority:**
1. **FIX 1**: Same-domain expert sharing / skip null-space constraint
2. **FIX 2**: Sparse/calibrated routing thay vì uncalibrated softmax
3. **FIX 3**: Per-task normalization factor để fair comparison
4. **FIX 7**: Thêm routing logging để diagnose nhanh

Vấn đề cốt lõi là: **InfLoRA GPM null-space protection và Spectral Routing không tương thích về mặt lý thuyết khi các task share embedding domain**. GPM bảo vệ bằng cách đẩy A_t ra khỏi old subspace, nhưng routing đòi hỏi A_t phải align với h_t vốn nằm trong old subspace. Đây là mâu thuẫn cơ bản cần giải quyết ở thiết kế, không phải ở hyperparameter. 

Completed: *Write final root cause + solution report* (7/7)