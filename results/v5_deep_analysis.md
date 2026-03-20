# V5 Deep Analysis — SpecRoute Prototype Routing

## 1. Tổng quan

V5 AP(EM)=59.55, rất gần ROOT=59.70 (Δ=-0.15). Đây là cải tiến vượt bậc so với V2 (30.73) và V3 (27.66).

**Kết luận chính**: Prototype routing HOẠT ĐỘNG — giải quyết GPM-Routing Paradox, đưa performance từ ~50% ROOT lên ~100% ROOT.

---

## 2. Phân tích win/loss so với ROOT

### V5 thắng ROOT (7 tasks, avg +2.84):
| Task | Type | Δ EM | Nhận xét |
|------|------|-----:|----------|
| multirc | Reading comprehension | +9.92 | **Cải tiến lớn nhất**. Prototype routing chọn đúng LoRA cho task phức tạp |
| wic | Word sense disambiguation | +2.98 | V3 fail hoàn toàn, V5 vượt ROOT |
| rte | NLI (small) | +2.17 | V3 fail, V5 vượt ROOT |
| copa | Causal reasoning | +2.00 | Slight improvement |
| agnews | Topic classification | +1.37 | |
| qqp | Paraphrase detection | +0.87 | |
| boolq | Yes/No QA | +0.58 | |

### V5 thua ROOT (8 tasks, avg -2.77):
| Task | Type | Δ EM | Nhận xét |
|------|------|-----:|----------|
| yahoo | Topic classification | -7.62 | **Thua nhiều nhất**. Single-task quality gap |
| amazon | Sentiment (5-class) | -4.04 | Single-task quality |
| sst2 | Sentiment (binary) | -4.01 | Single-task quality |
| cb | NLI (tiny) | -3.57 | Both near-zero, CB inherently broken |
| yelp | Sentiment (5-class) | -1.37 | Small gap |
| imdb | Sentiment (binary) | -0.90 | Small gap |
| dbpedia | Topic classification | -0.49 | Negligible |
| mnli | NLI (large) | -0.15 | Negligible |

### Pattern Analysis

**V5 thắng ở**: tasks phức tạp (multirc, wic, rte, copa, boolq) — đây đều là tasks cần routing chính xác. Prototype routing từ embedding space phân biệt tốt hơn MLP routing cho các task có cấu trúc input khác biệt.

**V5 thua ở**: sentiment tasks (amazon, sst2, yelp, imdb) và topic classification (yahoo) — đây là các task V5 train_loss vẫn OK nhưng single-task quality thấp hơn ROOT. 

**Root cause**: SpecRoute KHÔNG có KL distillation loss (đã bỏ ở V2). ROOT dùng KL distill để transfer knowledge giữa tasks → sentiment tasks (giống nhau về domain) được benefit. SpecRoute dùng strict orthogonality (GPM) → mỗi task phải "learn from scratch" trong null-space → yếu hơn cho same-domain tasks.

---

## 3. Forgetting Pattern 

Average forgetting = -0.85 (rất thấp, tốt hơn expected cho 15-task).

**Forgetting cao nhất**:
- rte: -4.69 (trained 52.71 → final 48.01). rte được train sau qqp (task 7 sau task 6). multirc (task 13) và boolq (task 14) gây forgetting cho rte.
- yahoo: -2.30 (51.96 → 49.66). Yahoo bị multirc và boolq gây forgetting nhẹ.
- multirc: -1.46 (61.90 → 60.44). Chỉ boolq và wic sau multirc.

**Zero forgetting**: cb (0→0, never learned), copa (44→44), qqp (77.82→77.83), boolq (61.01→61.01), wic (58.46→58.46).

**Nhận xét**: GPM protection HOẠT ĐỘNG TỐT. Forgetting rất thấp. Vấn đề chính là SINGLE-TASK QUALITY, không phải forgetting.

---

## 4. CB Failure — Deep Dive

CB (CommitmentBank) = 0.00 EM suốt training.

**Dữ liệu**: 250 samples, 10 epochs, 8 steps/epoch = 80 total steps.
Loss: 5.25 → 3.63 (giảm ~31% nhưng eval_em=0% suốt).

**Tại sao fail?**
1. **Extreme low-resource**: 250 samples quá ít cho 3-class NLI task
2. **Epoch 10 vẫn chưa đủ**: Loss chưa converge (vẫn giảm ở step cuối)
3. **ROOT cũng gần-fail**: EM=3.57 (chỉ 2/56 test samples đúng)
4. **Task đặc thù**: CB answers = "entailment/contradiction/neutral" — 3 labels phức tạp, T5-small khó handle với 250 samples

**Giải pháp khả dĩ** (KHÔNG vi phạm zero-replay):
- Tăng epochs cho tiny datasets (ví dụ: epochs = max(10, 200/steps_per_epoch))
- Sử dụng weight decay thấp hơn cho tiny datasets
- **KHÔNG nên ưu tiên**: CB cũng fail ở ROOT, đây là limitation chung không phải của SpecRoute

---

## 5. Single-Task Quality Gap (Yahoo, SST2, Amazon)

Đây là vấn đề quan trọng nhất cần giải quyết.

### Yahoo (Δ = -7.62)
- V5 trained=51.96, ROOT final=57.28
- Train_loss=0.582 (moderate, not great)
- Yahoo là task 12/15, có 10000 samples → đủ data
- **Hypothesis**: Preconditioning + entropy regularization có thể đang interfere với learning cho large-scale topic classification. Hoặc orthogonality constraint quá strict → yahoo phải learn trong subspace nhỏ.

### SST2 (Δ = -4.01) 
- V5 trained=81.42, ROOT=85.21
- SST2 là task 9, after imdb (task 8 — cùng domain sentiment)
- GPM forces SST2's A ⊥ IMDB's A → SST2 phải learn trong null-space restricted
- ROOT's MLP routing cho phép SST2 share knowledge với IMDB → higher accuracy

### Amazon (Δ = -4.04)
- V5 trained=48.84, ROOT=52.05  
- Amazon (task 2) phải orthogonal với yelp (task 1) — cùng domain 5-class sentiment
- Tương tự SST2: strict orthogonality hurts same-domain tasks

### Kết luận: Nguyên nhân gốc là STRICT ORTHOGONALITY

ROOT không bắt buộc LoRA directions phải orthogonal (GPM chỉ protect, routing qua MLP).
SpecRoute + InfLoRA bắt buộc A_k ⊥ A_j (GPM on LoRA) → same-domain tasks phải learn trong subspace hạn chế → single-task quality giảm.

Đây là trade-off cơ bản: **strict orthogonality = low forgetting BUT lower single-task quality**.

---

## 6. So sánh theoretical expectations

Conversation summary ghi V5 expected AP(EM) = 40-55. Actual = 59.55 — **VƯỢT EXPECTATIONS**.

Prototype routing giải quyết đúng bài toán đã phân tích (GPM-Routing Paradox). Kế hoạch relaxed orthogonality từ V5 design (η=0.1) chưa rõ có được implement không — cần verify.

---

## 7. Đề xuất cho V6 (bám sát research_rule.txt: theory → weakness → solution)

### Weakness đã nhận diện
**Single-task quality gap do strict orthogonality** — V5 forgetting chỉ -0.85 (gần zero) nhưng mất -2.77 EM trung bình trên 8 tasks so với ROOT.

### Phân tích lý thuyết
Strict InfLoRA: A_new ∈ null(P_old) where P_old = Σ A_i A_i^T.
→ Remaining null-space shrinks: dim(null) = d − k·r (với k tasks, rank r).
→ Same-domain tasks (yelp↔amazon, imdb↔sst2) cần similar directions nhưng bị forced vào orthogonal subspaces.
→ B must compensate harder → lower quality.

ROOT avoids this: LoRA GPM protects but routing qua learned MLP → tasks CAN share LoRA capacity implicitly.

### Hướng V6 (đề xuất, cần phân tích thêm trước khi implement)

**Option A: Relaxed Orthogonality (KHÔNG có trong V5 — đã verify)**
- orthogonal_relaxation KHÔNG xuất hiện trong V5 script hay cl_trainer_specroute.py
- V5 dùng STRICT orthogonality (pure InfLoRA GPM)
- Đề xuất: A_new ∈ (1−η)·null(P_old) + η·P_old — cho phép small overlap
- η ∈ [0.05, 0.2]: keep forgetting low nhưng improve same-domain quality

**Option B: Task-Aware Learning Rate**
- Tiny tasks (CB, COPA) dùng higher LR hoặc more epochs
- Adaptive schedule: epochs = max(10, min_steps / steps_per_epoch)
- Simple, no theory change needed

**Option C: Prototype Quality Enhancement**
- Current prototype = running mean of frozen embeddings → may not discriminate well between similar tasks
- Could weight prototype by class-conditional means hoặc use PCA of embeddings
- Cần verify cosine similarity matrix giữa prototypes (diagnostic log)

### Priority
1. ~~Verify nếu orthogonal_relaxation có active trong V5~~ → **ĐÃ VERIFY: KHÔNG CÓ** (strict orthogonality)
2. **Option A**: Implement relaxed orthogonality — có tiềm năng lớn nhất
3. **Option B** cho CB/COPA: simple fix, no theoretical risk
4. **Option C** chỉ nếu Option A không đủ

---

## 8. Kết luận

V5 là milestone quan trọng: prototype routing **chứng minh GPM-Routing Paradox analysis đúng** và **giải quyết được 5/6 task failures**. AP ngang ROOT (59.55 vs 59.70) là kết quả excellent cho parameter-free routing (so với ROOT's learned MLP routing).

Bottleneck hiện tại chuyển từ **routing quality** (đã giải quyết) sang **single-task learning quality** (do strict orthogonality). Đây đúng với C2 analysis (single-task quality) đã thảo luận trước đó.

**Nếu giải quyết được single-task gap** (-2.77 avg trên 8 losing tasks), V6 có thể đạt AP(EM) ~62-63, vượt ROOT.
