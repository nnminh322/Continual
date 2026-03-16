# PHÂN TÍCH PHÊ BÌNH VÀ HỆ THỐNG HÓA Ý TƯỞNG TỪ DISCUSSTION.TXT
## Từ lập luận thô → Kiểm chứng → Phản biện → Đề xuất phương pháp luận

**Ngày**: 9 tháng 3, 2026  
**Phương pháp**: Trích xuất các ý tưởng của người nghiên cứu từ nửa sau discusstion.txt → tách khỏi flattery AI → kiểm chứng từng ý bằng toán + literature → phản biện → hệ thống hóa thành methodology

**Nguyên tắc**: Tài liệu này *không* re-explain SpecRoute hay GainLoRA. Tài liệu này tập trung vào **các ý tưởng gốc của bạn** — phân tích cái đúng, cái sai, cái bị AI overstate, và xây dựng methodology từ phần solid.

---

## I. PROBLEM DEFINITION — Không Phải "Improve Routing", Mà Là "What Is The Right Problem?"

### 1.1 Setting chính thức

Cho:
- Pre-trained backbone $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ (frozen)
- Chuỗi $T$ tasks đến tuần tự: $\mathcal{T}_1, \mathcal{T}_2, ..., \mathcal{T}_T$
- Mỗi task $\mathcal{T}_t$ có dataset $\mathcal{D}_t = \{(x_i^{(t)}, y_i^{(t)})\}$ chỉ available trong giai đoạn train task $t$

Constraints:
- **Zero-replay**: Khi train task $t$, không có $\mathcal{D}_{t'}, t' < t$
- **No task-ID at inference**: Tại test time, không biết $x$ thuộc task nào
- **Expandable LoRA**: Mỗi task $t$ thêm LoRA branch $\Delta W_t = B_t A_t$ (rank $r$), freeze sau khi train xong

Sau $T$ tasks, forward pass cho input $h$:

$$\text{output}(h) = W_0 h + \sum_{t=1}^{T} w_t(h) \cdot B_t A_t h$$

trong đó $w_t(h) \in [0,1]$ là routing weight.

### 1.2 Ba sub-problems không thể tách rời

Bất kỳ phương pháp nào trong setting này đều phải giải **đồng thời** 3 bài toán:

| Sub-problem | Đầu vào | Đầu ra | Constraint |
|-------------|---------|--------|------------|
| **R: Routing** | Input $h$, expert set $\{\Delta W_t\}$ | Weights $w(h) \in \mathbb{R}^T$ | No task-ID, computable from $h$ alone |
| **P: Protection** | New task gradient $\nabla_{\theta} \mathcal{L}_T$ | Projected gradient $\tilde{\nabla}$ | Old experts' functionality preserved |
| **A: Allocation** | Available subspace $M^{\perp}$, new task demand | How much of $M^{\perp}$ to use | Fair capacity across tasks |

**Tại sao không thể tách rời?**

Routing quality phụ thuộc vào expert isolation (P), vì nếu new task can thiệp old expert → routing signal bị corrupt. Expert isolation phụ thuộc vào subspace budget (A), vì tight orthogonal constraint → good isolation nhưng limited capacity. Capacity limitation ảnh hưởng chất lượng expert → ảnh hưởng routing relevance.

Vòng tròn: **R ← P ← A ← R**.

### 1.3 Tại sao đây KHÔNG phải bài toán MoE

Mixture of Experts (trong LLM) và expandable LoRA CL trông giống nhau (nhiều expert, cần routing) nhưng khác biệt bản chất:

| Aspect | MoE (LLM) | Expandable LoRA CL |
|--------|-----------|---------------------|
| Expert creation | Đồng thời (jointly trained) | Tuần tự (each expert only sees its task) |
| Routing | Learned gating, optimized end-to-end | Cannot learn across tasks (forgetting risk) |
| Load balancing | Desirable (use all experts equally) | NOT desirable (want SELECTIVE activation) |
| Expert overlap | Expected, managed by auxiliary losses | Constrained by orthogonal projection |
| Data at routing time | All data available | Zero-replay → only current data |

Hệ quả: **Mọi technique của MoE routing (OT balancing, learned gating, regularization) đều không directly applicable.** Cần routing mechanism riêng cho CL setting.

---

## II. INFORMATION LANDSCAPE — Cái Gì Hợp Lệ, Cái Gì Vi Phạm?

### 2.1 Taxonomy of available information

Sau khi train xong task $t$, trước khi quên $\mathcal{D}_t$, ta có thể extract và lưu:

| Loại thông tin | Ví dụ | Hợp lệ? | Lý do |
|---------------|-------|---------|-------|
| **Model parameters** | Frozen $A_t, B_t$, GPM bases $U_t$ | ✅ | Là artifact của quá trình train, không phải data |
| **Derived quantities from parameters** | SVD of $\Delta W_t = U_t \Sigma_t V_t^T$ | ✅ | Computed from model params alone |
| **Data statistics** | Mean features $\mu_t$, covariance $\Sigma_t$ | ❌ | Summary of $\mathcal{D}_t$ → violates zero-replay |
| **Distribution parameters** | vMF $(\mu_t, \kappa_t)$ | ❌ | Fitted on $\mathcal{D}_t$ → violates zero-replay |
| **Auxiliary learned params** | Prompt keys, trans_input MLPs | ⚠️ Hợp lệ nhưng có forgetting risk | Phải train → gradient update có thể corrupt old |

### 2.2 Phân biệt tinh tế: GPM bases vs data statistics

GPM computation:
1. Forward pass data qua LoRA → collect input covariance matrix $C_t \in \mathbb{R}^{d \times d}$
2. SVD: $C_t = U_t S_t V_t^T$ → lấy principal directions $U_t[:, :k]$
3. Lưu $U_t[:, :k]$ (directions), BỎ $S_t$ (magnitudes)

Tại sao hợp lệ? Vì GPM bases encode **hướng (directions)** mà LoRA input hoạt động — đây là property của model + data combination mà cần forward pass để extract. Tuy nhiên, chỉ giữ lại **subspace** (span of directions), không giữ **distribution** (how data distributes within subspace).

**Lằn ranh đỏ**: Nếu một method lưu mean feature vector $\mu_t = \frac{1}{N}\sum_i f(x_i^{(t)})$ → đây là data statistic, vi phạm zero-replay. Feature Distributions paper (ICML 2025) làm chính xác điều này — cần position rõ ràng.

### 2.3 Hệ quả cho routing design

Từ Section 2.1, routing mechanism chỉ được sử dụng:
1. **Frozen model parameters**: $\{A_t, B_t\}_{t=1}^{T}$, frozen backbone $W_0$
2. **Quantities derived from frozen parameters**: SVD, norms, angles, etc.
3. **Current input** $h$ tại inference time

Routing **KHÔNG ĐƯỢC** sử dụng:
1. Learned parameters (prompt keys, gating networks) → forgetting risk
2. Data statistics từ old tasks (means, distributions) → zero-replay violation
3. Task labels → no task-ID

**Proposition 1**: *Trong zero-replay expandable LoRA CL, routing mechanism parameter-free (derived entirely from frozen expert weights + current input) là thỏa mãn tất cả constraints.*

*Lưu ý*: Đây không có nghĩa learned routing "sai" — GainLoRA dùng learned params + GPM protection cho routing params → hợp lệ nhưng cần thêm mechanism (GPM for trans_input, per-step projection). Parameter-free routing loại bỏ nhu cầu các mechanism phụ này.

---

## III. EXPERT CHARACTERIZATION — Từ Frozen Weights Đến Task Identity

### 3.1 Fundamental question: "Expert này LÀM GÌ?"

Mỗi frozen expert $\Delta W_t = B_t A_t \in \mathbb{R}^{d_{out} \times d_{in}}$ thực hiện:

$$h \mapsto \Delta W_t h = B_t (A_t h)$$

Từ SVD: $\Delta W_t = U_t \Sigma_t V_t^T$, decompose thành:
- $V_t^T h$: **Project input** lên principal input directions (WHAT the expert "looks at")
- $\Sigma_t$: **Scale** each projected component (HOW MUCH the expert cares)
- $U_t$: **Map to output** space (WHERE the expert "writes")

### 3.2 Spectral Signature: định nghĩa chính thức

**Definition**: *Spectral signature* của expert $t$ là cặp:
$$\mathcal{S}_t = \{(v_{t,i}, \sigma_{t,i})\}_{i=1}^{r}$$

trong đó $v_{t,i}$ là right singular vector thứ $i$ (input direction), $\sigma_{t,i}$ là singular value tương ứng.

**Tại sao dùng right singular vectors (V) chứ không phải left (U)?**

Routing quyết định từ **input** $h$ → cần so sánh $h$ với **input directions** mà expert listens to. Right singular vectors $V_t$ chính là "input space receptors" của expert. Left singular vectors $U_t$ encode output space — relevant cho aggregation, không phải routing.

### 3.3 Projection Fit: đo lường "expert $t$ relevant tới input $h$ bao nhiêu?"

**Definition**: *Weighted Projection Fit* của expert $t$ cho input $h$:

$$\text{fit}_t(h) = \frac{\sum_{i=1}^{r} \sigma_{t,i}^2 (v_{t,i}^T h)^2}{\sum_{i=1}^{r} \sigma_{t,i}^2 \cdot \|h\|^2}$$

**Giải thích từng thành phần**:
- $(v_{t,i}^T h)^2$: bao nhiêu "năng lượng" của $h$ nằm theo hướng $v_{t,i}$
- $\sigma_{t,i}^2$: expert coi trọng hướng $v_{t,i}$ bao nhiêu (singular values lớn = modification mạnh)
- $\|h\|^2$: chuẩn hóa
- Tử số: tổng weighted projection energy
- Mẫu số: maximum possible (khi $h$ nằm hoàn toàn trong span($V_t$))

**Tính chất toán học**:
- $\text{fit}_t(h) \in [0, 1]$
- $\text{fit}_t(h) = 1 \iff h \in \text{span}(V_t)$ và $h$ aligned với dominant singular vectors
- $\text{fit}_t(h) = 0 \iff h \perp \text{span}(V_t)$ (expert hoàn toàn không "thấy" input)

**Liên hệ với Rayleigh Quotient**: 

Nếu ta define $M_t = V_t \text{diag}(\sigma_t^2) V_t^T$ (PSD matrix), thì:

$$\text{fit}_t(h) = \frac{h^T M_t h}{\text{tr}(M_t) \cdot h^T h}$$

Đây chính là **normalized Rayleigh quotient** — công cụ chuẩn trong spectral theory, KHÔNG phải construction ad hoc.

### 3.4 Tại sao projection fit là "đúng" cho bài toán này? (Và tại sao nó có thể "sai")

**Tại sao đúng (theoretical argument)**:

1. **Respect expert structure**: fit_t(h) được derive trực tiếp từ SVD of expert weights — encode what the expert WAS TRAINED to do.

2. **Per-input**: Mỗi $h$ khác nhau cho projection fit khác nhau → supports mixed-task batches (crucial for real inference).

3. **Parameter-free**: Computed from frozen quantities + current input → no forgetting risk.

4. **Discriminative by construction**: Nếu experts operate trên orthogonal input subspaces (guaranteed approximately by GPM), thì:
   $$\text{span}(V_t) \approx \perp \text{span}(V_{t'}), \quad t \neq t'$$
   $$\Rightarrow \text{fit}_t(h) \text{ high} \implies \text{fit}_{t'}(h) \text{ low for } t' \neq t$$

**Tại sao có thể sai (honest caveats)**:

1. **Modification energy ≠ modification quality**: $\sigma_{t,i}^2 (v_{t,i}^T h)^2$ đo expert sẽ **modify mạnh** input $h$ theo hướng $v_{t,i}$. Nhưng modification mạnh KHÔNG có nghĩa là modification ĐÚNG. Expert có thể modify mạnh nhưng sai hướng output.

   *Counter-argument*: Expert được train trên task $t$ → modification patterns encode task-relevant transformations. Projection fit cao → input tương tự training distribution → modification likely correct. Nhưng đây là **assumption**, không phải guarantee.

2. **GPM orthogonality là approximate**: Thực tế, null-space projection không hoàn hảo. Subspace overlap nhỏ vẫn tồn tại → discriminative property bị weakened.

3. **Mean pooling loses structure**: Cả GainLoRA và SpecRoute dùng `avg_inputs_embeds = mean(token_embeddings)` cho routing. Hai sequences có content khác nhau nhưng similar average → misrouted.

---

## IV. ROUTING MECHANISM — Derive from Principles

### 4.1 Formulation: routing as maximum likelihood expert assignment

Cho input $h$, routing weights $w(h) = [w_1(h), ..., w_T(h)]$ sao cho weighted combination approximates oracle:

$$\sum_{t=1}^{T} w_t(h) \cdot \Delta W_t h \approx \Delta W_{\text{oracle}(h)} h$$

trong đó $\text{oracle}(h)$ là expert "đúng" (trained on task mà $h$ thuộc về).

### 4.2 Competitive routing (softmax) vs. Independent gating (sigmoid)

**Independent gating (GainLoRA)**:

$$w_t(h) = |2\sigma(4 \cdot \text{cos}(k_t, f_t(h))) - 1| \quad \in [0, 1] \text{ independently}$$

*Ưu điểm*: Cho phép multiple experts fire đồng thời (useful nếu task mới overlap concept cũ).
*Nhược điểm*: 
- $\sum_t w_t(h) \neq 1$ → modification magnitude thay đổi theo số experts → scale instability
- Tất cả experts có thể fire simultaneously → blurring
- Cho phép $\sum_t w_t = 0$ → no modification at all → information loss

**Competitive routing (softmax)**:

$$w_t(h) = \frac{\exp(\text{fit}_t(h) / \tau)}{\sum_{t'} \exp(\text{fit}_{t'}(h) / \tau)}$$

*Ưu điểm*:
- $\sum_t w_t(h) = 1$ → constant modification energy → stable training
- Forces **competition** → natural selection of most relevant expert(s)
- $\tau \to 0$: hard routing (winner-take-all); $\tau \to \infty$: uniform averaging

*Nhược điểm*:
- Phải assign TOÀN BỘ weight → nếu input không thuộc task nào rõ ràng, vẫn phải "chọn"
- Soft assignment → mỗi expert vẫn contribute dù ít → small interference

**Trong CL setting**: Competitive routing phù hợp hơn vì:
1. Tasks non-overlapping → mỗi input thuộc đúng 1 task → competition là đúng inductive bias
2. Scale stability quan trọng hơn flexibility (15 tasks × 48 layers × 2 projections = many routing decisions)
3. GPM already ensures expert isolation → independent gating phải học isolation from scratch (redundant)

### 4.3 Thuật toán routing hoàn chỉnh

```
INPUT: h ∈ R^{d_model}  (averaged input embedding)
       {S_t}_{t=1}^{T}  (spectral signatures: {V_t, σ_t} for each layer, each projection)
       τ > 0             (temperature)

FOR EACH ENCODER LAYER l, PROJECTION TYPE p ∈ {Q, V}:
  FOR EACH TASK t = 1, ..., T:
    V_t^{(l,p)}, σ_t^{(l,p)} = S_t[l, p]     # frozen spectral signature
    proj = V_t^{(l,p)} h                       # project input onto expert's input space
    fit_t^{(l,p)} = Σ_i σ²_{t,i} proj²_i / (Σ_i σ²_{t,i} · ||h||²)
  END FOR

  # Average fit across layers (global routing decision)
  fit_t = mean over (l, p) of fit_t^{(l,p)}

  # Competitive routing
  w(h) = softmax([fit_1, ..., fit_T] / τ)

RETURN w(h) ∈ R^T, Σ_t w_t = 1
```

**Lưu ý implementation**: Trong code hiện tại, fit scores được average chỉ over encoder layers (consistent — routing decision từ encoder, apply cho cả decoder). Decoder không tham gia routing computation.

### 4.4 Special case: current task (đang train)

Khi đang train task $T$, LoRA weights $(A_T, B_T)$ chưa frozen → chưa có spectral signature.

**Giải pháp hiện tại**: Dùng rows of $A_T$ trực tiếp (thay vì SVD) — vì khi $r$ nhỏ (=4), $\Delta W = BA$ có rank $r$, và $A$ (khi normalized) approximate right singular vectors.

**Giải thích**: Cho $\Delta W = BA$, nếu $B^T B = I$ (orthonormal), thì SVD of $\Delta W$ có $V = $ rows of $A$ (up to scaling). Trong thực tế $B^T B \neq I$, nên đây là approximation. Nhưng tại $r=4$, sai số nhỏ.

**Hệ quả**: Fit cho current task:
$$\text{fit}_T(h) = \frac{\|A_T h\|^2}{r \cdot \|h\|^2}$$

(unweighted, vì chưa có singular values — treat all directions equally)

---

## V. ANTI-FORGETTING — Gradient Projection as Structural Isolation

### 5.1 Bài toán: bảo vệ expert cũ khi train expert mới

Khi train task $T$, gradient $\nabla_{A_T} \mathcal{L}_T$ có thể vô tình interfere với experts cũ thông qua **shared representation space** (cùng backbone $W_0$, cùng input space $\mathbb{R}^{d_{in}}$).

Cách interference xảy ra:
1. Input $h$ cho old task $t$ đi qua new expert $T$ (routing error)
2. New expert $T$ train trên subspace overlap với old expert $t$ → modify shared directions

### 5.2 GPM (Gradient Projection Memory) — mechanism chính

**Idea**: Đảm bảo new LoRA operates trong **null-space** của old LoRA input subspaces.

**Formalization**: Gọi $\mathcal{M}_t = \text{span}(U_t^{GPM})$ là input subspace that expert $t$ uses. Accumulated protected subspace:

$$\mathcal{M}_{1:T-1} = \text{span}\left(\bigcup_{t=1}^{T-1} U_t^{GPM}\right)$$

*(incremental — có thể compute bằng progressive SVD update)*

New LoRA $A$ initialization:

$$A_T = A_T^{init} - \text{Proj}_{\mathcal{M}_{1:T-1}}(A_T^{init})$$

trong đó $\text{Proj}_{\mathcal{M}}(X) = U_{\mathcal{M}} U_{\mathcal{M}}^T X$ (project onto old subspace, then subtract).

**Guarantee**: $A_T h \perp \mathcal{M}_{1:T-1}$ for all $h$, i.e., new LoRA input activations are orthogonal to old LoRA input activations.

### 5.3 Per-step projection (cần thiết khi có learned routing params)

GainLoRA có `trans_input` (MLP) và `prompt_key` là learned parameters → mỗi optimizer step phải project gradient update:

```python
# After optimizer.step():
new_weight = current_weight - project_onto_old_subspace(current_weight - old_weight)
```

SpecRoute loại bỏ learned routing params → **KHÔNG CẦN** per-step projection cho routing. Chỉ cần GPM cho LoRA layers.

**Hệ quả thực tế**: SpecRoute training loop đơn giản hơn significatv (no custom `_inner_training_loop`, no per-step weight manipulation, use base class trainer).

### 5.4 Interaction giữa GPM và routing

**Key insight**: GPM + spectral routing tạo **dual protection**:

1. **GPM** (structural): New expert operates in orthogonal subspace → CAN'T interfere with old expert outputs
2. **Spectral routing** (functional): Old-task inputs routed to old experts → WON'T be processed by new expert

Individually, mỗi mechanism leaky:
- GPM alone: orthogonality approximate, small interference possible
- Routing alone: misrouting → wrong expert processes input

Together: even if routing makes small mistake, GPM ensures interference is orthogonal (small). Even if GPM leaks slightly, routing directs input to correct expert.

**Điều này nghĩa là**: Ta không cần perfect routing NOR perfect orthogonality — chỉ cần cả hai "tốt vừa đủ" để bù cho nhau.

---

## VI. SUBSPACE ALLOCATION — The Honest Hard Problem

### 6.1 Bài toán capacity

Input space $\mathbb{R}^{d_{in}}$ (d=1024 cho T5-Large). Mỗi task claim subspace of dimension ≤ $k_t$ cho GPM. Available null-space:

$$\dim(\mathcal{M}_{1:T}^{\perp}) = d - \dim(\mathcal{M}_{1:T}) \geq d - \sum_{t=1}^{T} k_t$$

Với $T = 15$ tasks, nếu mỗi task claim $k = 60$ dims: $1024 - 900 = 124$ dims remaining → **tight but feasible**.

### 6.2 Threshold controls capacity

GPM threshold $\epsilon$ controls $k_t$: higher threshold → more directions retained → larger $k_t$ → faster exhaustion.

| Strategy | Formula | Effect |
|----------|---------|--------|
| **GainLoRA original** | $\epsilon_t = (1-\epsilon_0) \cdot t/T + \epsilon_0$ | Tăng dần → early tasks protect nhiều, late tasks protect ít. **Unfair**: early tasks "chiếm" subspace disproportionately. |
| **Constant threshold** (SpecRoute) | $\epsilon_t = \epsilon_0, \forall t$ | Mỗi task protect cùng tỷ lệ. **Fair** nhưng vẫn linear depletion. |
| **Importance-weighted** (NOT YET IMPLEMENTED) | $k_t$ allocated based on task complexity | Potentially optimal nhưng cần metric cho "importance" |

### 6.3 Thẳng thắn: ESA (Elastic Subspace Allocation) hiện tại yếu

Cái gọi là "ESA" trong SpecRoute thực tế chỉ là **thay đổi threshold schedule từ tăng dần sang hằng số**. Đây là hyperparameter change, không phải algorithmic contribution.

**Nếu muốn ESA thực sự contributes**, cần ít nhất 1 trong:

1. **Importance-weighted protection**: Singular values lớn ($\sigma_i$ lớn) → direction quan trọng cho expert → protect mạnh hơn. Singular values nhỏ → direction ít quan trọng → có thể release cho future tasks.

   $$k_t = \min\{k : \sum_{i=1}^{k} \sigma_i^2 / \sum_j \sigma_j^2 \geq \epsilon\}$$

   Hiện tại SpecRoute KHÔNG dùng singular values trong GPM decision — chỉ dùng input covariance SVD (khác).

2. **Subspace recycling**: Detect directions trong $\mathcal{M}_{1:T-1}$ mà không expert nào dùng actively (routing weight luôn ~0) → release.

3. **Adaptive threshold based on remaining capacity**: $\epsilon_t = f(d - \dim(\mathcal{M}_{1:t-1}))$ — threshold giảm khi subspace cạn → force later tasks to be more selective.

**Status**: Cả 3 đều chưa implement. Bất kỳ cái nào nếu implement + ablation study → mới thực sự là contribution.

---

## VII. REPRESENTATION DRIFT — The Elephant in the Room

### 7.1 Vấn đề

Spectral signatures $\{V_t, \sigma_t\}$ frozen → KHÔNG drift. Nhưng input embedding $h$ **CÓ drift**.

**Cơ chế drift**: Trong encoder/decoder architecture, output of layer $l$:

$$h^{(l+1)} = f\left(W_0^{(l)} h^{(l)} + \sum_t w_t(h^{(0)}) \cdot B_t^{(l)} A_t^{(l)} h^{(l)}\right)$$

Khi thêm LoRA branch mới (task $T+1$), $w_t$ thay đổi (vì thêm competitor) → $h^{(l+1)}$ thay đổi → cascade qua layers.

**Hệ quả**: Projection fit $\text{fit}_t(h)$ tại task $T+1$ khác so với task $T$, dù $V_t, \sigma_t$ giữ nguyên — vì $h$ thay đổi.

### 7.2 So sánh: GainLoRA Handle drift bằng cách nào?

GainLoRA dùng **previous_trans_input** — frozen MLP snapshot per task. Mỗi old task $t$ có riêng:

$$f_t(x) = \text{SiLU}(W^{out}_t \cdot \text{SiLU}(W^{in}_t \cdot x))$$

Routing: compute $f_t(\bar{h})$ rồi cosine similarity với frozen prompt_key $k_t$.

**Ý tưởng**: Mỗi expert "nhìn" input qua "lăng kính" riêng (frozen MLP), expect cosine similarity patterns từ khi nó được train. Input có thể drift, nhưng prompt_key + trans_input snapshot là "matched pair" → somehow robust.

**Nhưng vẫn leaky**: $\bar{h}$ (average input embedding) vẫn drift → $f_t(\bar{h})$ output khác → cosine similarity thay đổi. Frozen MLP + frozen key KHÔNG fully compensate cho input drift, chỉ reduce sensitivity.

### 7.3 SpecRoute: explicitly acknowledge drift, don't pretend to solve it

SpecRoute claim "zero routing forgetting" — chính xác hơn nên nói:

> **"Zero parameter drift in routing mechanism"** — routing computation không có learned parameters nên không có parameter forgetting. Nhưng **representation drift** (thay đổi trong input embeddings do accumulated LoRA effects) vẫn tồn tại.

**Tại sao representation drift có thể manageable (hypothesis, chưa proven)**:

1. **LoRA rank nhỏ** ($r = 4$): Mỗi task chỉ modify rank-4 subspace. Total modification after 15 tasks: rank ≤ 60 (nếu orthogonal). Trong space 1024-dim, đây là ~6% dimensions → $h$ drift nhỏ.

2. **GPM ensures orthogonal modification**: New task modify directions mà old task KHÔNG dùng → old task's projection space ít bị ảnh hưởng.

3. **Backbone frozen**: $W_0$ không thay đổi → bulk of transformation stable. LoRA chỉ thêm residual.

**Cần kiểm chứng thực nghiệm**:
- Đo $\|\text{fit}_t(h) \text{ at task } T - \text{fit}_t(h) \text{ at task } t\|$ qua tasks
- If drift small → hypothesis confirmed
- If drift large → need explicit drift compensation mechanism

### 7.4 Potential mitigation (chưa implement, nhưng well-defined)

Nếu representation drift nghiêm trọng, options:

1. **Snapshot input normalization**: Store $\mu_t^{proj}, \sigma_t^{proj}$ (mean/std of projected features at training time) → normalize at inference: $\hat{h} = (h - \mu_t^{proj})/\sigma_t^{proj}$ trước khi compute fit.
   - **Vấn đề**: $\mu_t^{proj}$ là data statistic → có thể vi phạm zero-replay
   - **Counter**: chỉ cần mean/std of LoRA LAYER output (model output, not data) — ambiguous territory

2. **Relative fit**: Thay vì absolute fit $\text{fit}_t(h)$, dùng relative ranking. Distribution shift affects all fits similarly → ranking preserved.
   - Softmax inherently does this partially (chỉ care ordering, not absolute values)

3. **Self-calibration**: Periodically (every $k$ tasks), recompute spectral signatures on new LoRA weights.
   - Nhưng old LoRA weights frozen → signatures không thay đổi → chỉ current task affected → not helpful

---

## VIII. THE COMPLETE ALGORITHM — End to End

### 8.1 Training phase (cho task $T$)

```
INPUTS: Pre-trained backbone W₀
        Frozen experts {(A_t, B_t)}_{t=1}^{T-1}
        Spectral signatures {S_t}_{t=1}^{T-1}
        GPM bases {M_{1:T-1}}
        Training data D_T

STEP 1 — Initialize new LoRA branch:
  A_T^{init} ← random (Kaiming)
  A_T ← A_T^{init} - Proj_{M_{1:T-1}}(A_T^{init})   # null-space projection
  B_T ← 0 OR random (scaled small)

STEP 2 — Train with routing:
  for each batch (x, y) in D_T:
    h̄ ← mean_pool(encoder_embed(x))                   # average input embedding
    w(h̄) ← spectral_routing(h̄, {S_t}_{t<T}, A_T)     # Section IV.3
    
    for each layer l:
      LoRA_output_l ← Σ_t w_t(h̄) · B_t^(l) A_t^(l) h^(l)  # weighted aggregation
      h^(l+1) ← layer_l(h^(l)) + LoRA_output_l
    
    loss ← task_loss(output, y)
    loss.backward()
    
    # Only A_T and B_T have gradients (others frozen)
    optimizer.step()                                     # No per-step projection needed

STEP 3 — End of task:
  Freeze A_T, B_T
  
  # Compute spectral signature
  ΔW_T = B_T @ A_T
  U, Σ, V^T = SVD(ΔW_T)
  S_T = {V[:r], Σ[:r]}                                 # store for future routing
  
  # Update GPM
  Compute input covariance from forward passes
  SVD → extract top-k directions
  M_{1:T} = M_{1:T-1} ∪ new_directions
  
  Save: {A_T, B_T, S_T, M_{1:T}}
  Discard: D_T (zero-replay)
```

### 8.2 Inference phase

```
INPUT: Test sample x (no task-ID)

STEP 1 — Encode + route:
  h̄ ← mean_pool(encoder_embed(x))
  w(h̄) ← softmax([fit_1(h̄), ..., fit_T(h̄)] / τ)

STEP 2 — Forward with routing:
  for each layer l:
    LoRA_output_l ← Σ_t w_t(h̄) · B_t^(l) A_t^(l) h^(l)
    h^(l+1) ← layer_l(h^(l)) + LoRA_output_l

STEP 3 — Decode output
```

### 8.3 Complexity analysis

| Operation | GainLoRA | SpecRoute | Comment |
|-----------|----------|-----------|---------|
| Routing computation | $O(T \cdot d \cdot h_{mlp} + T \cdot d)$ | $O(T \cdot r \cdot d \cdot L)$ | SpecRoute: matrix-vector per layer per task |
| Trainable routing params | $O(2 \cdot d \cdot h_{mlp} + d)$ per task | $0$ | SpecRoute: no routing params |
| GPM targets | LoRA + trans_input + prompt_key | LoRA only | SpecRoute: simpler GPM |
| Per-step overhead | Null-space projection for routing params | None | SpecRoute: standard training loop |
| End-of-task | GPM + freeze + save snapshots | GPM + freeze + SVD | SVD is $O(d_{out} \cdot d_{in} \cdot r)$ — cheap for small $r$ |
| Memory per task | $A_t, B_t$ + prompt_key + trans_input weights | $A_t, B_t$ + spectral sig $(V_t, \sigma_t)$ | Similar; spectral sig slightly smaller than trans_input |

---

## IX. POSITIONING IN THE LANDSCAPE

### 9.1 So sánh phương pháp-agnostic

| Criterion | GainLoRA | InfLoRA | MINGLE | Feature Dist. | TreeLoRA | SpecRoute |
|-----------|----------|---------|--------|---------------|----------|-----------|
| Routing type | Learned (MLP+key) | None (equal weight) | Learned (MoE gate) | Feature similarity | Gradient similarity | Spectral projection |
| Routing forgetting risk | ⚠️ Managed by GPM | N/A | ⚠️ Managed by EMA | ❌ Stores data stats | ⚠️ Needs old gradients | ✅ Parameter-free |
| Zero-replay | ✅ | ✅ | ✅ | ⚠️ Stores mean features | ⚠️ Needs gradient similarity | ✅ |
| Anti-forgetting | GPM on LoRA + routing | Null-space init | OGP (orthogonal) | None explicit | None explicit | GPM on LoRA only |
| Subspace allocation | Increasing threshold | Fixed threshold | EMA relaxation | N/A | N/A | Constant threshold |
| Aggregation | Weighted sum (sigmoid) | Equal sum | Top-k MoE | Weighted sum | Tree selection | Weighted sum (softmax) |

### 9.2 Novelty assessment (honest)

**Clearly novel**:
- Using SVD of frozen LoRA weights (not data features, not learned keys) as routing signal — no prior work does exactly this.
- Elimination of ALL learned routing parameters in expandable LoRA CL — GainLoRA, MINGLE both require learned routing.

**Partially novel**:
- Weighted Rayleigh quotient for routing — Rayleigh quotient is textbook, but application to LoRA-CL routing is new.
- Demonstrating that parameter-free routing + GPM = sufficient (if it works empirically) — conceptual contribution.

**NOT novel**:
- GPM/null-space projection — from InfLoRA, GainLoRA
- Expandable LoRA architecture — from O-LoRA, InfLoRA, GainLoRA
- Softmax routing in MoE-like structures — foundational MoE work
- SVD as analysis tool for LoRA — SD-LoRA analyzes magnitude/direction

**Closest competitor**: Feature Distributions (ICML 2025) — stores characterization per expert, uses similarity for routing. Key difference: they store data-level features (mean activation vectors), we store weight-level signatures (SVD of frozen params). They arguably violate or stretch zero-replay; we don't.

---

## X. WHAT NEEDS TO BE TRUE — Assumptions Checklist

Mỗi assumption dưới đây CẦN PHẢI TRUE để methodology work. Mỗi cái cần empirical validation.

### 10.1 Core assumptions

| # | Assumption | Status | How to test |
|---|-----------|--------|-------------|
| A1 | Projection fit correlates with "correct expert" assignment | ❓ UNTESTED | Compute fit accuracy on task-boundary evaluation sets |
| A2 | GPM+routing dual protection sufficient to prevent forgetting | ❓ UNTESTED | Compare forgetting metric with vs without routing |
| A3 | Representation drift is small enough to not corrupt routing | ❓ UNTESTED | Track fit_t(h) variance across tasks for fixed test inputs |
| A4 | mean_pool captures enough task-relevant signal for routing | ❓ UNTESTED | Compare with max_pool, CLS token, attention-weighted pool |
| A5 | Softmax temperature τ is not overly sensitive | ❓ UNTESTED | τ ablation study |
| A6 | rank r=4 is sufficient for spectral signatures to be discriminative | ❓ UNTESTED | r ablation |

### 10.2 Implied assumptions (from GainLoRA that we inherit)

| # | Assumption | Status |
|---|-----------|--------|
| A7 | T5-Large backbone generalizable to other architectures (LLaMA) | Partially tested (GainLoRA has LLaMA configs) |
| A8 | 15 tasks is within GPM capacity for d=1024 | Expected (d=1024 >> 15*r*2) |
| A9 | Q and V projections sufficient (not K) | From GainLoRA design, standard in LoRA literature |

---

## XI. EXPERIMENTAL VALIDATION PLAN

### 11.1 What the experiments MUST show (not "nice to have")

1. **SpecRoute vs. GainLoRA on identical setting**: Same data, same preprocessing, same evaluation protocol. Show routing improves OR at least matches.

2. **Routing accuracy analysis**: On held-out validation sets of old tasks, what fraction of inputs are correctly routed (highest weight to correct expert)?

3. **Forgetting curve**: Plot per-task performance after each subsequent task. Compare degradation.

4. **Representation drift measurement**: For fixed test inputs from task $t$, track $\text{fit}_t(h)$ value as tasks $t+1, ..., T$ are added. If fit_t(h) drops significantly → drift is a problem.

### 11.2 Ablation studies (ranked by importance)

1. **Routing mechanism**: Spectral projection vs. prompt key (use SpecRoute architecture but GainLoRA routing) vs. random routing vs. uniform routing
2. **Aggregation**: Softmax vs. sigmoid vs. top-1 hard routing  
3. **Temperature τ**: Sweep from 0.01 to 10.0
4. **Threshold ε**: 0.99, 0.995, 0.999, increasing schedule, constant
5. **Mean pool vs. alternatives**: CLS token, max pool, attention-weighted

### 11.3 Analysis experiments (for paper)

1. **Visualization**: t-SNE of spectral signatures across tasks — do they cluster meaningfully?
2. **Routing weight heatmaps**: Per-task routing weight distribution over time
3. **Subspace dimension tracking**: Plot $\dim(\mathcal{M}_{1:t})$ vs $t$ — how fast does subspace fill?
4. **Singular value spectra**: Plot $\sigma_1, ..., \sigma_r$ for each task — do they vary meaningfully?

---

## XII. HONEST ASSESSMENT — Strengths and Weaknesses of This Methodology

### 12.1 Strengths

1. **Principled derivation**: Method follows from constraints (zero-replay, no task-ID) → information landscape → natural choice. Not "proposed then justified".

2. **Simplicity**: Removes learned routing entirely. Training loop simplifies. Fewer hyperparameters. Fewer mechanisms to maintain.

3. **Architectural alignment**: Routing signal comes FROM the experts themselves — not from separate parameters that might disagree with expert function.

4. **Dual protection theory**: GPM + routing => redundant safety mechanisms that compensate for each other's imperfections.

### 12.2 Weaknesses

1. **No empirical validation yet**: The entire framework is theoretical. Until experiments confirm, every section above is hypothesis.

2. **Representation drift is real, unaddressed**: We acknowledge it, hypothesize it's small, but don't solve it. If drift is large, the methodology needs significant revision.

3. **ESA is weak**: Subspace allocation is essentially a hyperparameter. This is the weakest part of the framework.

4. **Mean pooling is a bottleneck**: Entire routing decision based on 1 vector (average embedding). Rich sequence information lost.

5. **Modification energy ≠ quality**: Fundamental gap between "expert will modify input strongly" and "expert will modify input correctly". This is assumption, not theorem.

6. **Only tested on NLP**: Setting is specific (T5, NLP tasks). Generalization to vision/multimodal unknown.

### 12.3 What would KILL this approach

Red flags that would indicate fundamental issues:
- If routing accuracy is not significantly better than random → spectral signatures are not discriminative  
- If performance degrades significantly on later tasks (>2% compared to task-specific training) → GPM + routing dual protection insufficient
- If representation drift causes >10% routing accuracy drop between task $t$ and task $T$ → need drift compensation
- If τ has narrow "sweet spot" and small deviations cause large performance changes → method not robust

---

## XIII. RELATIONSHIP TO method.md (RTA Framework)

`method.md` describes RTA (Riemannian Topological Alignment) — a DIFFERENT direction involving:
- Bingham distributions (anisotropic) on hypersphere
- Riemannian KL divergence for topology preservation  
- Parallel transport for drift correction

**Comparison**:

| Aspect | SpecRoute (this doc) | RTA (method.md) |
|--------|---------------------|-----------------|
| Paradigm | Expandable LoRA + routing | Feature distribution preservation |
| Anti-forgetting | GPM (subspace isolation) | Riemannian distillation + topology lock |
| Drift handling | Acknowledge but don't solve | Parallel transport correction |
| Data requirement | Zero-replay compliant | Requires distribution parameters (violates?) |
| Maturity | Code exists, needs experiments | Purely theoretical |
| Complexity | Low (SVD + softmax) | High (manifold computation, Bingham fitting) |

**Key question**: RTA addresses representation drift explicitly (via parallel transport). Could elements of RTA complement SpecRoute's weakness? Possibly — but would need to verify that Bingham fitting doesn't violate zero-replay, and that parallel transport is tractable for 1024-dim space.

---

## XIV. CONCLUSION — WHAT THIS METHODOLOGY IS AND ISN'T

### What it IS:
- A principled framework that starts from problem constraints and derives method choices
- An architecture-agnostic approach to routing in expandable LoRA CL
- A clear specification of what information is legitimate under zero-replay
- An honest assessment of assumptions, limitations, and open problems

### What it ISN'T:
- A proven method (no experiments)
- A complete solution to all CL problems (subspace allocation, representation drift still open)
- A guaranteed improvement over GainLoRA (empirical question)
- A paper-ready manuscript (needs experiments, related work section, polished writing)

### Priority actions (ordered):
1. **Run SpecRoute vs. GainLoRA on SuperNI Order 1** — if doesn't match or beat GainLoRA, revisit fundamentals
2. **Measure routing accuracy** — confirm spectral signatures are actually discriminative  
3. **Measure representation drift** — confirm it's manageable
4. **Develop ESA properly** — importance-weighted protection
5. **Write paper** — only after 1-4 confirm methodology
