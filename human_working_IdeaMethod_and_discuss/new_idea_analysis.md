# Phân Tích Ý Tưởng Mới: Statistical Knowledge Signatures + OT Routing + Backbone Anti-Drift
## Comprehensive Analysis Report

---

# PHẦN 1: TỔNG QUAN Ý TƯỞNG MỚI

## 1.1 Bối cảnh & Động lực
Quan sát: Các paper top conference 2025 (NeurIPS, ICML, ICLR, ACL...) quan tâm rất nhiều tới **knowledge isolation via submodule + routing**:
- GainLoRA (NeurIPS'25): LoRA branches + gating
- MINGLE (NeurIPS'25): MoE + Null-Space Gating
- SMoLoRA (ICCV'25): Separable Mixture of LoRA
- TreeLoRA (ICML'25): Hierarchical gradient-similarity tree
- HiDe-LLaVA (ACL'25): Task-specific expansion + CKA fusion
- MoE-Adapters (CVPR'24): Standard MoE routing
- ... và nhiều paper khác

→ Xu hướng rõ ràng: **Submodule architecture + routing mechanism** là paradigm chủ đạo 2025.

## 1.2 Ba Thành Phần Của Ý Tưởng Mới

### Component 1: Statistical Knowledge Signatures
- Sử dụng công cụ thống kê mạnh (vMF, Bingham, GMM...) để **khái quát hóa không gian tri thức** của mỗi module
- Mỗi module/expert có một "chữ ký thống kê" (signature/fingerprint) mô tả phân phối dữ liệu mà nó đã học
- Khác biệt với gating networks: signature mang ý nghĩa thống kê rõ ràng, không phải learned weights

### Component 2: Optimal Transport Routing  
- Sử dụng OT làm **cơ chế routing có nguyên tắc** (principled routing)
- Cost matrix dựa trên **khoảng cách phân phối** giữa input và signatures của các modules
- Thay thế softmax gating/top-k selection bằng OT matching

### Component 3: Backbone Anti-Drift & Anti-Invasion
- Phần backbone chung (shared) được bảo vệ bởi:
  - Loss phạt drift representation (tâm cụm cũ không được trôi quá xa)
  - Loss phạt xâm lấn (class mới không được xâm phạm vùng class cũ)
- Kế thừa từ simple_idea cũ, áp dụng vào modular architecture

---

# PHẦN 2: ĐÁNH GIÁ TÍNH MỚI (NOVELTY ASSESSMENT)

## 2.1 Kết luận tổng quát: **NOVELTY CAO**

Không có paper nào (trong 109 papers khảo sát + ~30 papers bổ sung) kết hợp cả 3 thành phần. Từng thành phần riêng lẻ có prior work nhưng ở **mục đích và cách dùng khác**.

## 2.2 Cross-check với 109 Papers Khảo Sát

### Component 1 — Statistical Signatures cho Modules

| Paper | Gì đã làm | Khác biệt với ý tưởng mới |
|-------|-----------|---------------------------|
| **35. Feature Distributions** (ICML'25) | "Presentative feature distribution" để chọn PEFT block | Distribution = **mean vector only**, không phải rich statistical model (vMF/Bingham). Dùng cho block selection, không phải knowledge fingerprint |
| **73. PromptCCD** (ECCV'24) | GMM cho prompt pool routing | GMM = Gaussian, không geometric. Dùng cho category discovery, không phải CL routing |
| **96. FeCAM** (NeurIPS'23) | Class-specific covariance + Mahalanobis | Statistical modeling nhưng cho **classification** (single model), không phải module signature |
| **65. CLAP4CLIP** (NeurIPS'24) | Probabilistic feature modeling | Gaussian distribution, CLIP-based, không phải module fingerprint |

**Kết luận Component 1:** Paper 35 gần nhất nhưng chỉ dùng mean-vector representation, không phải rich statistical model. **Không có paper nào dùng vMF/Bingham/Directional distributions làm "chữ ký tri thức" cho module.**

### Component 2 — OT-based Routing

| Paper | Routing mechanism | Khác biệt |
|-------|------------------|-----------|
| **01. GainLoRA** (NeurIPS'25) | Gating modules | Learned gating, không distributional |
| **02. MINGLE** (NeurIPS'25) | Null-Space Constrained Gating | Algebraic constraint, không OT |
| **09. MoDE** (NeurIPS'25) | Modality-based separation | By modality, không distributional |
| **14. SMoLoRA** (ICCV'25) | Dual routing (visual + instruction) | Separable by function, không OT |
| **21. PLAN** (ICCV'25) | Orthogonal basis allocation | Algebraic, không OT |
| **23. ARM** (ACL'25) | Activation-guided routing | Activation-based, không distributional |
| **27. HiDe-LLaVA** (ACL'25) | CKA similarity fusion | Similarity metric, không OT |
| **41. TreeLoRA** (ICML'25) | Gradient-similarity tree | Gradient-based, không distributional |
| **82. MoE-Adapters** (CVPR'24) | Standard MoE gating | Softmax gating, không OT |
| **102. MRN** (ICCV'23) | Multiplexed routing | Language-specific paths, không OT |

**Kết luận Component 2: Trong 109 papers, KHÔNG có paper nào dùng OT cho routing trong CL.** Tất cả dùng gating networks, activation-based, gradient-similarity, hoặc algebraic constraints.

### Component 3 — Backbone Anti-Drift trong Modular Architecture

| Paper | Drift handling | Khác biệt |
|-------|---------------|-----------|
| **77. LDC** (ECCV'24) | Learnable drift compensation | **Single model**, không phải modular backbone |
| **20. Dual Drift** (ICCV'25) | Prototype drift analysis | Single model, prototype-level |
| **61. LoRA-** (CVPR'25) | Drift-Resistant Space | LoRA subtraction, không phải anti-drift loss |
| **47. Proxy-FDA** (ICML'25) | Feature distribution alignment | Single model + proxies |
| **13. MG-CLIP** (ICCV'25) | Modality gap preservation | CLIP-specific, không phải backbone share |

**Kết luận Component 3: Drift compensation đã được nghiên cứu, nhưng TRONG CONTEXT SINGLE-MODEL.** Không có paper nào áp dụng anti-drift + anti-invasion loss cho **backbone của modular architecture.**

## 2.3 Cross-check với Papers Bổ Sung (Ngoài 109)

### OT trong MoE/Routing (không phải CL)

| Paper | Chi tiết | Mối quan hệ |
|-------|---------|-------------|
| **BASE Layers** (ICML'21) | OT (linear assignment) cho balanced expert allocation | OT dùng cho **load-balancing**, KHÔNG phải distribution-matching routing. Cost matrix = learned scores, không phải distributional distances |
| **Grassmannian MoE** (arXiv Feb'26) | Matrix Bingham distributions trên Grassmannian manifold cho routing | **RỦI RO CAO NHẤT** — dùng Bingham cho routing. NHƯNG: (a) KHÔNG phải CL, (b) Bingham controls routing entropy (sparsity), KHÔNG characterize knowledge |
| **Selective Sinkhorn Routing** (Nov'25) | Sinkhorn-based routing cho MoE | OT cho load-balancing, không phải knowledge matching |

### Statistical Distributions trong CL (không phải module signatures)

| Paper | Chi tiết | Mối quan hệ |
|-------|---------|-------------|
| **vMF for Online CL** (AAAI'24) | vMF distribution cho online CL | vMF dùng như **training loss** (concentration penalty), KHÔNG dùng làm module fingerprint |
| **SCDEM** (Apr'25) | OT trong CL context | OT cho **feature alignment**, không phải routing |

### MoE + CL (không phải OT routing)

| Paper | Chi tiết | Routing mechanism |
|-------|---------|------------------|
| **CaRE** (arXiv Feb'26) | Continual Learning with Routing among Experts | Learned routing, không OT |
| **PASs-MoE** (arXiv Jan'26) | Parameter-Adaptive Sparse MoE | Adaptive sparsity, không OT |
| **TRGE** (arXiv Aug'25) | Task-Regularized Gradient Experts | Gradient-based expert selection |

## 2.4 Phân Tích Rủi Ro Novelty

### Rủi ro CAO — Grassmannian MoE (arXiv:2602.17798)
- **Overlap:** Dùng Bingham distribution + manifold geometry cho routing
- **Khác biệt quan trọng:**
  1. KHÔNG phải CL — chỉ là MoE cho language modeling
  2. Bingham controls **routing entropy** (sparsity vs utilization tradeoff)
  3. KHÔNG characterize "knowledge" của expert — chỉ control gating weight distribution
  4. KHÔNG có anti-drift/anti-invasion component
- **Kết luận:** Có thể cite as related work nhưng mục đích hoàn toàn khác

### Rủi ro TRUNG BÌNH — Paper 35 (Feature Distributions, ICML'25)
- **Overlap:** Dùng "feature distribution" để chọn module
- **Khác biệt:** Distribution = mean-vector, không rich statistical model. Dùng cho PEFT block selection, không phải principled routing
- **Kết luận:** Có thể position ý tưởng mới như generalization/upgrade

### Rủi ro THẤP — Các paper còn lại
- BASE Layers, SERS, FeCAM: Mỗi paper chỉ chạm 1 component ở mức surface-level

## 2.5 Bốn Khoảng Trống Novelty Được Xác Nhận

| # | Novelty Gap | Chưa có paper nào làm |
|---|-------------|----------------------|
| 1 | **Rich statistical signatures** | Dùng vMF/Bingham/directional distributions làm fingerprint cho expert knowledge space |
| 2 | **OT with distributional-distance cost** | OT routing dựa trên khoảng cách phân phối (KL, Wasserstein) giữa input và module signatures |
| 3 | **Three-component integration** | Kết hợp statistical signatures + OT routing + backbone protection trong 1 framework |
| 4 | **Anti-drift/invasion trong modular backbone** | Áp dụng center drift penalty + invasion loss cho shared backbone của modular architecture |

---

# PHẦN 3: PHÂN TÍCH TÍNH HỢP LÝ (SOUNDNESS ANALYSIS)

## 3.1 Component 1 — Statistical Knowledge Signatures

### Hợp lý ✅
- **Cơ sở lý thuyết:** Feature space của các encoder hiện đại (BERT, ViT) thường nằm trên manifold có cấu trúc (hypersphere cho normalized features, cone cho ReLU features). Dùng distribution phù hợp geometry (vMF cho hypersphere, Bingham cho elliptical) capture nhiều thông tin hơn mean vector.
- **Ưu điểm so với gating network:** Signature có interpretability (có thể đo concentration, direction, spread), trong khi gating weights là black-box.
- **Evidence từ literature:**
  - FeCAM (96): Chứng minh class-specific covariance (statistical tool) tốt hơn mean-only prototype
  - CLAP4CLIP (65): Probabilistic modeling > deterministic features
  - Angle Matters (48): Angle/direction trong feature space quyết định forgetting → distribution captures direction information

### Điểm cần lưu ý ⚠️
- **Cách cập nhật incremental:** Khi task mới đến, signature cần update. vMF có sufficient statistics (mean direction + concentration) → có thể online update. GMM phức tạp hơn.
- **Chi phí lưu trữ:** Mỗi module cần lưu signature parameters. vMF: O(d+1) mỗi module (mean direction vector + κ). Bingham: O(d²) mỗi module. Với d nhỏ (projection) → chấp nhận được.
- **Khuyến nghị:** Bắt đầu với vMF (đơn giản nhất, phù hợp hypersphere features) → mở rộng Bingham/GMM nếu cần.

## 3.2 Component 2 — OT-based Routing

### Hợp lý ✅
- **Cơ sở lý thuyết:** OT cung cấp optimal matching giữa 2 distributions, là framework tự nhiên cho "matching input to expert". Sinkhorn algorithm cho phép differentiable approximation.
- **Ưu điểm so với softmax gating:**
  - **Principled:** Tối ưu hóa global assignment thay vì local gating scores
  - **Load-balanced by design:** OT constraints tự nhiên balance load (đã chứng minh trong BASE Layers)
  - **Distribution-aware:** Cost matrix encode khoảng cách phân phối, không phải raw scores
- **Feasibility:** Sinkhorn iterations: O(n²·k) với n tokens, k experts. Với k nhỏ (CL thường 5-20 experts) → tractable.

### Điểm cần lưu ý ⚠️
- **Inference latency:** Sinkhorn cần iterative → chậm hơn softmax gating đơn giản. Mitigation: ít iterations (5-10), hoặc amortized inference.
- **Cost matrix construction:** Cần define cách tính khoảng cách giữa input sample/batch và module signature. Options: vMF log-likelihood, Wasserstein distance, KL divergence.
- **Khuyến nghị:** Dùng Sinkhorn với regularization ε lớn (fast convergence) + vMF log-likelihood as cost.

## 3.3 Component 3 — Backbone Anti-Drift

### Hợp lý ✅
- **Cơ sở lý thuyết:** Shared backbone trong modular architecture vẫn bị update → representation drift. No paper hiện tại address this explicitly.
- **Evidence:**
  - LDC (77): Chứng minh drift compensation cải thiện performance
  - Dual Drift (20): Inner-task + inter-task prototype drift đều gây forgetting
  - LoRA- (61): Drift-resistant space concept validates the need
- **Tự nhiên với modular architecture:** Backbone là phần chia sẻ giữa tất cả modules → drift ảnh hưởng TẤT CẢ old tasks đồng thời. Anti-drift loss ở backbone level → bảo vệ toàn bộ.

### Điểm cần lưu ý ⚠️
- **Balance plasticity-stability:** Anti-drift loss quá mạnh → backbone không học được features mới. Cần adaptive weighting.
- **Anti-invasion definition:** Trong modular architecture, "vùng class cũ" được define qua module signatures → tự nhiên link với Component 1.
- **Khuyến nghị:** Dùng EMA-based center tracking + dynamic λ scheduling (từ method.md RTA framework).

## 3.4 Tính Nhất Quán Nội Bộ (Internal Consistency)

| Aspect | Assessment | Giải thích |
|--------|-----------|------------|
| Component 1 ↔ 2 | ✅ Consistent | Signatures (C1) cung cấp distribution cho OT cost matrix (C2). Chúng designed to work together. |
| Component 2 ↔ 3 | ✅ Consistent | OT routing (C2) phân bổ input → modules. Anti-drift (C3) bảo vệ shared backbone. Hai cơ chế orthogonal, không conflict. |
| Component 1 ↔ 3 | ✅ Synergistic | Signatures (C1) cũng detect drift: nếu backbone drift → feature distribution thay đổi → signatures outdated → signal để trigger anti-drift. |

## 3.5 Đánh Giá Tổng Thể Tính Hợp Lý

**Ý tưởng hợp lý ở mức idea-level.** Ba thành phần có cơ sở lý thuyết vững, tương thích nội bộ, và address gap thực sự trong literature. Tiềm năng contribution mạnh nếu implementation đúng.

**Rủi ro lớn nhất:** Computational overhead (OT + distribution estimation + anti-drift) có thể significant. Cần careful engineering.

---

# PHẦN 4: KHẢO SÁT PAPERS 2025 — MOTIVATION ĐỂ APPLY Ý TƯỞNG MỚI

## 4.1 Tiêu Chí Đánh Giá Mới (cho New Idea)

| Tiêu chí | Mô tả | Trọng số |
|----------|--------|----------|
| **M1. Submodule architecture** | Paper dùng multi-module/expert/LoRA → new idea phù hợp | ★★★ |
| **M2. Routing có thể nâng cấp** | Routing hiện tại đơn giản (gating, top-k) → OT routing có thể improve | ★★★ |
| **M3. Backbone drift problem** | Paper có shared backbone bị drift → anti-drift loss applicable | ★★ |
| **M4. Domain phù hợp** | ML/NLP ưu tiên, CV thấp hơn | ★★ |
| **M5. Reproducibility** | Có code, benchmark rõ ràng | ★ |

Lưu ý: Đánh giá ở mức **phác thảo** — xem paper có motivation/feasibility để apply, KHÔNG xem chi tiết công cụ cụ thể (vMF có hợp hay không).

## 4.2 Papers 2025 Có Motivation Cao (Score ≥ 7/10)

### 🥇 Paper 01 | GainLoRA | NeurIPS'25 | NLP
**Motivation Score: 9/10**
- ✅ M1: LoRA branches per task + gating modules — multi-module architecture
- ✅ M2: Gating = simple learned module → OT routing có thể thay thế, phân bổ principled hơn
- ✅ M3: Shared base model bị update → backbone drift likely
- ✅ M4: NLP (LLM continual learning)
- **Lý do apply:** GainLoRA dùng gating đơn giản để integrate LoRA branches. Thay gating bằng (1) statistical signature cho mỗi LoRA branch + (2) OT routing matching input distribution → principled expert selection. Anti-drift loss bảo vệ base LLM.

### 🥇 Paper 02 | MINGLE | NeurIPS'25 | ML
**Motivation Score: 9/10**
- ✅ M1: MoE + low-rank experts + gating
- ✅ M2: Null-Space Constrained Gating — algebraic, không capture knowledge distribution
- ✅ M3: Test-time merging implies shared components
- ✅ M4: ML/Multi
- **Lý do apply:** MINGLE dùng null-space projection cho gating. Statistical signatures sẽ capture knowledge space richer hơn null-space constraint. OT routing provides global optimal assignment thay vì local gating.

### 🥇 Paper 41 | TreeLoRA | ICML'25 | ML
**Motivation Score: 9/10**
- ✅ M1: Layer-wise LoRA allocation via hierarchical tree
- ✅ M2: Gradient-similarity → heuristic, không capture full knowledge distribution
- ✅ M3: Shared pretrained model as backbone
- ✅ M4: ML (cả ViTs + LLMs)
- **Lý do apply:** TreeLoRA dùng gradient similarity để allocate LoRA. Gradient similarity = proxy cho task similarity nhưng không capture full distribution. Statistical signatures cho mỗi LoRA node trong tree → richer characterization. OT routing thay multi-armed bandit.

### 🥈 Paper 14 | SMoLoRA | ICCV'25 | ML/Multi
**Motivation Score: 8/10**
- ✅ M1: Separable Mixture of LoRA + dual routing
- ✅ M2: Dual routing (visual + instruction) → có thể upgrade sang OT matching
- ⚠️ M3: Shared backbone (VL model)
- ✅ M4: VL (multimodal, nhưng IT setting phổ dụng)
- **Lý do apply:** SMoLoRA dùng separable routing cho 2 modalities. OT routing có thể unify dual routing thành 1 cost matrix, với signatures capture both visual + instruction knowledge.

### 🥈 Paper 35 | Feature Distributions | ICML'25 | NLP
**Motivation Score: 8/10**
- ✅ M1: Multi-PEFT-block (expanding/reusing)
- ✅ M2: "Presentative feature distribution" for block selection — TRỰC TIẾP liên quan nhưng dùng mean-vector, not rich statistics
- ⚠️ M3: Pre-trained LLM backbone
- ✅ M4: NLP (LLM continual learning)
- **Lý do apply:** Paper ĐÃ dùng idea "feature distribution" để chọn block → **đây chính là starting point tốt nhất** cho new idea. Upgrade: thay mean-vector bằng vMF signature + thay selection bằng OT routing. Paper đã validate rằng distribution-based selection works.

### 🥈 Paper 82 | MoE-Adapters | CVPR'24 | ML/Multi
**Motivation Score: 8/10**
- ✅ M1: MoE adapter architecture
- ✅ M2: Standard MoE gating → classic candidate cho OT routing upgrade
- ⚠️ M3: VLM backbone
- ⚠️ M4: VL (CV-leaning)
- **Lý do apply:** Standard MoE gating là simplest routing, easiest to upgrade to OT. Có code (github.com/JiazuoYu/MoE-Adapters4CL).

### 🥈 Paper 27 | HiDe-LLaVA | ACL'25 | NLP
**Motivation Score: 8/10**
- ✅ M1: Task-specific expansion + task-general fusion
- ✅ M2: CKA similarity guides layer-wise handling → distribution signatures provide richer similarity
- ✅ M3: Shared LLaVA backbone
- ✅ M4: NLP (instruction tuning)
- **Lý do apply:** HiDe-LLaVA dùng CKA similarity → scalar measure. Distribution signature captures richer information (direction, spread, concentration). OT routing replaces CKA-based fusion.

### 🥈 Paper 23 | ARM | ACL'25 | ML
**Motivation Score: 8/10**
- ✅ M1: MoE (Knowledge Experts) + routing
- ✅ M2: Activation-guided routing → doesn't capture knowledge distribution
- ⚠️ M3: LLM backbone
- ✅ M4: NLP (knowledge editing, nhưng MoE architecture phổ biến)
- **Lý do apply:** ARM dùng activation-guided routing (heuristic). Statistical signatures + OT routing provides principled alternative.

## 4.3 Papers 2025 Có Motivation Trung Bình (Score 5-7/10)

### Paper 09 | MoDE | NeurIPS'25 | ML/Multi
**Motivation Score: 7/10**
- ✅ M1: Modality-specific experts
- ⚠️ M2: Expert isolation by modality (not really routing) → OT routing less applicable
- ✅ M3: Unified model backbone
- **Lý do:** Routing theo modality → fixed, không cần OT. Nhưng anti-drift cho backbone hữu ích.

### Paper 21 | PLAN | ICCV'25 | ML
**Motivation Score: 7/10**
- ✅ M1: Orthogonal basis vectors per task
- ⚠️ M2: Orthogonal allocation ≠ routing (pre-determined), nhưng distribution signatures có thể guide allocation
- ✅ M3: Shared backbone
- **Lý do:** PLAN allocate trước, không route at inference. Nhưng signatures có thể guide better allocation.

### Paper 08 | CaLoRA | NeurIPS'25 | ML
**Motivation Score: 6/10**
- ✅ M1: LoRA branches + causal analysis
- ⚠️ M2: Gradient projection based on task correlation — already somewhat distributional
- ⚠️ M3: LoRA-level, not backbone
- **Lý do:** CaLoRA đã dùng causal attribution → more sophisticated than simple gating. OT routing vẫn có thể improve nhưng gap nhỏ hơn.

### Paper 18 | Instruction-Grounded VP | ICCV'25 | ML/Multi
**Motivation Score: 6/10**
- ✅ M1: Mixture of visual projectors
- ⚠️ M2: Expert recommendation + pruning → OT could improve recommendation
- ⚠️ M3: VLM backbone shared
- **Lý do:** Projector-level MoE. OT routing applicable nhưng projector-specific.

### Paper 17 | TWIST&SCOUT | ICCV'25 | NLP
**Motivation Score: 5/10**
- ✅ M1: Twin experts (frozen + learnable)
- ❌ M2: No routing mechanism (fixed twin structure) — khó apply OT
- ✅ M3: Shared model backbone
- **Lý do:** Twin expert structure cố định → không có routing để upgrade. Chỉ Component 3 (anti-drift) applicable.

### Paper 44 | SEFE | ICML'25 | ML/Multi
**Motivation Score: 6/10**
- ✅ M1: RegLoRA (regularized LoRA) — multi-module
- ⚠️ M2: Regularization-based, not routing
- ⚠️ M3: Shared backbone
- **Lý do:** SEFE phân loại forgetting (superficial vs essential). Signatures có thể detect loại forgetting nào.

### Paper 61 | LoRA- | CVPR'25 | ML
**Motivation Score: 6/10**
- ⚠️ M1: LoRA subtraction (not standard MoE routing)
- ⚠️ M2: Drift-Resistant Space = alternative approach, OT routing không trực tiếp applicable
- ✅ M3: Drift là central problem → directly relevant to Component 3
- **Lý do:** Concept DRS và Component 3 (anti-drift) complementary. Có thể combine signatures + DRS.

### Paper 77 | LDC | ECCV'24 | ML
**Motivation Score: 6/10**
- ❌ M1: Single model + lightweight drift module
- ❌ M2: No routing
- ✅ M3: Drift compensation → directly relevant to Component 3
- **Lý do:** LDC concept trực tiếp liên quan Component 3 nhưng single-model → cần adapt to modular setting.

## 4.4 Papers KHÔNG có motivation (Score < 5)

Các nhóm papers KHÔNG phù hợp apply:
- **Knowledge Editing papers** (03, 10, 12, 22, 25, 36, 37, 38, 42, 50): Fact-level editing, không phải representation-level CL
- **Benchmark/Analysis papers** (34, 37, 48, 52, 90): Không có model để apply
- **Training-free/Data-level papers** (24, 28, 32, 55, 58, 89): Không có modular architecture
- **Prompt-based papers** (46, 56, 68, 87, 100, 105, 109): Prompt pool ≠ modular experts
- **Single-model non-geometric** (04, 11, 16, 40, 79, 95, 97, 104): Không có submodule + routing

---

# PHẦN 5: LỌC PAPERS KHẢ THI TRÊN T4/P100 (16GB VRAM)

## 5.1 Tiêu Chí GPU Feasibility

| Factor | T4/P100 Compatible | Cần > 16GB |
|--------|-------------------|------------|
| ViT-B/ViT-L + LoRA | ✅ | |
| CLIP ViT-B + adapters | ✅ | |
| BERT/RoBERTa | ✅ | |
| LLaMA-7B + LoRA (QLoRA 4-bit) | ✅ (borderline) | |
| LLaMA-7B full fine-tune | | ❌ |
| LLaMA-13B+ | | ❌ |
| LLaVA-7B + LoRA | ✅ (tight) | |
| LLaVA-13B+ | | ❌ |
| Diffusion models (SD) | ⚠️ depends | |

## 5.2 Bảng Feasibility — Papers Có Motivation Cao

| Rank | Paper | Motivation | GPU Feasible | Base Model | Code | Tổng đánh giá |
|------|-------|-----------|-------------|------------|------|---------------|
| ⭐1 | **35. Feature Distributions** | 8/10 | ✅ Likely (PEFT on LLM, small modules) | LLM + PEFT blocks | ❌ | **TOP PICK NLP** — closest to idea, PEFT = low VRAM |
| ⭐2 | **82. MoE-Adapters** | 8/10 | ✅ (CLIP ViT-B/L + adapters) | CLIP ViT | ✅ github | **TOP PICK ML** — standard MoE, clear upgrade path, có code |
| ⭐3 | **41. TreeLoRA** | 9/10 | ✅ (ViT) / ⚠️ (LLM, depends on size) | ViT + LLM | ❌ | **TOP PICK ML** — tree structure natural for signatures |
| ⭐4 | **01. GainLoRA** | 9/10 | ⚠️ Depends on LLM size (7B QLoRA OK) | LLM + LoRA | ❌ | **TOP PICK NLP** — nếu LLM ≤ 7B |
| 5 | **02. MINGLE** | 9/10 | ⚠️ Test-time merging may need multiple models loaded | MoE experts | ❌ | Phức tạp, nhưng high motivation |
| 6 | **14. SMoLoRA** | 8/10 | ⚠️ (LLaVA-7B + LoRAs → tight) | LLaVA + LoRA | ✅ github | VL, có code, tight memory |
| 7 | **27. HiDe-LLaVA** | 8/10 | ⚠️ (LLaVA + expansion → tight/infeasible) | LLaVA + expansion | ❌ | Architecture growth → memory grows |
| 8 | **23. ARM** | 8/10 | ⚠️ Depends on LLM base | LLM + MoE | ❌ | KE domain, phức tạp |
| 9 | **09. MoDE** | 7/10 | ⚠️ MM model size varies | Unified MM model | ❌ | Multimodal, not pure routing |
| 10 | **21. PLAN** | 7/10 | ✅ (LoRA-based, small modules) | Pre-trained + LoRA | ❌ | Allocation, not routing |

## 5.3 Top Recommendations — Ưu tiên ML/NLP + T4/P100 Feasible

### 🏆 Recommendation #1: Paper 35 — Feature Distributions (ICML'25)
- **Domain:** NLP (LLM Continual Learning)
- **Why:** Đây là paper ĐÃ dùng concept "feature distribution" cho module selection → **closest prior work** và **tốt nhất để demonstrate upgrade**. Thay mean-vector bằng vMF signature + thay selection heuristic bằng OT routing → clear, publishable contribution.
- **GPU:** PEFT blocks = lightweight, likely feasible on T4
- **Risk:** Không có public code → phải reimplement

### 🏆 Recommendation #2: Paper 82 — MoE-Adapters (CVPR'24)
- **Domain:** ML/Multi (VLM Continual Learning)
- **Why:** Standard MoE gating → **easiest upgrade path** to OT routing. Well-established benchmark. Có public code (github). CLIP-based → T4 feasible.
- **GPU:** ✅ CLIP ViT-B + adapters fit T4 easily
- **Risk:** VL domain (not pure NLP), nhưng methodology general

### 🏆 Recommendation #3: Paper 41 — TreeLoRA (ICML'25)
- **Domain:** ML (ViTs + LLMs)
- **Why:** Hierarchical structure rất phù hợp cho statistical signatures (signature tại mỗi tree node). Gradient-similarity → natural upgrade to distribution-based similarity. ICML'25 = strong baseline.
- **GPU:** ✅ cho ViT experiments. ⚠️ cho LLM tùy size.
- **Risk:** Không có code, phức tạp hơn (tree structure + bandit)

### 🏆 Recommendation #4: Paper 01 — GainLoRA (NeurIPS'25)
- **Domain:** NLP (LLM Continual Learning)
- **Why:** LoRA branches + gating = classic substrate cho OT routing upgrade. NeurIPS'25 = top venue. LLM CL = hot topic.
- **GPU:** ⚠️ Nếu base model ≤ 7B + QLoRA → feasible. Nếu > 13B → không.
- **Risk:** Không có code, LLM base model size uncertain

### 🏆 Recommendation #5: Paper 14 — SMoLoRA (ICCV'25)
- **Domain:** ML/Multi (VL Instruction Tuning)
- **Why:** Dual-routing concept → OT có thể unify. Có code (github). ICCV'25.
- **GPU:** ⚠️ LLaVA-7B + multiple LoRAs → tight on T4 nhưng có thể feasible với optimization.
- **Risk:** VL domain, memory tight

## 5.4 Bảng Tóm Tắt Ưu Tiên

| Priority | Paper | Domain | Motivation | GPU | Code | Action |
|----------|-------|--------|-----------|-----|------|--------|
| **1st** | 35 Feature Dist | NLP | 8 | ✅ | ❌ | Reimplement + upgrade distribution + OT |
| **2nd** | 82 MoE-Adapters | ML | 8 | ✅ | ✅ | Direct upgrade gating → OT routing |
| **3rd** | 41 TreeLoRA | ML | 9 | ✅/⚠️ | ❌ | Upgrade gradient-similarity → distribution signatures |
| **4th** | 01 GainLoRA | NLP | 9 | ⚠️ | ❌ | If LLM ≤ 7B, upgrade gating → OT |
| **5th** | 14 SMoLoRA | ML/VL | 8 | ⚠️ | ✅ | Unify dual routing → OT, có code |

---

# PHẦN 6: TỔNG KẾT & KHUYẾN NGHỊ

## 6.1 Tóm Tắt Đánh Giá

| Dimension | Assessment | Chi tiết |
|-----------|-----------|----------|
| **Novelty** | 🟢 **CAO** | 4 novelty gaps confirmed. Grassmannian MoE là rủi ro cao nhất nhưng khác mục đích |
| **Soundness** | 🟢 **HỢP LÝ** | 3 components có cơ sở lý thuyết, consistent nội bộ, synergistic |
| **Motivation cho 2025** | 🟢 **MẠNH** | 8+ papers có architecture phù hợp để apply. Xu hướng submodule+routing support idea |
| **T4/P100 Feasibility** | 🟡 **KHẢ THI CÓ ĐIỀU KIỆN** | 3-5 papers feasible (PEFT/CLIP-based). LLM >7B cần QLoRA hoặc smaller model |

## 6.2 Chiến Lược Đề Xuất

### Phase 1: Proof-of-concept (1-2 tháng)
- **Target:** Paper 82 (MoE-Adapters) — có code, T4 feasible, clear upgrade path
- **Goal:** Implement statistical signatures (vMF) + OT routing thay thế standard gating
- **Validation:** So sánh với baseline MoE gating trên same benchmarks

### Phase 2: Main contribution (2-3 tháng)
- **Target:** Paper 35 (Feature Distributions) hoặc Paper 01 (GainLoRA)
- **Goal:** Full framework với 3 components (signatures + OT + anti-drift)
- **Contribution:** Demonstrate superior performance qua principled routing + backbone protection

### Phase 3: Paper writing
- **Position:** "From Gating to Matching: Statistical Knowledge Signatures with Optimal Transport Routing for Continual Learning"
- **Claim:** Principled routing via distribution matching outperforms heuristic gating in modular CL

## 6.3 Rủi Ro & Mitigation

| Risk | Level | Mitigation |
|------|-------|-----------|
| Grassmannian MoE tiếp cận CL | Medium | Differentiate: knowledge characterization vs routing entropy control |
| OT inference overhead | Medium | Sinkhorn with few iterations + ε-regularization |
| Lack of code for most targets | Medium | Start with Paper 82 (có code) |
| vMF not suitable for all feature spaces | Low | Test multiple distributions; fallback to GMM |
| Combined overhead too high for T4 | Medium | Start with small-scale experiments (ViT-B) |

---

*Generated: Analysis of new_idea_modifier.txt against 109 surveyed papers + ~30 additional papers*
*Focus: Novelty, Soundness, Motivation for 2025 papers, T4/P100 Feasibility*
