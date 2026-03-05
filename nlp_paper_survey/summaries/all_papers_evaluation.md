# Đánh giá 109 Papers cho Simple Idea
## (Continual Learning Survey 2023-2025)

### Tiêu chí đánh giá Simple Idea:
1. **Single model** — không có submodule/adaptive/multi-module
2. **Geometric feature distribution** — sử dụng/có thể áp dụng thống kê hình học cho feature space
3. **Anti-forgetting via representation drift** — chống quên qua kiểm soát trôi biểu diễn
4. **External knowledge calibration** — hiệu chỉnh phân phối bằng nguồn tri thức ngoài
5. **Chưa model hình học** — chưa đề xuất tường minh cơ chế hình học → còn chỗ đóng góp

### Ký hiệu:
- ✅ PHÙ HỢP (Score ≥ 7): Paper phù hợp làm target paper cho simple_idea
- ⚠️ THAM KHẢO (Score 4-6): Có ý tưởng liên quan, tham khảo thêm
- ❌ KHÔNG PHÙ HỢP (Score ≤ 3): Không phù hợp cho simple_idea

---

# ===================== 2025 — NeurIPS (12 papers) =====================

## Paper 01 | GainLoRA | NeurIPS 2025 | [LoRA]
**Title:** Gated Integration of Low-Rank Adaptation for CL of LLMs
**Method:** Expands new LoRA branch per task + gating modules to integrate old/new branches
**Architecture:** Multi-module (LoRA branches + gating modules)
**Anti-forgetting:** Gating minimizes influence of new LoRA on old tasks
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Multi-module LoRA architecture, gating = adaptive submodule. Vi phạm single-model constraint.

---

## Paper 02 | MINGLE | NeurIPS 2025 | [LoRA]
**Title:** Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging
**Method:** MoE architecture + Null-Space Constrained Gating (NSCG) + Adaptive Relaxation Strategy
**Architecture:** Multi-module (multiple low-rank experts + gating mechanism)
**Anti-forgetting:** Null-space projection preserves prior routing patterns
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** MoE + LoRA experts + gating = multi-module. Test-time merging, không phải single model.

---

## Paper 03 | MemEIC | NeurIPS 2025 | [KE]
**Title:** Continual and Compositional Knowledge Editing
**Method:** Hybrid external-internal editor: dual external memory + dual LoRA adapters + knowledge connector
**Architecture:** Multi-module (external memory + LoRA + connector)
**Anti-forgetting:** Cross-modal evidence retrieval + disentangled parameter updates
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Knowledge editing ở mức factual, multi-module (memory + LoRA + connector).

---

## Paper 04 | InternAL | NeurIPS 2025 | [LLM-CL]
**Title:** Investigating and Mitigating CF in Medical Knowledge Injection
**Method:** Probes LLM's internal knowledge related to injection target + augments training data
**Architecture:** Single model (augmented training only)
**Anti-forgetting:** Proximity-dependent forgetting analysis → internal knowledge augmentation
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Single model nhưng data augmentation strategy, không model feature geometry. Insight về proximity-dependent forgetting thú vị nhưng approach là data-level.

---

## Paper 05 | Bisecle | NeurIPS 2025 | [VL]
**Title:** Binding and Separation in CL for Video Language Understanding
**Method:** Multi-directional supervision + contrastive prompt learning
**Architecture:** Prompt-based (contrastive prompt learning mechanism)
**Anti-forgetting:** Pattern separation orthogonalizes overlapping representations
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Prompt-based + video-language, không single model đơn thuần.

---

## Paper 06 | CMCL | NeurIPS 2025 | [MM]
**Title:** Continual Multimodal Contrastive Learning
**Method:** Gradient projection from dual sides onto subspaces preventing interference
**Architecture:** Single model (gradient projection approach)
**Anti-forgetting:** Dual stability/plasticity upper bounds + gradient subspace projection
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** Single model + gradient projection. Có theoretical bounds cho stability/plasticity. Nhưng multimodal contrastive, không phải classification. Không model feature geometry tường minh.

---

## Paper 07 | Low-rank Forgetting Analysis | NeurIPS 2025 | [Analysis]
**Title:** Demystifying Language Model Forgetting with Low-rank Example Associations
**Method:** Phát hiện M×N forgetting matrices có low-rank structure → matrix completion dự đoán forgetting
**Architecture:** Analysis paper, no specific method
**Anti-forgetting:** Replay forgotten examples dựa trên dự đoán
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Analysis paper, không propose method. Nhưng insight: forgetting patterns có cấu trúc low-rank → liên quan hình học.

---

## Paper 08 | CaLoRA | NeurIPS 2025 | [LoRA]
**Title:** Backward Transfer via Causal-Aware LoRA in CL
**Method:** Parameter-level causal attribution + cross-task gradient adaptation
**Architecture:** Multi-module (LoRA branches + causal analysis)
**Anti-forgetting:** Gradient projection based on task correlation + affinity
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** LoRA-based multi-module.

---

## Paper 09 | MoDE | NeurIPS 2025 | [MM]
**Title:** Mitigating Intra- and Inter-modal Forgetting in Unified Multimodal Models
**Method:** Modality-Decoupled Experts: isolates modality-specific updates + knowledge distillation
**Architecture:** Multi-module (modality-specific experts)
**Anti-forgetting:** Expert isolation prevents gradient conflict between modalities
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Multi-module expert strategy.

---

## Paper 10 | MEMOIR | NeurIPS 2025 | [KE]
**Title:** Lifelong Model Editing with Minimal Overwrite
**Method:** Residual memory module + sample-dependent sparse masks
**Architecture:** Multi-module (residual memory appended to model)
**Anti-forgetting:** Sparse activation patterns confine edits to distinct parameter subsets
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Knowledge editing + residual memory = multi-module.

---

## Paper 11 | SERS | NeurIPS 2025 | [LLM-CL]
**Title:** Self-Evolving Pseudo-Rehearsal for CF with Task Similarity in LLMs
**Method:** Pseudo-rehearsal + dynamic regularizer driven by Wasserstein distance giữa task distributions
**Architecture:** Single model + pseudo-rehearsal strategy
**Anti-forgetting:** Wasserstein distance measures task similarity → adjusts regularizer
### ⚠️ THAM KHẢO (Score: 4/10)
**Lý do:** Wasserstein distance giữa task distributions = measure hình học. Nhưng pseudo-rehearsal là data strategy, không model feature geometry tường minh.

---

## Paper 12 | CARML | NeurIPS 2025 | [KE]
**Title:** Reliable Lifelong Multimodal Editing: Conflict-Aware Retrieval
**Method:** Retrieval-augmented editing + intra-modal uncertainty + inter-modal conflict quantification + prompt prefixes
**Architecture:** Multi-module (retrieval + scope classifier + prompt generation)
**Anti-forgetting:** Dynamic retrieval + hard correction on output logits
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Knowledge editing + retrieval + multi-module.

---

# ===================== 2025 — ICCV (9 papers) =====================

## Paper 13 | MG-CLIP | ICCV 2025 | [VL]
**Title:** Mind the Gap: Preserving and Compensating for Modality Gap in CLIP-Based CL
**Method:** Modality gap preservation (regularization) + modality gap compensation (adaptation)
**Architecture:** CLIP-based + adaptation approach
**Anti-forgetting:** Modality gap phản ánh mức độ bảo toàn pre-trained knowledge
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** Modality gap analysis = geometric concept. Nhưng CLIP-based, không hoàn toàn single model. Insight: modality gap ↔ forgetting level là geometry observation.

---

## Paper 14 | SMoLoRA | ICCV 2025 | [IT]
**Title:** Exploring and Defying Dual CF in Continual Visual Instruction Tuning
**Method:** Separable Mixture of LoRA + dual-routing (visual understanding + instruction following)
**Architecture:** Multi-module (dual LoRA routing)
**Anti-forgetting:** Specialized adaptation via separable routing
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** MoE + LoRA = multi-module.

---

## Paper 15 | DMNSP | ICCV 2025 | [VL]
**Title:** Dynamic Multi-Layer Null Space Projection for VL CL
**Method:** Projects updates into null space of previous tasks across multiple layers
**Architecture:** Adapter-based VLM
**Anti-forgetting:** Null space projection per modality
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Adapter-based + null space projection. Multi-module by design.

---

## Paper 16 | QUAD | ICCV 2025 | [VL]
**Title:** Ask and Remember: Questions-Only Replay for Continual VQA
**Method:** Question-only replay + Attention Consistency Distillation (intra/inter-modal)
**Architecture:** Single model + attention distillation
**Anti-forgetting:** Attention consistency maintains visual-linguistic associations
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Replay strategy + attention distillation, không model feature geometry.

---

## Paper 17 | TWIST&SCOUT | ICCV 2025 | [MM]
**Title:** Grounding Multimodal LLM-Experts by Forget-Free Tuning
**Method:** Twin-expert stepwise tuning: frozen module + learnable module
**Architecture:** Multi-module (twin experts)
**Anti-forgetting:** Frozen expert preserves old knowledge
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Twin expert = multi-module.

---

## Paper 18 | Instruction-Grounded VP | ICCV 2025 | [IT]
**Title:** Instruction-Grounded Visual Projectors for CL of Generative VLMs
**Method:** Mixture of visual projectors + expert recommendation + expert pruning
**Architecture:** Multi-module (mixture of projectors)
**Anti-forgetting:** Expert pruning reduces interference
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Mixture of experts = multi-module.

---

## Paper 19 | ENGINE | ICCV 2025 | [VL]
**Title:** External Knowledge Injection for CLIP-Based CIL
**Method:** Dual-branch injection (visual augmentation + GPT-4 textual descriptors) + post-tuning re-ranking
**Architecture:** Dual-branch (visual + textual)
**Anti-forgetting:** External knowledge (GPT-4) compensates overwritten features
### ⚠️ THAM KHẢO (Score: 4/10)
**Lý do:** "External knowledge" gần với simple_idea concept. Nhưng dual-branch = multi-module, và external knowledge ở đây = text descriptions từ GPT-4, không phải geometric calibration.

---

## Paper 20 | Dual Drift VQA | ICCV 2025 | [VL]
**Title:** Overcoming Dual Drift for Continual Long-Tailed VQA
**Method:** Identifies inner-task prototype drift (long-tail imbalance) + inter-task prototype drift (learning new tasks)
**Architecture:** VQA model with prototype analysis
**Anti-forgetting:** Stabilize prototypes + maintain balanced representations
### ✅ PHÙ HỢP (Score: 7/10)
**Lý do:** **Prototype drift** = trực tiếp liên quan đến center drift trong simple_idea. Dual drift analysis (inner-task + inter-task) = mở rộng concept "center drift" từ simple_idea. Paper address prototype drift nhưng CHƯA dùng geometric distribution modeling (vMF/Bingham) → simple_idea có thể đóng góp cơ chế thống kê hình học. Long-tailed setting tạo thêm challenge thú vị.

---

## Paper 21 | PLAN | ICCV 2025 | [LoRA]
**Title:** Proactive Low-Rank Allocation for Continual Learning
**Method:** Orthogonal basis vectors per task + perturbation-based allocation strategy
**Architecture:** LoRA-based multi-module
**Anti-forgetting:** Orthogonal subspace allocation minimizes interference
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** LoRA-based multi-module.

---

# ===================== 2025 — ACL (13 papers) =====================

## Paper 22 | KDE | ACL 2025 | [KE]
**Title:** Knowledge Decoupling via Orthogonal Projection for Lifelong Editing
**Method:** Stores basis vectors in knowledge cache + projects gradient orthogonally
**Architecture:** LLM + extra trainable modules + knowledge cache
**Anti-forgetting:** Orthogonal projection decouples knowledge
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Knowledge editing + extra modules.

---

## Paper 23 | ARM | ACL 2025 | [KE]
**Title:** Serial Lifelong Editing via Mixture of Knowledge Experts
**Method:** MoE scheme + Activation-guided Routing
**Architecture:** Multi-module (mixture of knowledge experts)
**Anti-forgetting:** Domain-specific experts + complete overwrite
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** KE + MoE = multi-module.

---

## Paper 24 | Stability Gap CPT | ACL 2025 | [LLM-CL]
**Title:** Efficient Domain Continual Pretraining by Mitigating Stability Gap
**Method:** Epoch strategy + data sampling (domain relevance + corpus distribution)
**Architecture:** Single model (training strategy)
**Anti-forgetting:** Stability gap mitigation via training recipe
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Training strategy, không model feature geometry. Insight "stability gap" thú vị nhưng approach là data-level.

---

## Paper 25 | NSE | ACL 2025 | [KE]
**Title:** Neuron-Level Sequential Editing for LLMs
**Method:** Iteratively selects neurons based on activation values for editing
**Architecture:** Single model (neuron selection)
**Anti-forgetting:** Optimizes hidden states with original weights + selective neuron editing
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Knowledge editing ở neuron level, fact-based.

---

## Paper 26 | CLoRA | ACL 2025 | [LoRA]
**Title:** Controlled Low-Rank Adaptation with Subspace Regularization
**Method:** Constrains updating matrix's null space direction
**Architecture:** LoRA-based
**Anti-forgetting:** Null space regularization limits output change
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** LoRA-based + subspace constraint.

---

## Paper 27 | HiDe-LLaVA | ACL 2025 | [IT]
**Title:** Hierarchical Decoupling for Continual IT of Multimodal LLM
**Method:** Task-specific expansion + task-general fusion based on CKA similarity
**Architecture:** Multi-module (expansion + fusion)
**Anti-forgetting:** CKA similarity guides layer-wise handling
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Task-specific expansion = multi-module.

---

## Paper 28 | MMER | ACL 2025 | [MM]
**Title:** Multi-Modality Expansion and Retention through Parameter Merging
**Method:** Training-free: merges LLM parameters + binary masks for modality separation
**Architecture:** Training-free merging (not actually single model at inference)
**Anti-forgetting:** Binary masks decouple modality-specific parameters
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Parameter merging + binary masks, not single-model CL.

---

## Paper 29 | GORP | ACL 2025 | [LoRA]
**Title:** Continual Gradient Low-Rank Projection Fine-Tuning
**Method:** Combines full + low-rank parameters in unified low-rank gradient subspace
**Architecture:** LoRA-based
**Anti-forgetting:** Low-rank gradient subspace preserves knowledge
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** LoRA-based.

---

## Paper 30 | DGAR | ACL 2025 | [NLP-Task]
**Title:** Generative Adaptive Replay CL Model for Temporal KG Reasoning
**Method:** Diffusion model generates historical entity distribution representations + layer-by-layer adaptive replay
**Architecture:** TKGR model + pre-trained diffusion model
**Anti-forgetting:** Diffusion generates historical distribution, enhances common features between historical/current
### ⚠️ THAM KHẢO (Score: 6/10)
**Lý do:** **Generates historical entity distributions** bằng diffusion model → rất liên quan concept "modeling distribution" của simple_idea. Tuy nhiên: (a) dùng diffusion model riêng = multi-module, (b) domain là Temporal KG không phải classification. Idea "enhance common features between distributions" có thể tham khảo.

---

## Paper 31 | MoNIM | ACL 2025 | [LLM-CL]
**Title:** Learn to Memorize: Scalable CL with Mixture-of-Neighbors Induction Memory
**Method:** kNN-LM reconceptualized as learnable MoNIM bypass layer
**Architecture:** Multi-module (bypass FFN-like layer)
**Anti-forgetting:** Non-parametric memory integration
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Multi-module (bypass layer).

---

## Paper 32 | KPIG | ACL 2025 | [IT]
**Title:** Don't Half-listen: Key-part Information in CIT
**Method:** Information gain on masked parts → dynamic replay + refined training objective
**Architecture:** Single model + replay strategy
**Anti-forgetting:** Key-part information gain guides replay
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Data replay strategy, không model geometry.

---

## Paper 33 | Recurrent-KIF | ACL 2025 | [LLM-CL]
**Title:** Recurrent Knowledge Identification and Fusion
**Method:** Inner loop (rapid adaptation + parameter identification) + outer loop (redundant pruning + key merging)
**Architecture:** Multi-loop optimization (inner-outer loops)
**Anti-forgetting:** Dynamic parameter importance + knowledge pruning/merging
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Multi-loop + parameter importance estimation, không model feature distribution.

---

## Paper 34 | TiC-LM | ACL 2025 | [Benchmark]
**Title:** Web-Scale Benchmark for Time-Continual LLM Pretraining
**Method:** Benchmark (114 Common Crawl dumps)
**Architecture:** N/A (benchmark)
### ❌ KHÔNG PHÙ HỢP (Score: 0/10)
**Lý do:** Benchmark, không phải method.

---

# ===================== 2025 — ICML (14 papers) =====================

## Paper 35 | Feature Distributions PE-CL | ICML 2025 | [LLM-CL]
**Title:** Exploiting Presentative Feature Distributions for PE-CL of LLMs
**Method:** Characterizes each PEFT block by "presentative feature distribution" → selects/reuses PEFT blocks based on distribution similarity
**Architecture:** Multi-PEFT-block (expanding or reusing)
**Anti-forgetting:** Distribution similarity selects appropriate PEFT block → reduces redundancy
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** "Feature distribution" matching concept trực tiếp liên quan simple_idea. Nhưng PEFT-block expanding = multi-module, và distribution ở đây dùng cho block selection, không phải geometric anti-forgetting. Tham khảo cách model "presentative feature distribution".

---

## Paper 36 | RLEdit | ICML 2025 | [KE]
**Title:** Reinforced Lifelong Editing for Language Models
**Method:** RL-based editing policy via lightweight hypernetwork
**Architecture:** LLM + hypernetwork (multi-module)
**Anti-forgetting:** RL learns adaptive editing policy from history
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Knowledge editing + hypernetwork = multi-module.

---

## Paper 37 | WikiBigEdit | ICML 2025 | [KE]
**Title:** Understanding the Limits of Lifelong Knowledge Editing
**Method:** Large-scale benchmark (500K QA pairs from Wikidata edits)
**Architecture:** N/A (benchmark)
### ❌ KHÔNG PHÙ HỢP (Score: 0/10)
**Lý do:** Benchmark/analysis, không phải method.

---

## Paper 38 | Knowledge Swapping | ICML 2025 | [KE]
**Title:** Knowledge Swapping via Learning and Unlearning
**Method:** "Learning Before Forgetting" two-stage pipeline
**Architecture:** Single model + constrained optimization
**Anti-forgetting:** Injects new knowledge → then unlearns old
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Knowledge editing task (inject/remove facts), not representation-level.

---

## Paper 39 | CPT Learning Dynamics | ICML 2025 | [Analysis]
**Title:** Learning Dynamics in Continual Pre-Training for LLMs
**Method:** Analysis: distribution shift effect + learning rate annealing effect → CPT scaling law
**Architecture:** Analysis paper
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Analysis paper. Insight "distribution shift effect" thú vị.

---

## Paper 40 | Large CIT | ICML 2025 | [IT]
**Title:** Large Continual Instruction Assistant
**Method:** EMA update + plasticity-stability coefficient + exemplar replay
**Architecture:** Single model + EMA
**Anti-forgetting:** EMA + dynamic coefficient + exemplar replay
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** EMA + replay, không model feature geometry.

---

## Paper 41 | TreeLoRA | ICML 2025 | [LoRA]
**Title:** Efficient CL via Layer-Wise LoRAs Guided by Hierarchical Gradient-Similarity Tree
**Method:** Layer-wise LoRA allocation based on gradient similarity tree + multi-armed bandit
**Architecture:** Multi-module (hierarchical LoRA layers)
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** LoRA multi-module.

---

## Paper 42 | ALKN | ICML 2025 | [KE]
**Title:** Adaptive Localization of Knowledge Negation for Continual LLM Unlearning
**Method:** Dynamic masking + sparsified gradients for unlearning
**Architecture:** Single model + masking
**Anti-forgetting:** Localizes gradient updates to critical parameters
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Knowledge editing/unlearning.

---

## Paper 43 | HippoRAG 2 | ICML 2025 | [LLM-CL]
**Title:** From RAG to Memory: Non-Parametric CL for LLMs
**Method:** Knowledge graph-based memory for non-parametric CL
**Architecture:** External memory (knowledge graph)
**Anti-forgetting:** Non-parametric — no parameter updates, no forgetting
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Non-parametric approach, hoàn toàn khác paradigm.

---

## Paper 44 | SEFE | ICML 2025 | [MM]
**Title:** Superficial and Essential Forgetting Eliminator for Multimodal CIT
**Method:** Answer Style Diversification (superficial forgetting) + RegLoRA (essential forgetting)
**Architecture:** Multi-module (RegLoRA = regularized LoRA)
**Anti-forgetting:** Phân loại forgetting thành superficial vs essential
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** LoRA-based multi-module. Insight phân loại forgetting thú vị.

---

## Paper 45 | LADA | ICML 2025 | [VL]
**Title:** Scalable Label-Specific CLIP Adapter for CL
**Method:** Label-specific memory units appended to frozen CLIP + feature distillation
**Architecture:** Multi-module (label-specific adapters + frozen CLIP)
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Label-specific adapters = multi-module.

---

## Paper 46 | KA-Prompt | ICML 2025 | [VL]
**Title:** Componential Prompt-Knowledge Alignment for Domain Incremental Learning
**Method:** Decomposes prompts into functional components + alignment loss
**Architecture:** Prompt-based
**Anti-forgetting:** Componential alignment ensures consistent domain specialization
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Prompt-based approach.

---

## Paper 47 | Proxy-FDA | ICML 2025 | [VL]
**Title:** Proxy-based Feature Distribution Alignment for Fine-tuning VFMs without Forgetting
**Method:** Nearest neighbor graphs → compact proxies representing feature distribution → aligns evolving distribution with proxies during fine-tuning
**Architecture:** Single model + lightweight proxies
**Anti-forgetting:** Feature distribution alignment với pre-trained distribution structure
### ✅ PHÙ HỢP (Score: 8/10)
**Lý do:** **Feature distribution alignment** — trực tiếp liên quan simple_idea! Proxy-FDA model phân phối feature space bằng nearest-neighbor graph proxies và align feature khi fine-tune. Exemplar-free. Tuy nhiên: (1) dùng proxy + NN graph, CHƯA dùng statistical tool hình học (vMF/Bingham), (2) CHƯA có external knowledge calibration. → Simple_idea có thể đóng góp: thay proxy bằng geometric distribution modeling + external knowledge calibration.

---

## Paper 48 | Angle Matters | ICML 2025 | [Analysis]
**Title:** Understanding Forgetting of Replay-based CL via Feature Learning: Angle Matters
**Method:** Lý thuyết: angle between task signal vectors determines forgetting degree
**Architecture:** Analysis paper (theoretical framework)
**Key insight:** Aligned tasks (smaller angle) → replay hiệu quả hơn; orthogonal tasks → khó mitigate
### ✅ PHÙ HỢP (Score: 7/10)
**Lý do:** **Geometric analysis** — trực tiếp support simple_idea! Chứng minh **angle trong feature space** quyết định mức độ forgetting. Đây là theoretical backing cho simple_idea: nếu model feature geometry (angles, distributions) thì có thể control forgetting. Paper này là analysis, simple_idea có thể dùng insight này để justify geometric approach.

---

# ===================== 2025 — ICLR (8 papers) =====================

## Paper 49 | LOIRE | ICLR 2025 | [LLM-CL]
**Title:** LifelOng learning via pre-trained LM gRowth Efficiently
**Method:** Plug-in layer growth — adds lightweight layers for new data
**Architecture:** Multi-module (growing architecture)
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Architecture growth = multi-module.

---

## Paper 50 | OOO | ICLR 2025 | [KE]
**Title:** On LLM Continual Unlearning
**Method:** Orthogonal LoRA + OOD detector
**Architecture:** Multi-module (orthogonal LoRA + OOD detector)
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Unlearning + multi-module.

---

## Paper 51 | SD-LoRA | ICLR 2025 | [LoRA]
**Title:** Scalable Decoupled Low-Rank Adaptation for CIL
**Method:** Decouples magnitude + direction of LoRA components, follows low-loss trajectory
**Architecture:** LoRA-based (but single LoRA, not expanding)
**Anti-forgetting:** Low-loss trajectory convergence for all tasks
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** LoRA-based. Nhưng insight về direction/magnitude decoupling →  geometric concept.

---

## Paper 52 | Spurious Forgetting | ICLR 2025 | [Analysis]
**Title:** Spurious Forgetting in CL of Language Models
**Method:** Phân biệt "task alignment loss" vs "true knowledge loss" + Freezing strategy
**Architecture:** Analysis + bottom-layer freezing
**Key insight:** Performance drop ≠ knowledge loss; often just task alignment shift
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** Insight quan trọng: spurious forgetting ≠ true forgetting. Orthogonal weight updates cause task alignment shifts. Liên quan simple_idea: nếu model geometry correctly, có thể phân biệt spurious vs true forgetting. Freezing strategy = simple baseline.

---

## Paper 53 | Function Vectors | ICLR 2025 | [IT]
**Title:** Unlocking Function Vectors for Mitigating CF in CIT
**Method:** Function vectors (compact functional representation) + FV regularization
**Architecture:** Single model + FV regularization
**Anti-forgetting:** Stabilize function vectors → prevent bias in function activation
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** Function vectors = compact representation of model functions. Insight: CF do **bias in function activation**, không phải overwriting task functions. FV regularization khá gần concept "stabilize representation" nhưng ở function level, không phải feature geometry level.

---

## Paper 54 | C-CLIP | ICLR 2025 | [VL]
**Title:** Multimodal Continual Learning for VLMs
**Method:** Novel framework preventing forgetting + enhancing new task learning
**Architecture:** CLIP-based
**Anti-forgetting:** Preserves zero-shot + few-shot capabilities
### ⚠️ THAM KHẢO (Score: 4/10)
**Lý do:** CLIP-based continual VL learning. Details limited from abstract.

---

## Paper 55 | Adapt-∞ | ICLR 2025 | [MM]
**Title:** Scalable Continual Multimodal IT via Dynamic Data Selection
**Method:** Dynamic data selection framework — selects beneficial samples adaptively
**Architecture:** Data selection strategy
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Data selection strategy, không model approach.

---

## Paper 56 | VL Synergy | ICLR 2025 | [VL]
**Title:** Vision and Language Synergy for Rehearsal Free CL
**Method:** Language as input for prompt generation + task-wise generators + soft task-ID
**Architecture:** Prompt-based + task-wise generators
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Prompt-based multi-module.

---

# ===================== 2025 — CVPR (5 papers) =====================

## Paper 57 | Language-Guided CBM | CVPR 2025 | [VL]
**Title:** Language Guided Concept Bottleneck Models for Interpretable CL
**Method:** Concept Bottleneck Layer aligned with CLIP → learns interpretable concepts
**Architecture:** CLIP + Concept Bottleneck
**Anti-forgetting:** Semantic concept consistency across tasks
### ⚠️ THAM KHẢO (Score: 4/10)
**Lý do:** Concept bottleneck = interpretable representation. Nhưng CLIP-based + bottleneck = multi-module.

---

## Paper 58 | AdaDARE-γ | CVPR 2025 | [MM]
**Title:** Balancing Stability and Plasticity in Multi-modal LLMs
**Method:** Adaptive parameter selection from fine-tuned model + controlled injection
**Architecture:** Fine-tuned model selection (training-free)
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Parameter selection/merging, not learning-based model.

---

## Paper 59 | GIFT | CVPR 2025 | [VL]
**Title:** Synthetic Data is Elegant GIFT for Continual VLMs
**Method:** Diffusion-generated images + contrastive distillation + Fisher-based weight consolidation
**Architecture:** VLM + diffusion model (for data generation)
**Anti-forgetting:** Contrastive distillation on synthetic image-text pairs + Fisher information weight consolidation
### ⚠️ THAM KHẢO (Score: 4/10)
**Lý do:** Fisher information = liên quan hình học (FIM = Riemannian metric). Nhưng dùng diffusion model riêng + distillation = multi-approach. Fisher weight consolidation là classical EWC concept.

---

## Paper 60 | CL-LoRA | CVPR 2025 | [LoRA]
**Title:** Continual Low-Rank Adaptation for Rehearsal-Free CIL
**Method:** Task-shared + task-specific dual adapters + orthogonal matrices + gradient reassignment
**Architecture:** Multi-module (dual adapters)
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** Dual-adapter = multi-module.

---

## Paper 61 | LoRA- | CVPR 2025 | [LoRA]
**Title:** LoRA Subtraction for Drift-Resistant Space in Exemplar-Free CL
**Method:** Subtracts LoRA weights of old tasks from pre-trained weights → Drift-Resistant Space (DRS)
**Architecture:** LoRA-based
**Anti-forgetting:** DRS stabilizes feature drift → enables triplet loss for plasticity
### ⚠️ THAM KHẢO (Score: 6/10)
**Lý do:** **Drift-Resistant Space** — trực tiếp address feature drift! Concept DRS = tạo space ổn định cho features. Exemplar-free. Nhưng LoRA-based. Insight: subtract old LoRA weights → creates drift-resistant subspace. Simple_idea có thể tham khảo concept "drift-resistant space" nhưng dùng geometric distribution thay vì LoRA subtraction.

---

# ===================== 2024 — NeurIPS (7 papers) =====================

## Paper 62 | ZAF | NeurIPS 2024 | [VL]
**Title:** Stabilizing Zero-Shot Prediction in Continual VL Tasks
**Method:** Zero-shot stability regularization + EMA-LoRA
**Architecture:** CLIP + LoRA
**Anti-forgetting:** Preserves pre-trained representations + EMA smoothing
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** LoRA-based approach.

---

## Paper 63 | RAIL | NeurIPS 2024 | [VL]
**Title:** Advancing Cross-domain Discriminability in CL of VLMs
**Method:** Recursive ridge regression adapter on pre-trained token representations
**Architecture:** CLIP + ridge regression adapter
**Anti-forgetting:** Task-adaptive classification maintaining zero-shot transferability
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Adapter-based.

---

## Paper 64 | Global Alignment | NeurIPS 2024 | [LLM-CL]
**Title:** Continual Learning with Global Alignment
**Method:** Composes task features from shared pre-trained token representations + global feature space
**Architecture:** Single shared feature space + task-specific composition weights
**Anti-forgetting:** Globally aligned features → natural knowledge transfer
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** Global shared feature space concept kha khá liên quan. Task features composed from shared tokens = geometric composition. Nhưng không model distribution tường minh.

---

## Paper 65 | CLAP4CLIP | NeurIPS 2024 | [VL]
**Title:** Continual Learning with Probabilistic Finetuning for VLMs
**Method:** Probabilistic modeling over visual-guided text features + distribution-level regularization + uncertainty estimation
**Architecture:** CLIP + probabilistic prompts
**Anti-forgetting:** Distribution-level regularization prevents forgetting
### ✅ PHÙ HỢP (Score: 7/10)
**Lý do:** **Probabilistic feature modeling + distribution-level regularization!** Trực tiếp liên quan simple_idea. CLAP4CLIP model uncertainty/distribution thay vì deterministic features. Distribution regularization = giữ distribution ổn định. Nhưng: dùng CLIP + prompts (không hoàn toàn single model), và distribution model ở đây là Gaussian, CHƯA geometric (vMF/Bingham). → Simple_idea có thể upgrade: geometric distribution modeling + external knowledge calibration.

---

## Paper 66 | TAALM | NeurIPS 2024 | [LLM-CL]
**Title:** Train-Attention: Meta-Learning Where to Focus in CKL
**Method:** Meta-learner predicts token-level importance weights
**Architecture:** LLM + meta-learner module
**Anti-forgetting:** Token importance weighting guides training
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Meta-learner = extra module.

---

## Paper 67 | ViLCo-Bench | NeurIPS 2024 | [Benchmark]
**Title:** VIdeo Language COntinual learning Benchmark
### ❌ KHÔNG PHÙ HỢP (Score: 0/10)
**Lý do:** Benchmark.

---

## Paper 68 | VPT Null Space | NeurIPS 2024 | [VL]
**Title:** Visual Prompt Tuning in Null Space for CL
**Method:** Null-space projection of prompt gradients
**Architecture:** Prompt-based + ViT
**Anti-forgetting:** Gradients in null space of previous tasks' feature space
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Prompt-based + null space projection.

---

# ===================== 2024 — ECCV (10 papers) =====================

## Paper 69 | RAPF | ECCV 2024 | [VL]
**Title:** CLIP Adaptive Representation Adjustment and Parameter Fusion
**Method:** Adaptive representation adjustment + parameter fusion adapter
**Architecture:** CLIP + adapter
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Adapter-based.

---

## Paper 70 | DIKI | ECCV 2024 | [VL]
**Title:** Mind the Interference: Retaining Pre-trained Knowledge in PE-CL of VLMs
**Method:** Fully residual mechanism for PE-CL (0.86% params)
**Architecture:** CLIP + residual adapters
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** PE/adapter-based.

---

## Paper 71 | Select & Distill | ECCV 2024 | [VL]
**Title:** Selective Dual-Teacher Knowledge Transfer for CL on VLMs
**Method:** Dual-teacher (fine-tuned + pre-trained) + selective distillation based on feature discrepancy
**Architecture:** Dual-teacher framework
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Dual-teacher = multi-model.

---

## Paper 72 | PILoRA | ECCV 2024 | [LoRA]
**Title:** Prototype Guided Incremental LoRA for Federated CIL
**Method:** Prototype representations guide LoRA adapter conditioning
**Architecture:** LoRA + prototypes (federated)
**Anti-forgetting:** Prototype-guided adaptation
### ⚠️ THAM KHẢO (Score: 4/10)
**Lý do:** LoRA-based nhưng dùng **prototypes** = liên quan simple_idea's center concept. Federated setting.

---

## Paper 73 | PromptCCD | ECCV 2024 | [LLM-CL]
**Title:** Learning Gaussian Mixture Prompt Pool for Continual Category Discovery
**Method:** Gaussian Mixture Model prompt pool for modeling feature distributions
**Architecture:** GMM prompt pool + pre-trained model
**Anti-forgetting:** GMM models category distributions for soft prompt assignment
### ⚠️ THAM KHẢO (Score: 6/10)
**Lý do:** **GMM distribution modeling!** Dùng Gaussian Mixture để model feature distribution → rất liên quan simple_idea. Nhưng: dùng cho prompt pool selection (category discovery), không phải anti-forgetting qua geometric modeling. Concept có thể tham khảo.

---

## Paper 74 | Anytime CL | ECCV 2024 | [VL]
**Title:** Anytime CL for Open Vocabulary Classification
**Method:** Framework for continuous adaptation of open-vocab classifiers
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)

---

## Paper 75 | CLIFF | ECCV 2024 | [VL]
**Title:** Continual Latent Diffusion for Open-Vocabulary Object Detection
**Method:** Variational Latent Sampler + Continual Diffusion Module for distribution transfer
**Architecture:** Diffusion-based detection
**Anti-forgetting:** Probabilistic object space + distribution transfer
### ⚠️ THAM KHẢO (Score: 4/10)
**Lý do:** Dùng **probabilistic object space** + distribution estimation (Variational Latent Sampler). Concept distribution transfer thú vị. Nhưng object detection specific + diffusion-based.

---

## Paper 76 | CLEO | ECCV 2024 | [VL]
**Title:** Continual Learning of Evolving Ontologies
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Ontology evolution, khác paradigm.

---

## Paper 77 | LDC | ECCV 2024 | [LLM-CL]
**Title:** Exemplar-free Continual Representation Learning via Learnable Drift Compensation
**Method:** Lightweight learnable module explicitly compensates representation drift + predicts/corrects drift in feature space
**Architecture:** Feature extractor + lightweight drift compensation module
**Anti-forgetting:** Trực tiếp predict + correct representation drift
### ✅ PHÙ HỢP (Score: 8/10)
**Lý do:** **Representation drift compensation** — trực tiếp cùng problem statement với simple_idea! LDC learn to predict + correct feature drift. Exemplar-free. NHƯNG: (1) dùng lightweight module riêng để compensate drift (= multi-module), (2) KHÔNG model geometric distribution (vMF/Bingham), (3) KHÔNG dùng external knowledge. → Simple_idea có thể đóng góp: thay learnable module bằng geometric statistical tool + external knowledge calibration, giữ single model.

---

## Paper 78 | Dual Teachers VLM | ECCV 2024 | [VL]
**Title:** Adapt without Forgetting: Distill Proximity from Dual Teachers
**Method:** Graph-based multi-modal proximity distillation + sample re-weighting
**Architecture:** Dual-teacher framework
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Dual-teacher multi-model.

---

# ===================== 2024 — ICML (2 papers) =====================

## Paper 79 | COPAL | ICML 2024 | [LLM-CL]
**Title:** Continual Pruning in Large Language Generative Models
**Method:** Sensitivity-guided pruning for continual adaptation
**Architecture:** Single model + pruning
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Pruning approach, không model feature geometry.

---

## Paper 80 | STELLA | ICML 2024 | [MM]
**Title:** Continual Audio-Video Pre-training with SpatioTemporal Localized Alignment
**Method:** Localized contrastive learning at spatiotemporal regions
**Architecture:** Audio-video model
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Audio-video specific, localized contrastive.

---

# ===================== 2024 — CVPR (7 papers) =====================

## Paper 81 | InfLoRA | CVPR 2024 | [LoRA]
**Title:** Interference-Free Low-Rank Adaptation for CL
**Method:** LoRA updates in orthogonal subspace to important directions of previous tasks
**Architecture:** LoRA-based
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** LoRA-based. Orthogonal subspace concept similar to null-space methods.

---

## Paper 82 | MoE Adapters | CVPR 2024 | [VL]
**Title:** Boosting CL of VLMs via MoE Adapters
**Method:** MoE adapter routing
**Architecture:** Multi-module (MoE adapters)
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** MoE = multi-module.

---

## Paper 83 | PriViLege | CVPR 2024 | [VL]
**Title:** Pre-trained VL Transformers Are Few-Shot Incremental Learners
**Method:** VLM + language-guided regularization for FSCIL
**Architecture:** VLM-based
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** VLM-based FSCIL, không model geometry.

---

## Paper 84 | Language-Guided Supervision | CVPR 2024 | [VL]
**Title:** Enhancing Visual CL with Language-Guided Supervision
**Method:** Textual descriptions as semantic anchors stabilizing feature representations
**Architecture:** Visual model + language supervision
**Anti-forgetting:** Language supervision provides semantic anchors → reduces feature drift
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** "Semantic anchors stabilize features" = concept gần với "external knowledge calibration" trong simple_idea. Language descriptions = external knowledge source. Nhưng implementation là language supervision, không phải geometric modeling.

---

## Paper 85 | LANDER | CVPR 2024 | [LLM-CL]
**Title:** Text-Enhanced Data-free Approach for Federated CIL
**Method:** Text descriptions → generate pseudo-features for rehearsal
**Architecture:** Federated + pseudo-feature generation
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Federated + pseudo-rehearsal.

---

## Paper 86 | Gen MM-CIL | CVPR 2024 | [MM]
**Title:** Generative Multi-modal Models are Good Class Incremental Learners
**Method:** Generative multimodal models synthesize training signals for previous classes
**Architecture:** Generative model
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Generative approach, khác paradigm.

---

## Paper 87 | ECLIPSE | CVPR 2024 | [VL]
**Title:** Efficient CL in Panoptic Segmentation with Visual Prompt Tuning
**Method:** Task-specific visual prompts + prompt interaction
**Architecture:** Prompt-based
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Prompt-based for segmentation.

---

# ===================== 2024 — ICLR (5 papers) =====================

## Paper 88 | SLM | ICLR 2024 | [LLM-CL]
**Title:** Scalable Language Model with Generalized Continual Learning
**Method:** JARe (Joint Adaptive Re-Parameterization) + DTKR (Dynamic Task-related Knowledge Retrieval)
**Architecture:** Multi-module (JARe + DTKR)
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Multi-module adaptive.

---

## Paper 89 | Reading Comprehension | ICLR 2024 | [LLM-CL]
**Title:** Adapting LLMs to Domains via Reading Comprehension
**Method:** Transforms raw corpora into reading comprehension format for CPT
**Architecture:** Single model + data transformation
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Data transformation strategy, không model approach.

---

## Paper 90 | Dissecting Forgetting | ICLR 2024 | [Analysis]
**Title:** Dissecting Learning and Forgetting in Language Model Finetuning
**Method:** Analysis: topic/style priors vs factual knowledge dalam forgetting
**Key insight:** Topic/style = simple features → learned fast, independently. Factual knowledge = learned slowly, needs capacity
### ⚠️ THAM KHẢO (Score: 4/10)
**Lý do:** Analysis paper. Insight: phân loại forgetting thành topic/style (nhanh) vs factual (chậm). Simple_idea có thể focus vào factual knowledge retention qua geometric modeling.

---

## Paper 91 | TiC-CLIP | ICLR 2024 | [VL]
**Title:** Continual Training of CLIP Models
**Method:** Rehearsal-based training from last checkpoint + old data replay
**Architecture:** CLIP + rehearsal
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Simple rehearsal approach.

---

## Paper 92 | CPPO | ICLR 2024 | [LLM-CL]
**Title:** Continual Learning for RL with Human Feedback
**Method:** Weighting strategy for policy learning vs experience solidification
**Architecture:** PPO-based RL
### ❌ KHÔNG PHÙ HỢP (Score: 1/10)
**Lý do:** RL approach, hoàn toàn khác paradigm.

---

# ===================== 2024 — AAAI (2 papers) =====================

## Paper 93 | Task-Aware Lang-Img | AAAI 2024 | [VL]
**Title:** Learning Task-Aware Language-Image Representation for CIL Object Detection
**Method:** Task-aware prompts + language supervision for CIOD
**Architecture:** VLM + prompts
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Prompt-based for object detection.

---

## Paper 94 | Fairness KD | AAAI 2025 | [VL]
**Title:** Maintaining Fairness in Logit-based KD for CIL
**Method:** Balanced distillation objective + logit adjustment for old/new class fairness
**Architecture:** KD-based CIL
**Anti-forgetting:** Logit adjustment accounts for class distribution shift
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** KD fairness adjustment, không model feature geometry.

---

# ===================== 2023 (15 papers) =====================

## Paper 95 | Soft-masking Mixed Tasks | EMNLP 2023 | [NLP-Task]
**Title:** Sub-network Discovery and Soft-masking for CL of Mixed Tasks
**Method:** Task-relevant sub-network identification + soft masks
**Architecture:** Single model + soft masks
**Anti-forgetting:** Soft masks protect critical parameters
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Parameter masking approach, không model feature geometry.

---

## Paper 96 | FeCAM | NeurIPS 2023 | [LLM-CL]
**Title:** Exploiting Heterogeneity of Class Distributions in Exemplar-Free CL
**Method:** Class-specific covariance matrices → Mahalanobis distance classification. Exploits heterogeneous class distributions in feature space.
**Architecture:** Single model + class-specific covariance storage
**Anti-forgetting:** Class distributions captured by covariance → no need for exemplars
### ✅ PHÙ HỢP (Score: 9/10)
**Lý do:** **PAPER RẤT PHÙ HỢP!** FeCAM model mỗi class bằng covariance matrix riêng → Mahalanobis distance. Trực tiếp liên quan simple_idea:
- **Feature distribution modeling** ✅: class-specific covariance = statistical tool
- **Single model** ✅: chỉ lưu covariance, không add module
- **Exemplar-free** ✅
- **CHƯA geometric** ✅: dùng Gaussian/Mahalanobis (flat Euclidean), CHƯA model hình học (vMF/Bingham cho hypersphere/manifold)
- **CHƯA external knowledge** ✅: chưa có external calibration
→ Simple_idea có thể nâng cấp: thay Gaussian covariance bằng geometric distribution (vMF/Bingham) phù hợp với geometry thực sự của feature space (hypersphere) + thêm external knowledge calibration.

---

## Paper 97 | SPG | ICML 2023 | [LLM-CL]
**Title:** Parameter-Level Soft-Masking for Continual Learning
**Method:** Learns importance scores per parameter + soft gradient modulation
**Architecture:** Single model + soft masks
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Parameter masking, không model feature geometry.

---

## Paper 98 | Mod-X | ICML 2023 | [VL]
**Title:** Continual VL Representation Learning with Off-Diagonal Information
**Method:** Aligns off-diagonal entries of contrastive matrices between old/new models
**Architecture:** VL model + off-diagonal alignment
**Anti-forgetting:** Preserves relative similarities giữa unmatched pairs
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** Off-diagonal information = structural relationship in representation space. Liên quan concept preservation of feature space structure. Nhưng contrastive alignment approach, không geometric distribution modeling.

---

## Paper 99 | CTP | ICCV 2023 | [VL]
**Title:** VL Continual Pretraining via Compatible Momentum Contrast and Topology Preservation
**Method:** Compatible momentum contrast + topology preservation in VL embedding space
**Architecture:** VL model + momentum contrast
**Anti-forgetting:** Topology preservation retains structural relationships
### ⚠️ THAM KHẢO (Score: 5/10)
**Lý do:** **Topology preservation** = liên quan hình học! Bảo toàn structural relationships trong joint VL space. Nhưng momentum contrast approach, không phải statistical distribution modeling.

---

## Paper 100 | LGCL | ICCV 2023 | [VL]
**Title:** Introducing Language Guidance in Prompt-based CL
**Method:** Language descriptions for task-level + class-level prompt guidance
**Architecture:** Prompt-based
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Prompt-based.

---

## Paper 101 | ZSCL | ICCV 2023 | [VL]
**Title:** Preventing Zero-Shot Transfer Degradation in CL of VLMs
**Method:** Reference dataset distillation preserving zero-shot capability
**Architecture:** CLIP + distillation
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Distillation approach.

---

## Paper 102 | MRN | ICCV 2023 | [NLP-Task]
**Title:** Multiplexed Routing Network for Incremental Multilingual Text Recognition
**Method:** Multiplexed routing → language-specific processing paths
**Architecture:** Multi-module (routing network)
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Routing network = multi-module.

---

## Paper 103 | CIGN | ICCV 2023 | [MM]
**Title:** Class-Incremental Grouping Network for Continual Audio-Visual Learning
**Method:** Grouping mechanism for audio-visual features + cross-modal interaction
**Architecture:** Multi-module (grouping + cross-modal)
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Multi-module grouping.

---

## Paper 104 | Continual DAP | ICLR 2023 | [LLM-CL]
**Title:** Continual Pre-training of Language Models
**Method:** Soft-masking controlling LM updates + proxy preserving general knowledge + contrastive representation
**Architecture:** Single model + soft masks + proxy
**Anti-forgetting:** Soft masks + proxy + contrastive learning
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Soft masking + proxy, không model feature geometry.

---

## Paper 105 | Progressive Prompts | ICLR 2023 | [LLM-CL]
**Title:** Continual Learning for Language Models without Forgetting
**Method:** Sequential soft prompt concatenation (new prompt per task)
**Architecture:** Frozen model + prompt sequence
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)
**Lý do:** Prompt-based.

---

## Paper 106 | VAG | ACL 2023 | [NLP-Task]
**Title:** Class-Incremental Learning based on Label Generation
**Method:** CIL as continual label generation task (generative LM)
**Architecture:** Generative model
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Generative approach.

---

## Paper 107 | Slow & Fast | ACL 2023 | [NLP-Task]
**Title:** Analyzing and Reducing Performance Gap in Cross-Lingual Transfer
**Method:** Differential learning rates (slow for cross-lingual params, fast for task-specific)
**Architecture:** Single model + differential LR
**Anti-forgetting:** Slow updates preserve cross-lingual alignment
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Differential LR strategy, không model geometry.

---

## Paper 108 | Exploring Data Geometry | CVPR 2023 | [LLM-CL]
**Title:** Exploring Data Geometry for Continual Learning
**Method:** Mixed curvature space (hyperbolic + Euclidean + spherical) + task-specific curvature parameters
**Architecture:** Single model operating in mixed-curvature space
**Anti-forgetting:** Geometrically appropriate space → more stable representations → reduces forgetting
### ✅ PHÙ HỢP (Score: 9/10)
**Lý do:** **PAPER RẤT PHÙ HỢP!** Trực tiếp explore data geometry cho CL!
- **Geometric feature space** ✅: Mixed curvature (hyperbolic + Euclidean + spherical)
- **Single model** ✅: single model với task-specific curvature
- **Anti-forgetting via geometry** ✅: appropriate geometry → stable representations
- **CHƯA dùng statistical distribution** ✅: dùng curvature space nhưng CHƯA model distribution (vMF/Bingham) trên manifold
- **CHƯA external knowledge** ✅: chưa có external calibration
→ Simple_idea có thể đóng góp: trên nền geometric space, thêm distribution modeling (vMF cho spherical part, wrapped Gaussian cho hyperbolic part) + external knowledge calibration. Paper này là closest prior work.

---

## Paper 109 | CODA-Prompt | CVPR 2023 | [VL]
**Title:** COntinual Decomposed Attention-based Prompting
**Method:** Decomposed attention prompt pool + input-conditioned composition
**Architecture:** Prompt-based
### ❌ KHÔNG PHÙ HỢP (Score: 3/10)
**Lý do:** Prompt-based.

---

# ===================== TỔNG HỢP =====================

## TOP PAPERS cho Simple Idea (Score ≥ 7):

| Rank | Paper | Score | Year | Venue | Lý do chính |
|------|-------|-------|------|-------|-------------|
| 🥇 | **96. FeCAM** | 9/10 | 2023 | NeurIPS | Class-specific covariance + Mahalanobis, CHƯA geometric distribution |
| 🥇 | **108. Data Geometry** | 9/10 | 2023 | CVPR | Mixed curvature space, CHƯA statistical distribution + external knowledge |
| 🥉 | **47. Proxy-FDA** | 8/10 | 2025 | ICML | Feature distribution alignment via proxies, CHƯA geometric + external |
| 🥉 | **77. LDC** | 8/10 | 2024 | ECCV | Learnable drift compensation, CHƯA geometric distribution |
| 5 | **48. Angle Matters** | 7/10 | 2025 | ICML | Angles in feature space determine forgetting (theoretical) |
| 5 | **65. CLAP4CLIP** | 7/10 | 2024 | NeurIPS | Probabilistic feature modeling + distribution regularization |
| 5 | **20. Dual Drift** | 7/10 | 2025 | ICCV | Prototype drift (inner-task + inter-task) |

## PAPERS THAM KHẢO QUAN TRỌNG (Score 5-6):

| Paper | Score | Lý do tham khảo |
|-------|-------|-----------------|
| **30. DGAR** | 6 | Diffusion-generated historical distributions |
| **61. LoRA-** | 6 | Drift-Resistant Space concept |
| **73. PromptCCD** | 6 | GMM distribution modeling for prompts |
| **06. CMCL** | 5 | Gradient projection + dual stability/plasticity bounds |
| **13. MG-CLIP** | 5 | Modality gap ↔ forgetting (geometric concept) |
| **35. Feature Dist** | 5 | "Presentative feature distribution" concept |
| **52. Spurious Forg.** | 5 | Spurious vs true forgetting distinction |
| **53. Function Vec.** | 5 | Function vector representation + CF from activation bias |
| **64. Global Align.** | 5 | Global shared feature space composition |
| **84. Lang-Guided** | 5 | Language as "external knowledge" for feature anchoring |
| **98. Mod-X** | 5 | Off-diagonal structural relationships |
| **99. CTP** | 5 | Topology preservation in VL space |

## PHÂN BỐ ĐÁNH GIÁ:
- ✅ PHÙ HỢP (≥7): **7 papers** (20, 47, 48, 65, 77, 96, 108)
- ⚠️ THAM KHẢO (4-6): **20 papers**
- ❌ KHÔNG PHÙ HỢP (≤3): **82 papers**

## HƯỚNG ÁP DỤNG SIMPLE IDEA:

### Target Paper ưu tiên cao nhất:
1. **FeCAM (96)** — Paper đã dùng class-specific covariance (Gaussian/Mahalanobis) nhưng CHƯA geometric. → Simple_idea replace Gaussian bằng vMF/Bingham phù hợp hypersphere + thêm external knowledge calibration.
2. **Data Geometry (108)** — Paper đã explore mixed curvature space nhưng CHƯA distribution modeling. → Simple_idea thêm distribution trên từng curvature component + external calibration.
3. **Proxy-FDA (47)** — Paper align feature distribution nhưng dùng proxy/NN graph. → Simple_idea: geometric statistical tool + external knowledge thay vì proxy.
4. **LDC (77)** — Paper compensate drift bằng learnable module. → Simple_idea: thay module bằng geometric distribution tracking + external calibration (single model, no extra module).

### Theoretical Support:
- **Angle Matters (48)** — Chứng minh angle trong feature space quyết định forgetting → justify geometric approach
- **CLAP4CLIP (65)** — Probabilistic modeling works → nhưng Gaussian, chưa geometric
- **Dual Drift (20)** — Prototype drift formalization → directly supports center drift concept


## ENHANCED EVALUATION DATA

### Domain Classification
| Paper ID | Domain | Title |
|----------|--------|-------|
| 01 | NLP        | Gated Integration of Low-Rank Adaptation for CL of Large Lan... |
| 02 | ML/Multi   | MINGLE: Mixture of Null-Space Gated Low-Rank Experts for Tes... |
| 03 | ML/Multi   | MemEIC: A Step Toward Continual and Compositional Knowledge ... |
| 04 | ML/Multi   | Investigating and Mitigating CF in Medical Knowledge Injecti... |
| 05 | ML/Multi   | Bisecle: Binding and Separation in CL for Video Language Und... |
| 06 | ML/Multi   | Continual Multimodal Contrastive Learning... |
| 07 | NLP        | Demystifying Language Model Forgetting with Low-rank Example... |
| 08 | ML/Multi   | Turning the Tables: Enabling Backward Transfer via Causal-Aw... |
| 09 | ML/Multi   | Mitigating Intra- and Inter-modal Forgetting in CL of Unifie... |
| 10 | NLP        | MEMOIR: Lifelong Model Editing with Minimal Overwrite and In... |
| 11 | NLP        | Self-Evolving Pseudo-Rehearsal for CF with Task Similarity i... |
| 12 | ML/Multi   | Reliable Lifelong Multimodal Editing: Conflict-Aware Retriev... |
| 13 | ML/Multi   | Mind the Gap: Preserving and Compensating for Modality Gap i... |
| 14 | ML/Multi   | SMoLoRA: Exploring and Defying Dual CF in Continual Visual I... |
| 15 | ML/Multi   | DMNSP: Dynamic Multi-Layer Null Space Projection for Vision-... |
| 16 | ML/Multi   | Ask and Remember: Questions-Only Replay Strategy for Continu... |
| 17 | NLP        | TWIST&SCOUT: Grounding Multimodal LLM-Experts by Forget-Free... |
| 18 | ML/Multi   | Instruction-Grounded Visual Projectors for CL of Generative ... |
| 19 | ML/Multi   | External Knowledge Injection for CLIP-Based Class-Incrementa... |
| 20 | ML/Multi   | Overcoming Dual Drift for Continual Long-Tailed Visual Quest... |
| 21 | ML/Multi   | PLAN: Proactive Low-Rank Allocation for Continual Learning... |
| 22 | NLP        | Knowledge Decoupling via Orthogonal Projection for Lifelong ... |
| 23 | ML/Multi   | Serial Lifelong Editing via Mixture of Knowledge Expert... |
| 24 | ML/Multi   | Efficient Domain Continual Pretraining by Mitigating the Sta... |
| 25 | NLP        | Neuron-Level Sequential Editing for Large Language Models... |
| 26 | NLP        | CLoRA: Controlled Low-Rank Adaptation with Subspace Regulari... |
| 27 | NLP        | HiDe-LLaVA: Hierarchical Decoupling for Continual IT of Mult... |
| 28 | NLP        | Multi-Modality Expansion and Retention for LLMs through Para... |
| 29 | NLP        | GORP: Continual Gradient Low-Rank Projection Fine-Tuning for... |
| 30 | ML/Multi   | DGAR: Generative Adaptive Replay CL Model for Temporal KG Re... |
| 31 | ML/Multi   | Learn to Memorize: Scalable CL in Semiparametric Models with... |
| 32 | NLP        | Don't Half-listen: Capturing Key-part Info in Continual Inst... |
| 33 | NLP        | Recurrent Knowledge Identification and Fusion for Language M... |
| 34 | NLP        | TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretrai... |
| 35 | NLP        | Exploiting Presentative Feature Distributions for PE-CL of L... |
| 36 | NLP        | Reinforced Lifelong Editing for Language Models... |
| 37 | NLP        | Understanding the Limits of Lifelong Knowledge Editing in LL... |
| 38 | ML/Multi   | Knowledge Swapping via Learning and Unlearning... |
| 39 | NLP        | Learning Dynamics in Continual Pre-Training for Large Langua... |
| 40 | NLP        | Large Continual Instruction Assistant... |
| 41 | ML/Multi   | TreeLoRA: Efficient CL via Layer-Wise LoRAs Guided by Hierar... |
| 42 | NLP        | Adaptive Localization of Knowledge Negation for Continual LL... |
| 43 | NLP        | From RAG to Memory: Non-Parametric CL for Large Language Mod... |
| 44 | ML/Multi   | SEFE: Superficial and Essential Forgetting Eliminator for Mu... |
| 45 | ML/Multi   | LADA: Scalable Label-Specific CLIP Adapter for Continual Lea... |
| 46 | NLP        | Componential Prompt-Knowledge Alignment for Domain Increment... |
| 47 | ML/Multi   | Proxy-FDA: Proxy-based Feature Distribution Alignment for Fi... |
| 48 | ML/Multi   | Understanding the Forgetting of Replay-based CL via Feature ... |
| 49 | ML/Multi   | LOIRE: LifelOng learning on Incremental data via pre-trained... |
| 50 | NLP        | On Large Language Model Continual Unlearning... |
| 51 | ML/Multi   | SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class In... |
| 52 | NLP        | Spurious Forgetting in Continual Learning of Language Models... |
| 53 | NLP        | Unlocking Function Vectors for Mitigating CF in Continual In... |
| 54 | ML/Multi   | C-CLIP: Multimodal Continual Learning for Vision-Language Mo... |
| 55 | NLP        | Adapt-inf: Scalable Continual Multimodal Instruction Tuning ... |
| 56 | ML/Multi   | Vision and Language Synergy for Rehearsal Free Continual Lea... |
| 57 | NLP        | Language Guided Concept Bottleneck Models for Interpretable ... |
| 58 | NLP        | AdaDARE-gamma: Balancing Stability and Plasticity in Multi-m... |
| 59 | ML/Multi   | Synthetic Data is Elegant GIFT for Continual Vision-Language... |
| 60 | ML/Multi   | CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free CI... |
| 61 | ML/Multi   | LoRA Subtraction for Drift-Resistant Space in Exemplar-Free ... |
| 62 | ML/Multi   | Stabilizing Zero-Shot Prediction: Antidote to Forgetting in ... |
| 63 | ML/Multi   | Advancing Cross-domain Discriminability in CL of Vision-Lang... |
| 64 | ML/Multi   | Continual Learning with Global Alignment... |
| 65 | ML/Multi   | CLAP4CLIP: Continual Learning with Probabilistic Finetuning ... |
| 66 | ML/Multi   | Train-Attention: Meta-Learning Where to Focus in Continual K... |
| 67 | ML/Multi   | ViLCo-Bench: VIdeo Language COntinual learning Benchmark... |
| 68 | ML/Multi   | Visual Prompt Tuning in Null Space for Continual Learning... |
| 69 | ML/Multi   | Class-Incremental Learning with CLIP: Adaptive Representatio... |
| 70 | ML/Multi   | Mind the Interference: Retaining Pre-trained Knowledge in PE... |
| 71 | ML/Multi   | Select and Distill: Selective Dual-Teacher KT for CL on VLMs... |
| 72 | ML/Multi   | PILoRA: Prototype Guided Incremental LoRA for Federated CIL... |
| 73 | NLP        | PromptCCD: Learning Gaussian Mixture Prompt Pool for Continu... |
| 74 | CV         | Anytime CL for Open Vocabulary Classification... |
| 75 | CV         | CLIFF: Continual Latent Diffusion for Open-Vocabulary Object... |
| 76 | ML/Multi   | CLEO: Continual Learning of Evolving Ontologies... |
| 77 | ML/Multi   | Exemplar-free Continual Representation Learning via Learnabl... |
| 78 | ML/Multi   | Adapt without Forgetting: Distill Proximity from Dual Teache... |
| 79 | NLP        | COPAL: Continual Pruning in Large Language Generative Models... |
| 80 | CV         | STELLA: Continual Audio-Video Pre-training with SpatioTempor... |
| 81 | ML/Multi   | InfLoRA: Interference-Free Low-Rank Adaptation for Continual... |
| 82 | ML/Multi   | Boosting CL of VLMs via Mixture-of-Experts Adapters... |
| 83 | NLP        | Pre-trained Vision and Language Transformers Are Few-Shot In... |
| 84 | CV         | Enhancing Visual CL with Language-Guided Supervision... |
| 85 | NLP        | Text-Enhanced Data-free Approach for Federated Class-Increme... |
| 86 | ML/Multi   | Generative Multi-modal Models are Good Class Incremental Lea... |
| 87 | CV         | ECLIPSE: Efficient CL in Panoptic Segmentation with Visual P... |
| 88 | NLP        | Scalable Language Model with Generalized Continual Learning... |
| 89 | NLP        | Adapting Large Language Models via Reading Comprehension... |
| 90 | NLP        | Dissecting Learning and Forgetting in Language Model Finetun... |
| 91 | ML/Multi   | TiC-CLIP: Continual Training of CLIP Models... |
| 92 | ML/Multi   | CPPO: Continual Learning for Reinforcement Learning with Hum... |
| 93 | CV         | Learning Task-Aware Language-Image Representation for CIL Ob... |
| 94 | ML/Multi   | Maintaining Fairness in Logit-based KD for Class-Incremental... |
| 95 | ML/Multi   | Sub-network Discovery and Soft-masking for CL of Mixed Tasks... |
| 96 | ML/Multi   | FeCAM: Exploiting Heterogeneity of Class Distributions in Ex... |
| 97 | ML/Multi   | Parameter-Level Soft-Masking for Continual Learning... |
| 98 | ML/Multi   | Continual Vision-Language Representation Learning with Off-D... |
| 99 | ML/Multi   | CTP: Towards Vision-Language Continual Pretraining via Compa... |
| 100 | NLP        | Introducing Language Guidance in Prompt-based Continual Lear... |
| 101 | ML/Multi   | Preventing Zero-Shot Transfer Degradation in CL of Vision-La... |
| 102 | NLP        | MRN: Multiplexed Routing Network for Incremental Multilingua... |
| 103 | CV         | Class-Incremental Grouping Network for Continual Audio-Visua... |
| 104 | NLP        | Continual Learning of Language Models... |
| 105 | NLP        | Progressive Prompts: CL for Language Models without Forgetti... |
| 106 | NLP        | Class-Incremental Learning based on Label Generation... |
| 107 | ML/Multi   | Analyzing and Reducing Performance Gap in Cross-Lingual Tran... |
| 108 | ML/Multi   | Exploring Data Geometry for Continual Learning... |
| 109 | ML/Multi   | CODA-Prompt: COntinual Decomposed Attention-based Prompting ... |

### Code Availability Checklist
| Paper ID | Public Code | Link |
|----------|-------------|------|
| 01 | ❌ | N/A |
| 02 | ❌ | N/A |
| 03 | ❌ | N/A |
| 04 | ❌ | N/A |
| 05 | ❌ | N/A |
| 06 | ❌ | N/A |
| 07 | ❌ | N/A |
| 08 | ❌ | N/A |
| 09 | ❌ | N/A |
| 10 | ❌ | N/A |
| 11 | ❌ | N/A |
| 12 | ❌ | N/A |
| 13 | ✅ | https://github.com/linlany/MindtheGap |
| 14 | ❌ | N/A |
| 15 | ❌ | N/A |
| 16 | ✅ | https://github.com/IemProg/QUAD |
| 17 | ❌ | N/A |
| 18 | ❌ | N/A |
| 19 | ❌ | N/A |
| 20 | ❌ | N/A |
| 21 | ❌ | N/A |
| 22 | ❌ | N/A |
| 23 | ❌ | N/A |
| 24 | ❌ | N/A |
| 25 | ❌ | N/A |
| 26 | ❌ | N/A |
| 27 | ❌ | N/A |
| 28 | ❌ | N/A |
| 29 | ❌ | N/A |
| 30 | ❌ | N/A |
| 31 | ❌ | N/A |
| 32 | ❌ | N/A |
| 33 | ❌ | N/A |
| 34 | ❌ | N/A |
| 35 | ❌ | N/A |
| 36 | ❌ | N/A |
| 37 | ❌ | N/A |
| 38 | ❌ | N/A |
| 39 | ❌ | N/A |
| 40 | ❌ | N/A |
| 41 | ❌ | N/A |
| 42 | ❌ | N/A |
| 43 | ❌ | N/A |
| 44 | ❌ | N/A |
| 45 | ❌ | N/A |
| 46 | ❌ | N/A |
| 47 | ❌ | N/A |
| 48 | ❌ | N/A |
| 49 | ❌ | N/A |
| 50 | ❌ | N/A |
| 51 | ❌ | N/A |
| 52 | ❌ | N/A |
| 53 | ❌ | N/A |
| 54 | ❌ | N/A |
| 55 | ❌ | N/A |
| 56 | ❌ | N/A |
| 57 | ❌ | N/A |
| 58 | ❌ | N/A |
| 59 | ✅ | https://huggingface.co/docs/hub |
| 60 | ✅ | https://huggingface.co/docs/hub |
| 61 | ❌ | N/A |
| 62 | ❌ | N/A |
| 63 | ❌ | N/A |
| 64 | ❌ | N/A |
| 65 | ❌ | N/A |
| 66 | ❌ | N/A |
| 67 | ❌ | N/A |
| 68 | ✅ | https://github.com/zugexiaodui/VPTinNSforCL |
| 69 | ❌ | N/A |
| 70 | ✅ | https://github.com/lloongx/DIKI |
| 71 | ✅ | https://huggingface.co/docs/hub |
| 72 | ✅ | https://github.com/Ghy0501/PILoRA |
| 73 | ✅ | https://huggingface.co/docs/hub |
| 74 | ✅ | https://github.com/jessemelpolio/AnytimeCL |
| 75 | ❌ | N/A |
| 76 | ✅ | https://huggingface.co/docs/hub |
| 77 | ✅ | https://github.com/alviur/ldc |
| 78 | ❌ | N/A |
| 79 | ❌ | N/A |
| 80 | ❌ | N/A |
| 81 | ✅ | https://huggingface.co/docs/hub |
| 82 | ✅ | https://github.com/JiazuoYu/MoE-Adapters4CL |
| 83 | ✅ | https://github.com/KHU-AGI/PriViLege |
| 84 | ✅ | https://huggingface.co/docs/hub |
| 85 | ✅ | https://github.com/tmtuan1307/lander |
| 86 | ✅ | https://github.com/DoubleClass/GMM |
| 87 | ✅ | https://github.com/clovaai/ECLIPSE |
| 88 | ❌ | N/A |
| 89 | ❌ | N/A |
| 90 | ❌ | N/A |
| 91 | ❌ | N/A |
| 92 | ❌ | N/A |
| 93 | ❌ | N/A |
| 94 | ❌ | N/A |
| 95 | ✅ | https://github.com/ZixuanKe/PyContinual |
| 96 | ✅ | https://github.com/dipamgoswami/FeCAM |
| 97 | ❌ | N/A |
| 98 | ✅ | https://huggingface.co/docs/hub |
| 99 | ❌ | N/A |
| 100 | ✅ | https://huggingface.co/docs/hub |
| 101 | ❌ | N/A |
| 102 | ❌ | N/A |
| 103 | ✅ | https://github.com/stoneMo/CIGN |
| 104 | ❌ | N/A |
| 105 | ❌ | N/A |
| 106 | ✅ | https://huggingface.co/docs/hub |
| 107 | ✅ | https://huggingface.co/docs/hub |
| 108 | ❌ | N/A |
| 109 | ❌ | N/A |


## ENHANCED EVALUATION DATA

### Domain Classification
| Paper ID | Domain | Title |
|----------|--------|-------|
| 01 | NLP        | Gated Integration of Low-Rank Adaptation for CL of Large Lan... |
| 02 | ML/Multi   | MINGLE: Mixture of Null-Space Gated Low-Rank Experts for Tes... |
| 03 | ML/Multi   | MemEIC: A Step Toward Continual and Compositional Knowledge ... |
| 04 | ML/Multi   | Investigating and Mitigating CF in Medical Knowledge Injecti... |
| 05 | ML/Multi   | Bisecle: Binding and Separation in CL for Video Language Und... |
| 06 | ML/Multi   | Continual Multimodal Contrastive Learning... |
| 07 | NLP        | Demystifying Language Model Forgetting with Low-rank Example... |
| 08 | ML/Multi   | Turning the Tables: Enabling Backward Transfer via Causal-Aw... |
| 09 | ML/Multi   | Mitigating Intra- and Inter-modal Forgetting in CL of Unifie... |
| 10 | NLP        | MEMOIR: Lifelong Model Editing with Minimal Overwrite and In... |
| 11 | NLP        | Self-Evolving Pseudo-Rehearsal for CF with Task Similarity i... |
| 12 | ML/Multi   | Reliable Lifelong Multimodal Editing: Conflict-Aware Retriev... |
| 13 | ML/Multi   | Mind the Gap: Preserving and Compensating for Modality Gap i... |
| 14 | ML/Multi   | SMoLoRA: Exploring and Defying Dual CF in Continual Visual I... |
| 15 | ML/Multi   | DMNSP: Dynamic Multi-Layer Null Space Projection for Vision-... |
| 16 | ML/Multi   | Ask and Remember: Questions-Only Replay Strategy for Continu... |
| 17 | NLP        | TWIST&SCOUT: Grounding Multimodal LLM-Experts by Forget-Free... |
| 18 | ML/Multi   | Instruction-Grounded Visual Projectors for CL of Generative ... |
| 19 | ML/Multi   | External Knowledge Injection for CLIP-Based Class-Incrementa... |
| 20 | ML/Multi   | Overcoming Dual Drift for Continual Long-Tailed Visual Quest... |
| 21 | ML/Multi   | PLAN: Proactive Low-Rank Allocation for Continual Learning... |
| 22 | NLP        | Knowledge Decoupling via Orthogonal Projection for Lifelong ... |
| 23 | ML/Multi   | Serial Lifelong Editing via Mixture of Knowledge Expert... |
| 24 | ML/Multi   | Efficient Domain Continual Pretraining by Mitigating the Sta... |
| 25 | NLP        | Neuron-Level Sequential Editing for Large Language Models... |
| 26 | NLP        | CLoRA: Controlled Low-Rank Adaptation with Subspace Regulari... |
| 27 | NLP        | HiDe-LLaVA: Hierarchical Decoupling for Continual IT of Mult... |
| 28 | NLP        | Multi-Modality Expansion and Retention for LLMs through Para... |
| 29 | NLP        | GORP: Continual Gradient Low-Rank Projection Fine-Tuning for... |
| 30 | ML/Multi   | DGAR: Generative Adaptive Replay CL Model for Temporal KG Re... |
| 31 | ML/Multi   | Learn to Memorize: Scalable CL in Semiparametric Models with... |
| 32 | NLP        | Don't Half-listen: Capturing Key-part Info in Continual Inst... |
| 33 | NLP        | Recurrent Knowledge Identification and Fusion for Language M... |
| 34 | NLP        | TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretrai... |
| 35 | NLP        | Exploiting Presentative Feature Distributions for PE-CL of L... |
| 36 | NLP        | Reinforced Lifelong Editing for Language Models... |
| 37 | NLP        | Understanding the Limits of Lifelong Knowledge Editing in LL... |
| 38 | ML/Multi   | Knowledge Swapping via Learning and Unlearning... |
| 39 | NLP        | Learning Dynamics in Continual Pre-Training for Large Langua... |
| 40 | NLP        | Large Continual Instruction Assistant... |
| 41 | ML/Multi   | TreeLoRA: Efficient CL via Layer-Wise LoRAs Guided by Hierar... |
| 42 | NLP        | Adaptive Localization of Knowledge Negation for Continual LL... |
| 43 | NLP        | From RAG to Memory: Non-Parametric CL for Large Language Mod... |
| 44 | ML/Multi   | SEFE: Superficial and Essential Forgetting Eliminator for Mu... |
| 45 | ML/Multi   | LADA: Scalable Label-Specific CLIP Adapter for Continual Lea... |
| 46 | NLP        | Componential Prompt-Knowledge Alignment for Domain Increment... |
| 47 | ML/Multi   | Proxy-FDA: Proxy-based Feature Distribution Alignment for Fi... |
| 48 | ML/Multi   | Understanding the Forgetting of Replay-based CL via Feature ... |
| 49 | ML/Multi   | LOIRE: LifelOng learning on Incremental data via pre-trained... |
| 50 | NLP        | On Large Language Model Continual Unlearning... |
| 51 | ML/Multi   | SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class In... |
| 52 | NLP        | Spurious Forgetting in Continual Learning of Language Models... |
| 53 | NLP        | Unlocking Function Vectors for Mitigating CF in Continual In... |
| 54 | ML/Multi   | C-CLIP: Multimodal Continual Learning for Vision-Language Mo... |
| 55 | NLP        | Adapt-inf: Scalable Continual Multimodal Instruction Tuning ... |
| 56 | ML/Multi   | Vision and Language Synergy for Rehearsal Free Continual Lea... |
| 57 | NLP        | Language Guided Concept Bottleneck Models for Interpretable ... |
| 58 | NLP        | AdaDARE-gamma: Balancing Stability and Plasticity in Multi-m... |
| 59 | ML/Multi   | Synthetic Data is Elegant GIFT for Continual Vision-Language... |
| 60 | ML/Multi   | CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free CI... |
| 61 | ML/Multi   | LoRA Subtraction for Drift-Resistant Space in Exemplar-Free ... |
| 62 | ML/Multi   | Stabilizing Zero-Shot Prediction: Antidote to Forgetting in ... |
| 63 | ML/Multi   | Advancing Cross-domain Discriminability in CL of Vision-Lang... |
| 64 | ML/Multi   | Continual Learning with Global Alignment... |
| 65 | ML/Multi   | CLAP4CLIP: Continual Learning with Probabilistic Finetuning ... |
| 66 | ML/Multi   | Train-Attention: Meta-Learning Where to Focus in Continual K... |
| 67 | ML/Multi   | ViLCo-Bench: VIdeo Language COntinual learning Benchmark... |
| 68 | ML/Multi   | Visual Prompt Tuning in Null Space for Continual Learning... |
| 69 | ML/Multi   | Class-Incremental Learning with CLIP: Adaptive Representatio... |
| 70 | ML/Multi   | Mind the Interference: Retaining Pre-trained Knowledge in PE... |
| 71 | ML/Multi   | Select and Distill: Selective Dual-Teacher KT for CL on VLMs... |
| 72 | ML/Multi   | PILoRA: Prototype Guided Incremental LoRA for Federated CIL... |
| 73 | NLP        | PromptCCD: Learning Gaussian Mixture Prompt Pool for Continu... |
| 74 | CV         | Anytime CL for Open Vocabulary Classification... |
| 75 | CV         | CLIFF: Continual Latent Diffusion for Open-Vocabulary Object... |
| 76 | ML/Multi   | CLEO: Continual Learning of Evolving Ontologies... |
| 77 | ML/Multi   | Exemplar-free Continual Representation Learning via Learnabl... |
| 78 | ML/Multi   | Adapt without Forgetting: Distill Proximity from Dual Teache... |
| 79 | NLP        | COPAL: Continual Pruning in Large Language Generative Models... |
| 80 | CV         | STELLA: Continual Audio-Video Pre-training with SpatioTempor... |
| 81 | ML/Multi   | InfLoRA: Interference-Free Low-Rank Adaptation for Continual... |
| 82 | ML/Multi   | Boosting CL of VLMs via Mixture-of-Experts Adapters... |
| 83 | NLP        | Pre-trained Vision and Language Transformers Are Few-Shot In... |
| 84 | CV         | Enhancing Visual CL with Language-Guided Supervision... |
| 85 | NLP        | Text-Enhanced Data-free Approach for Federated Class-Increme... |
| 86 | ML/Multi   | Generative Multi-modal Models are Good Class Incremental Lea... |
| 87 | CV         | ECLIPSE: Efficient CL in Panoptic Segmentation with Visual P... |
| 88 | NLP        | Scalable Language Model with Generalized Continual Learning... |
| 89 | NLP        | Adapting Large Language Models via Reading Comprehension... |
| 90 | NLP        | Dissecting Learning and Forgetting in Language Model Finetun... |
| 91 | ML/Multi   | TiC-CLIP: Continual Training of CLIP Models... |
| 92 | ML/Multi   | CPPO: Continual Learning for Reinforcement Learning with Hum... |
| 93 | CV         | Learning Task-Aware Language-Image Representation for CIL Ob... |
| 94 | ML/Multi   | Maintaining Fairness in Logit-based KD for Class-Incremental... |
| 95 | ML/Multi   | Sub-network Discovery and Soft-masking for CL of Mixed Tasks... |
| 96 | ML/Multi   | FeCAM: Exploiting Heterogeneity of Class Distributions in Ex... |
| 97 | ML/Multi   | Parameter-Level Soft-Masking for Continual Learning... |
| 98 | ML/Multi   | Continual Vision-Language Representation Learning with Off-D... |
| 99 | ML/Multi   | CTP: Towards Vision-Language Continual Pretraining via Compa... |
| 100 | NLP        | Introducing Language Guidance in Prompt-based Continual Lear... |
| 101 | ML/Multi   | Preventing Zero-Shot Transfer Degradation in CL of Vision-La... |
| 102 | NLP        | MRN: Multiplexed Routing Network for Incremental Multilingua... |
| 103 | CV         | Class-Incremental Grouping Network for Continual Audio-Visua... |
| 104 | NLP        | Continual Learning of Language Models... |
| 105 | NLP        | Progressive Prompts: CL for Language Models without Forgetti... |
| 106 | NLP        | Class-Incremental Learning based on Label Generation... |
| 107 | ML/Multi   | Analyzing and Reducing Performance Gap in Cross-Lingual Tran... |
| 108 | ML/Multi   | Exploring Data Geometry for Continual Learning... |
| 109 | ML/Multi   | CODA-Prompt: COntinual Decomposed Attention-based Prompting ... |

### Code Availability Checklist
| Paper ID | Public Code | Link |
|----------|-------------|------|
| 01 | ❌ | N/A |
| 02 | ❌ | N/A |
| 03 | ❌ | N/A |
| 04 | ❌ | N/A |
| 05 | ❌ | N/A |
| 06 | ❌ | N/A |
| 07 | ❌ | N/A |
| 08 | ❌ | N/A |
| 09 | ❌ | N/A |
| 10 | ❌ | N/A |
| 11 | ❌ | N/A |
| 12 | ❌ | N/A |
| 13 | ✅ | https://github.com/linlany/MindtheGap |
| 14 | ❌ | N/A |
| 15 | ❌ | N/A |
| 16 | ✅ | https://github.com/IemProg/QUAD |
| 17 | ❌ | N/A |
| 18 | ❌ | N/A |
| 19 | ❌ | N/A |
| 20 | ❌ | N/A |
| 21 | ❌ | N/A |
| 22 | ❌ | N/A |
| 23 | ❌ | N/A |
| 24 | ❌ | N/A |
| 25 | ❌ | N/A |
| 26 | ❌ | N/A |
| 27 | ❌ | N/A |
| 28 | ❌ | N/A |
| 29 | ❌ | N/A |
| 30 | ❌ | N/A |
| 31 | ❌ | N/A |
| 32 | ❌ | N/A |
| 33 | ❌ | N/A |
| 34 | ❌ | N/A |
| 35 | ❌ | N/A |
| 36 | ❌ | N/A |
| 37 | ❌ | N/A |
| 38 | ❌ | N/A |
| 39 | ❌ | N/A |
| 40 | ❌ | N/A |
| 41 | ❌ | N/A |
| 42 | ❌ | N/A |
| 43 | ❌ | N/A |
| 44 | ❌ | N/A |
| 45 | ❌ | N/A |
| 46 | ❌ | N/A |
| 47 | ❌ | N/A |
| 48 | ❌ | N/A |
| 49 | ❌ | N/A |
| 50 | ❌ | N/A |
| 51 | ❌ | N/A |
| 52 | ❌ | N/A |
| 53 | ❌ | N/A |
| 54 | ❌ | N/A |
| 55 | ❌ | N/A |
| 56 | ❌ | N/A |
| 57 | ❌ | N/A |
| 58 | ❌ | N/A |
| 59 | ✅ | https://huggingface.co/docs/hub |
| 60 | ✅ | https://huggingface.co/docs/hub |
| 61 | ❌ | N/A |
| 62 | ❌ | N/A |
| 63 | ❌ | N/A |
| 64 | ❌ | N/A |
| 65 | ❌ | N/A |
| 66 | ❌ | N/A |
| 67 | ❌ | N/A |
| 68 | ✅ | https://github.com/zugexiaodui/VPTinNSforCL |
| 69 | ❌ | N/A |
| 70 | ✅ | https://github.com/lloongx/DIKI |
| 71 | ✅ | https://huggingface.co/docs/hub |
| 72 | ✅ | https://github.com/Ghy0501/PILoRA |
| 73 | ✅ | https://huggingface.co/docs/hub |
| 74 | ✅ | https://github.com/jessemelpolio/AnytimeCL |
| 75 | ❌ | N/A |
| 76 | ✅ | https://huggingface.co/docs/hub |
| 77 | ✅ | https://github.com/alviur/ldc |
| 78 | ❌ | N/A |
| 79 | ❌ | N/A |
| 80 | ❌ | N/A |
| 81 | ✅ | https://huggingface.co/docs/hub |
| 82 | ✅ | https://github.com/JiazuoYu/MoE-Adapters4CL |
| 83 | ✅ | https://github.com/KHU-AGI/PriViLege |
| 84 | ✅ | https://huggingface.co/docs/hub |
| 85 | ✅ | https://github.com/tmtuan1307/lander |
| 86 | ✅ | https://github.com/DoubleClass/GMM |
| 87 | ✅ | https://github.com/clovaai/ECLIPSE |
| 88 | ❌ | N/A |
| 89 | ❌ | N/A |
| 90 | ❌ | N/A |
| 91 | ❌ | N/A |
| 92 | ❌ | N/A |
| 93 | ❌ | N/A |
| 94 | ❌ | N/A |
| 95 | ✅ | https://github.com/ZixuanKe/PyContinual |
| 96 | ✅ | https://github.com/dipamgoswami/FeCAM |
| 97 | ❌ | N/A |
| 98 | ✅ | https://huggingface.co/docs/hub |
| 99 | ❌ | N/A |
| 100 | ✅ | https://huggingface.co/docs/hub |
| 101 | ❌ | N/A |
| 102 | ❌ | N/A |
| 103 | ✅ | https://github.com/stoneMo/CIGN |
| 104 | ❌ | N/A |
| 105 | ❌ | N/A |
| 106 | ✅ | https://huggingface.co/docs/hub |
| 107 | ✅ | https://huggingface.co/docs/hub |
| 108 | ❌ | N/A |
| 109 | ❌ | N/A |
