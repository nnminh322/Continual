# Comprehensive Novelty Search Report
## Proposed Idea: Statistical Knowledge Signatures + OT Routing + Backbone Anti-Drift for Continual Learning

**Date**: March 6, 2026  
**Search Scope**: arXiv (multi-query), specific paper fetches, workspace context analysis

---

## I. EXISTING WORK: Papers That Partially Overlap

### A. OT-Based Routing in MoE (Component 2 overlap)

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 1 | **BASE Layers: Simplifying Training of Large, Sparse Models** (arXiv:2103.16716) | 2021 | ICML | Formulates token-to-expert assignment as a **linear assignment problem** (a special case of OT). Guarantees balanced compute loads without auxiliary losses. |
| 2 | **Selective Sinkhorn Routing for Improved Sparse MoE** (arXiv:2511.08972) | 2025 | - | Formulates token-to-expert assignment as an **optimal transport problem** using Sinkhorn algorithm. Derives gating scores directly from transport map. **Most directly relevant to Component 2.** |
| 3 | **Sparsity-Constrained Optimal Transport** (arXiv:2209.15466) | 2023 | ICLR | Theoretical OT framework with sparsity constraints applicable to MoE routing. |
| 4 | **Continual Pre-training of MoEs: How robust is your router?** (arXiv:2503.05029) | 2025 | - | Studies Sinkhorn-balanced routing during continual pre-training. Shows surprising robustness of OT-based routing to distribution shift in CL settings. |

**Key Difference from Proposed Idea**: These works use OT for **load-balancing** (assigning tokens to experts evenly). The proposed idea uses OT to **match input distributions to expert knowledge signatures** — a fundamentally different formulation where the cost matrix is derived from statistical distribution distances (e.g., vMF-to-vMF), not learned linear projections.

### B. MoE + Routing for Continual Learning (Components 1+2 overlap)

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 5 | **Scaling CL with Bi-Level Routing MoE (CaRE)** (arXiv:2602.03473) | 2026 | - | Bi-level routing: first selects task-specific routers, then routes to experts. Scales to 300+ tasks. Uses learned routers, not distribution matching. |
| 6 | **PASs-MoE: Mitigating Misaligned Co-drift among Router and Experts** (arXiv:2601.13020) | 2026 | - | Identifies "misaligned co-drift" between router & experts in CL. Uses LoRA pathway activation subspaces for routing. Addresses router drift but not via OT or statistical signatures. |
| 7 | **Separation and Collaboration: Two-Level Routing Grouped MoE for MDCL** (arXiv:2508.07738) | 2025 | - | Two-level routing (inter-group via task prototypes, intra-group via learned router). Uses task prototype distance for routing — conceptually related to "matching to knowledge signatures" but prototypes are simple mean vectors, not rich statistical distributions. |
| 8 | **SCDEM: Self-Controlled Dynamic Expansion Model for CL** (arXiv:2504.10561) | 2025 | - | Multi-backbone + dynamic expert expansion. Uses **OT distance** for Feature Distribution Consistency (FDC) to align old/new representations. **Closest overlap: uses OT in CL with expert expansion, but OT is for feature alignment, NOT routing.** |
| 9 | **Boosting CL of VLMs via MoE Adapters** (arXiv:2403.11549) | 2024 | CVPR | MoE adapters for continual VLM learning with routing. Standard softmax gating. |
| 10 | **SAME: Stabilized MoE for Multimodal Continual Instruction Tuning** (arXiv:2602.01990) | 2026 | - | MoE for continual instruction tuning. Focuses on stabilization strategies. |
| 11 | **Dynamic MoE of Curriculum LoRA Experts for Continual Multimodal IT** (arXiv:2506.11672) | 2025 | ICML | Dynamic architecture expansion under budget. Curriculum-based expert management. |
| 12 | **MoTE: Mixture of Task-specific Experts for PTM-Based CIL** (arXiv:2506.11038) | 2025 | KBS | Task-specific experts with pre-trained model. Standard routing mechanisms. |

### C. Statistical Distributions in Continual Learning (Component 1 overlap)

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 13 | **vMF/Angular Gaussian for Online CL** (arXiv:2306.03364) | 2024 | AAAI | Uses vMF and Angular Gaussian distributions for **representation learning** in online CL. Pushes representations toward fixed prior directions on hypersphere. **Directly relevant to Component 1** — but uses vMF as a loss function, NOT as a routing signature for expert modules. |
| 14 | **Interactive CL: Fast and Slow Thinking** (arXiv:2403.02628) | 2024 | CVPR | vMF-related distributions in CL context for cognitive-inspired learning. |
| 15 | **General Incremental Learning with Domain-aware Categorical Representations** (arXiv:2204.04078) | 2022 | CVPR | Domain-aware representations for incremental learning using distributional methods. |

### D. Backbone Feature Drift Compensation (Component 3 overlap)

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 16 | **Exemplar-free CL via Learnable Drift Compensation (LDC)** (arXiv:2407.08536) | 2024 | ECCV | Learns a drift compensation module to correct for feature drift in backbones. **Directly relevant to Component 3** but uses a learned correction, not a penalty loss. |
| 17 | **Exemplar-free CL of ViTs via Gated Class-Attention and Cascaded Feature Drift Compensation** (arXiv:2211.12292) | 2023 | - | Gated class-attention to minimize transformer drift + cascaded feature drift compensation. Relevant to anti-drift but uses gating/masking, not OT or invasion penalty. |
| 18 | **Scalable Analytic Classifiers with Associative Drift Compensation for CIL** (arXiv:2602.00144) | 2026 | - | Analytic classifiers with drift compensation for ViTs. Uses Gaussian Discriminant Analysis. |
| 19 | **Feature Drift Compensation Projection for Data-free Replay Continual Face Forgery Detection** (arXiv:2508.03189) | 2025 | - | Feature drift compensation projection for continual face forgery detection. |
| 20 | **Resurrecting Old Classes with New Data for Exemplar-Free CL** (arXiv:2405.19074) | 2024 | CVPR | Addresses drift compensation without exemplars. |

### E. Optimal Transport in Continual Learning (General)

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 21 | **Merging without Forgetting: Continual Fusion via OT** (arXiv:2511.19561) | 2025 | - | Uses OT for **model merging** in CL (aligning task-specific model weights). OT used for weight-space alignment, NOT input routing. |
| 22 | **LwI (workspace existing work)** | - | - | Uses OT (Sinkhorn) for **neuron alignment** between old and new models during continual learning. OT for model merging/alignment, not routing. |

### F. Geometric/Statistical Routing (Component 1+2 joint overlap)

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 23 | **Grassmannian MoE: Concentration-Controlled Routing on Subspace Manifolds** (arXiv:2602.17798) | 2026 | - | Routes using **Matrix Bingham distributions** on the Grassmannian manifold to control routing entropy. **HIGHEST OVERLAP WITH PROPOSED IDEA.** Uses statistical distributions (Bingham) for routing with concentration parameters as control knobs. However: (a) not CL-specific, (b) distributions characterize routing preferences, not task knowledge, (c) no drift/anti-invasion mechanisms. |
| 24 | **Spectral Manifold Regularization for Stable Routing in Deep MoE** (arXiv:2601.03889) | 2026 | - | Manifold-based regularization for stable/modular routing. May overlap with geometric characterization concepts. |

---

## II. NOVELTY GAPS: What Has NOT Been Done

### GAP 1: Statistical Knowledge Signatures as Expert "Fingerprints" (HIGH NOVELTY)
**No existing work** creates rich statistical distribution-based "signatures" (vMF, Bingham, GMM, etc.) that characterize what each expert **knows** — i.e., the knowledge space/competence region of each submodule. Existing works either:
- Use vMF as a **training loss** (Michel et al., AAAI 2024) — not as a module descriptor
- Use Bingham distributions for **routing control** (GrMoE, 2026) — not for knowledge characterization
- Use simple prototypes/centroids for task matching (TRGE, 2025) — not rich distributional signatures

**Your contribution**: Using multi-modal statistical distributions (vMF, Bingham, GMM combinations) as a formal **fingerprint** of each module's learned knowledge region. This creates a principled, interpretable language for what each expert "knows."

### GAP 2: OT as Distribution-Matching Routing (not just Load-Balancing) (HIGH NOVELTY)
All existing OT-based routing (BASE Layers, Sinkhorn Routing, SSR) uses OT to solve a **load-balancing** problem: distribute tokens evenly across experts. The cost matrix is typically derived from learned linear projections.

**No existing work** uses OT with a cost matrix derived from **distributional distances** between input statistics and expert knowledge signatures. This is a qualitatively different OT formulation:
- Existing: $\min_{\pi} \sum_{ij} c_{ij}\pi_{ij}$ where $c_{ij} = -\text{score}(x_i, e_j)$ (learned similarity)
- Proposed: $\min_{\pi} \sum_{ij} d(P_{\text{input}_i}, Q_{\text{expert}_j})\pi_{ij}$ where $d$ is a distributional distance (e.g., KL between vMF distributions)

### GAP 3: Three-Component Integration (VERY HIGH NOVELTY)
**No paper** combines all three:
1. Statistical distribution signatures for module knowledge
2. OT-based distribution-matching routing
3. Backbone anti-drift + anti-invasion penalty

The closest works address at most 2 of 3 and in different ways:
- SCDEM: OT for alignment + expert expansion (but no signature-based routing, no anti-invasion)
- GrMoE: Statistical routing (but not CL, no drift penalty)
- PASs-MoE: Router drift mitigation + expert isolation (but uses subspace methods, not OT or statistical signatures)
- LDC/FDC: Drift compensation (but single backbone, no expert routing)

### GAP 4: Anti-Invasion Loss in MoE-based CL (MODERATE-HIGH NOVELTY)
While drift compensation exists widely, the concept of an **anti-invasion loss** — explicitly preventing new task feature distributions from encroaching on old task knowledge regions in the shared backbone — is relatively unique when combined with MoE routing. Most drift compensation works operate on a single model; applying it specifically to the **shared backbone** in a modular architecture while letting the experts handle task-specific adaptation is novel.

---

## III. RISK AREAS: Where Novelty Might Be Challenged

### RISK 1: GrMoE (Grassmannian MoE) — **MEDIUM-HIGH RISK**
**Paper**: arXiv:2602.17798 (Feb 2026)  
**Why risky**: Uses Matrix Bingham distributions on Grassmannian manifolds for routing — this is statistical-distribution-based routing, the closest conceptual cousin to your idea.  
**Mitigation**: (a) GrMoE is NOT for continual learning, (b) Bingham controls routing entropy, not knowledge characterization, (c) no drift/anti-invasion mechanisms. Your work must clearly differentiate the "signature" interpretation from the "routing control" interpretation.

### RISK 2: Selective Sinkhorn Routing (SSR) — **MEDIUM RISK**
**Paper**: arXiv:2511.08972 (Nov 2025)  
**Why risky**: Already formulates token-to-expert as OT using Sinkhorn.  
**Mitigation**: SSR uses OT for load-balancing only — your OT formulation uses distributional distances as cost, making it fundamentally different in semantics.

### RISK 3: SCDEM — **MEDIUM RISK**
**Paper**: arXiv:2504.10561 (Apr 2025)  
**Why risky**: Uses OT distance + dynamic expert expansion in CL. Has Feature Distribution Consistency (FDC) via OT.  
**Mitigation**: SCDEM uses OT for alignment between old/new features (preservation), NOT for routing decisions. The routing in SCDEM is separate from the OT component.

### RISK 4: PASs-MoE + CaRE — **LOW-MEDIUM RISK**
**Papers**: arXiv:2601.13020, arXiv:2602.03473 (Jan-Feb 2026)  
**Why risky**: Active area of research on CL + MoE routing with drift considerations.  
**Mitigation**: These use learned subspace methods (PAS) and bi-level routing (task-router + expert-router), not distribution-matching OT.

### RISK 5: vMF for Online CL — **LOW RISK**
**Paper**: arXiv:2306.03364 (AAAI 2024)  
**Why risky**: Same statistical tool (vMF) same domain (CL).  
**Mitigation**: Uses vMF as training loss, not as module knowledge signature. No MoE, no routing.

---

## IV. OVERALL NOVELTY ASSESSMENT

### Rating: **HIGH (with specific caveats)**

### Justification:

**Strengths of novelty:**

1. **No existing paper** combines statistical knowledge signatures + OT-based distribution-matching routing + backbone anti-drift in a unified CL framework. The **three-way integration** is clearly novel.

2. **The "knowledge signature" concept** — using rich statistical distributions (vMF, Bingham, GMM) to create interpretable fingerprints of what each expert module has learned — is a genuinely new formulation. Existing works use distributions either for training losses or for routing entropy control, but not as descriptive signatures of module competence.

3. **OT for distribution-matching routing** (as opposed to load-balancing) is a new semantic interpretation of OT in the MoE context. Using distributional distances in the cost matrix of the transport problem is novel.

4. **Anti-invasion loss for shared backbone** in a modular CL architecture (protecting old task regions while allowing new learning) is novel as a combination — though drift compensation alone is well-studied.

**Caveats:**

1. **GrMoE (Feb 2026)** is the closest risk — a reviewer familiar with GrMoE might see conceptual similarity in "statistical distributions for routing." You MUST clearly explain why knowledge signatures ≠ routing entropy control.

2. **SSR (Nov 2025)** + **BASE Layers** have established OT for MoE routing — you need to clearly differentiate cost matrix semantics.

3. The field of **MoE for CL** is extremely active (12+ papers in 2025-2026 alone). Given the fast pace, there's a ~15-20% risk that a similar combined idea could appear before submission.

**Recommended positioning:**  
Frame as: *"First unified framework that creates interpretable statistical knowledge signatures for expert modules and uses Optimal Transport not for load balancing but for semantically-grounded distribution-matching routing in continual learning, complemented by backbone anti-drift protection."*

---

**Summary Table:**

| Component | Individual Novelty | Closest Overlap | Risk Level |
|-----------|-------------------|-----------------|------------|
| Statistical Knowledge Signatures | **High** | vMF for Online CL (AAAI'24), GrMoE (Feb'26) | Medium |
| OT as Distribution-Matching Routing | **High** | SSR (Nov'25), BASE Layers (ICML'21) | Medium |
| Backbone Anti-Drift + Anti-Invasion | **Medium** | LDC (ECCV'24), Cascaded FDC (2022) | Low-Medium |
| **Three-Component Integration** | **Very High** | SCDEM (Apr'25), PASs-MoE (Jan'26) | Low |
