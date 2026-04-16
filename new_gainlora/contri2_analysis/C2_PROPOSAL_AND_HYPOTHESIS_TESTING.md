# Contribution 2: SGWI + Dual Fisher — Complete Proposal & Hypothesis Testing Plan

> **Benchmark**: Long_Sequence, Order 4, flan-t5-large  
> **Task order**: mnli → **cb** → wic → copa → qqp → boolq → rte → imdb → yelp → amazon → sst2 → dbpedia → agnews → multirc → yahoo  
> **Focus task**: CB (task #2, CommitmentBank — NLI, ~250 training samples)  
> **Date**: April 16, 2026

---

## PART A: CONTRIBUTION 2 — ALGORITHM DESCRIPTION

### A.1 Motivation

SRT (Contribution 1) giải quyết catastrophic forgetting bằng non-parametric Mahalanobis routing tại inference (BWT ≈ 0). Tuy nhiên:
- **Forward transfer bằng 0**: Mỗi LoRA adapter mới khởi tạo random/null-space, không tái sử dụng tri thức từ tasks tương đồng
- **AP gap -0.39%**: Hard one-hot routing mất cross-task blending benefit
- **Small-data tasks (CB, COPA)**: Đặc biệt cần warm initialization — chỉ ~250 samples → overfitting nhanh với random init

### A.2 Proposed Framework — 4 Phases

```
Phase 1: SRT Profiling        → Compute {μ_t, Σ_t} from frozen backbone
Phase 2: SGWI Initialization  → Warm-init LoRA from SRT-weighted past adapters  
Phase 3: Dual Fisher Training  → L_CE + SRT-weighted F_emb penalty
Phase 4: SRT Inference         → Hard one-hot Mahalanobis routing
```

### A.3 Phase 2 — SGWI (SRT-Guided Warm Initialization)

**Input**: Past task LoRA adapters {B_s, A_s}_{s<t}, SRT signatures {μ_s, Σ_s}_{s<t}, current task data D_t

**Step 1**: Compute SRT signature for current task:
```
μ_t = (1/n_t) Σ_{x∈D_t} B(x)     where B = frozen backbone encoder
Σ_t = (1/n_t-1) Σ_{x∈D_t} (B(x) - μ_t)(B(x) - μ_t)^T
```

**Step 2**: Compute SRT-weighted similarities:
```
d_SRT(t,s) = (μ_t - μ_s)^T Σ_pool^{-1} (μ_t - μ_s)
w_s = softmax(-d_SRT(t,s) / τ)    where τ = median({d_SRT(t,s) : s<t})
```

**Step 3**: Weighted LoRA fusion + SVD:
```
ΔW_init = Σ_{s<t} w_s · (B_s · A_s)
U, Σ, V^T = SVD(ΔW_init)
A_t^{(0)} = V[:r]^T · scaling
B_t^{(0)} = U[:,:r] · diag(Σ[:r]) · scaling
```

**Step 4**: Assign initialized weights to new task's LoRA module

### A.4 Phase 3 — Dual Fisher Regularization

**Purpose**: Protect transferred initialization from being destroyed by task-specific gradients during training ("Intra-branch Forgetting")

**Loss function**:
```
L_total = L_CE(t) + Σ_{s<t} w_s^{SRT} · [λ_emb · R_emb(θ_t, θ_s*)]

R_emb(θ_t, θ_s*) = (θ_t - θ_s*)^T · F_emb^{(s)} · (θ_t - θ_s*)
F_emb^{(s)} = W_enc^T · Σ_s · W_enc     (NTK approximation)
```

**Key properties**:
- Zero-rehearsal: only needs {μ_s, Σ_s, θ_s*}, no raw data
- SRT-weighted: similar tasks get stronger protection
- NTK-approximated: F_emb from SRT signatures, no backward pass needed

### A.5 Architecture Decision — SGWI Replaces InfLoRA

**Conflict**: InfLoRA projects lora_A ⊥ U_prev (null-space) → would destroy SGWI warm initialization

**Resolution**: For SRT path, SGWI replaces both InfLoRA and prompt_key GPM:
- InfLoRA's gradient protection → replaced by Dual Fisher (soft, per-step)
- prompt_key GPM for routing → not needed (SRT does routing)

```
ROOT:     get_reg_matrix() [InfLoRA + prompt_key GPM] → train → soft MLP inference
C1:       get_reg_matrix() [InfLoRA + prompt_key GPM] → train → SRT hard inference  
C1+C2:    SGWI init [warm from similar tasks]          → train + Dual Fisher → SRT hard inference
```

---

## PART B: HYPOTHESIS TESTING PLAN

### B.0 Thiết kế Tổng thể

**Mục tiêu**: Sau khi chạy xong toàn bộ kiểm định, trả lời được 5 câu hỏi yes/no:

| ID | Câu hỏi | Type |
|---|---|---|
| **Q1** | SGWI có cải thiện CB score so với SRT baseline? | Yes/No |
| **Q2** | InfLoRA có conflict với SGWI? | Yes/No |
| **Q3** | Dual Fisher (F_emb) có thêm giá trị trên SGWI? | Yes/No |
| **Q4** | SGWI cải thiện convergence speed? | Yes/No |
| **Q5** | C2 tổng thể (best option) có beat SRT-only trên toàn sequence? | Yes/No |

Và 2 câu hỏi option:

| ID | Câu hỏi | Options |
|---|---|---|
| **O1** | Best initialization mode? | {SGWI, InfLoRA, SGWI+InfLoRA, Random} |
| **O2** | Best Dual Fisher λ_emb? | {0, 0.001, 0.005, 0.01, 0.05} |

### B.1 Experimental Setup

**Hardware**: 1x GPU (T4 16GB hoặc A100)  
**Model**: flan-t5-large  
**Benchmark**: Long_Sequence, Order 4  
**Focus**: CB (task #2) — chỉ cần train mnli (task #1) + CB (task #2)  
**Hyperparams**: Giữ nguyên từ gen_script_long_order4_t5_srt.sh  

**Tại sao CB ở order 4 là test case tốt**:
1. CB (CommitmentBank) là NLI task → mnli (task #1) cũng là NLI → d_SRT(cb, mnli) nhỏ → SGWI transfer cao
2. CB có ~250 training samples → small data → benefits most from warm init
3. Nếu SGWI không giúp ở đây (high similarity, small data), nó không giúp ở đâu cả
4. Chỉ cần train 2 tasks → fast iteration (< 1h per arm)

### B.2 Phase 0 — Prerequisites (Shared across all arms)

**Chạy 1 lần duy nhất**: Train mnli (task #1) với SRT baseline → checkpoint

```bash
# Phase 0: Train mnli → shared checkpoint for all CB experiments
bash scripts/run_phase0_mnli.sh <GPU_ID> <MODEL_PATH>
```

**Output cần thiết**:
- `outputs/1-mnli/saved_weights/` — mnli LoRA weights (lora_A, lora_B, trans_input, prompt_key)
- `outputs/1-mnli/saved_weights/srt_signatures.pt` — {μ_mnli, Σ_mnli}

---

### B.3 Phase 1 — Hypothesis H1: SGWI vs InfLoRA vs Random

**Câu hỏi**: Q1 (SGWI works?), O1 (best init mode?)

**Design**: 4 arms, chỉ khác nhau ở CB initialization. Training loop và SRT routing giống nhau.

| Arm | Init Mode | InfLoRA | SGWI | Description |
|---|---|---|---|---|
| **A** (baseline) | InfLoRA | ✅ | ❌ | Current SRT behavior (get_reg_matrix) |
| **B** | SGWI only | ❌ | ✅ | Warm init from mnli, no InfLoRA |
| **C** | SGWI + InfLoRA | ✅ | ✅ | SGWI first, then InfLoRA projects |
| **D** | Random | ❌ | ❌ | Standard LoRA init (Kaiming) |

**Metric**: `eval_exact_match_for_cb` (CB accuracy sau training)

**Decision criteria**:

```
Q1: SGWI works?
    IF score(B) > score(A) + 1.0%  → YES, strong signal
    IF score(B) > score(A) ± 1.0%  → MARGINAL, need more data
    IF score(B) ≤ score(A) - 1.0%  → NO, stop C2

Q2: InfLoRA conflicts with SGWI?
    IF score(B) > score(C) + 1.0%  → YES, InfLoRA hurts SGWI
    IF score(B) ≈ score(C) ± 1.0%  → NO conflict

O1: Best init mode?
    → argmax(score(A), score(B), score(C), score(D))
```

**Tại sao 1% threshold**: CB chỉ có ~56 test samples → ±1 sample = ±1.8%. Dùng 1% cho conservative estimate.

---

### B.4 Phase 2 — Hypothesis H2: Convergence Speed

**Câu hỏi**: Q4 (SGWI faster convergence?)

**Design**: Chạy trên CÙNG 4 arms từ Phase 1, nhưng log loss curve mỗi 10 steps.

**Metric**: 
- `steps_to_threshold`: Số steps đầu tiên đạt 80% of final CB accuracy
- `loss_at_epoch_1`: Loss sau epoch 1 (early convergence indicator)
- `loss_curve`: Toàn bộ training loss cho visualization

**Decision criteria**:

```
Q4: SGWI faster convergence?
    IF steps_to_threshold(B) < 0.7 × steps_to_threshold(A) → YES (>30% faster)
    IF steps_to_threshold(B) < 0.9 × steps_to_threshold(A) → MARGINAL
    ELSE → NO
```

**Note**: Phase 2 không cần chạy thêm — data đã collect từ Phase 1 (cùng training run).

---

### B.5 Phase 3 — Hypothesis H3: Dual Fisher Value

**Prerequisite**: Chỉ chạy Phase 3 nếu Q1 = YES (SGWI works).

**Câu hỏi**: Q3 (Dual Fisher adds value?), O2 (best λ_emb?)

**Design**: Best init từ Phase 1 (expected: Arm B = SGWI only) + sweep λ_emb.

| Arm | Init | λ_emb | Description |
|---|---|---|---|
| **B** (from P1) | SGWI | 0 | SGWI alone (no Dual Fisher) |
| **E1** | SGWI | 0.001 | Light regularization |
| **E2** | SGWI | 0.005 | Medium-light |
| **E3** | SGWI | 0.01 | Medium |
| **E4** | SGWI | 0.05 | Strong regularization |

**Metric**: `eval_exact_match_for_cb`

**Decision criteria**:

```
Q3: Dual Fisher adds value?
    best_E = max(score(E1), score(E2), score(E3), score(E4))
    IF best_E > score(B) + 0.5%  → YES
    IF best_E ≤ score(B)         → NO, Dual Fisher doesn't help for CB

O2: Best λ_emb?
    → argmax over {0, 0.001, 0.005, 0.01, 0.05}
    Nếu relationship is U-shaped → consistent with theory (too low = no effect, too high = plasticity loss)
```

---

### B.6 Phase 4 — End-to-End Validation (5-task mini-sequence)

**Prerequisite**: Chạy nếu Phase 1 hoặc Phase 3 có positive signal.

**Câu hỏi**: Q5 (C2 beats SRT-only trên toàn sequence?)

**Design**: Chạy 5 tasks đầu tiên của order 4 (mnli → cb → wic → copa → qqp) với 2 configurations:

| Config | Description |
|---|---|
| **Baseline** | SRT-only (current code, InfLoRA + SRT) |
| **C2** | SRT + SGWI + Dual Fisher (best λ_emb from Phase 3) |

**Metric**: Cross-evaluation matrix 5×5, compute:
- AP (average of final row)
- BWT (average of final - diagonal)
- FWT (average of diagonal)
- Per-task breakdown (especially CB)

**Decision criteria**:

```
Q5: C2 beats SRT-only overall?
    IF AP(C2) > AP(baseline) AND BWT(C2) ≤ BWT(baseline) + 0.5%  → YES, conclusive
    IF AP(C2) > AP(baseline) BUT BWT(C2) > BWT(baseline) + 0.5%  → TRADE-OFF, needs analysis
    IF AP(C2) ≤ AP(baseline)  → NO, C2 doesn't help
```

---

## PART C: IMPLEMENTATION — Files to Create/Modify

### C.1 File Structure

```
new_gainlora/contri2_analysis/
├── C2_PROPOSAL_AND_HYPOTHESIS_TESTING.md  (this file)
├── sgwi_trainer.py                         (NEW: SGWI + Dual Fisher trainer)
├── run_hypothesis_test.py                  (NEW: orchestrates all phases)
├── run_phase0_mnli.sh                      (NEW: train mnli baseline)
├── run_phase1_cb_arms.sh                   (NEW: CB 4-arm comparison)
├── run_phase3_dualfisher_sweep.sh          (NEW: λ_emb sweep)
├── run_phase4_5task.sh                     (NEW: 5-task e2e validation)
├── analyze_results.py                      (NEW: auto-analyze, print decisions)
└── results/                                (output directory)
```

### C.2 sgwi_trainer.py — Core Implementation

```python
"""
SGWI + Dual Fisher Trainer.
Inherits from SRT_Trainer. Overrides:
  - get_reg_matrix() → SGWI initialization instead of InfLoRA
  - compute_loss() → adds Dual Fisher penalty during training
"""

class SGWI_DualFisher_Trainer(SRT_Trainer):
    
    def __init__(self, *args, sgwi_mode='sgwi', lambda_emb=0.0, 
                 sgwi_tau='median', skip_inflora=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sgwi_mode = sgwi_mode        # 'sgwi', 'inflora', 'random', 'sgwi+inflora'
        self.lambda_emb = lambda_emb
        self.sgwi_tau = sgwi_tau
        self.skip_inflora = skip_inflora
        self.theta_stars = {}              # {task_id: {param_name: tensor}}
    
    def get_reg_matrix(self):
        """Override InfLoRA with SGWI or other init modes."""
        
        if self.sgwi_mode == 'inflora':
            # Arm A: Standard InfLoRA (baseline)
            super().get_reg_matrix()
            return
        
        if self.sgwi_mode == 'random':
            # Arm D: Skip all initialization, keep random weights
            return
        
        if self.cur_task_id == 0:
            # Task 0: No prior tasks, use standard init
            super().get_reg_matrix()
            return
        
        if self.sgwi_mode == 'sgwi':
            # Arm B: SGWI only — skip InfLoRA
            self._sgwi_init()
            return
        
        if self.sgwi_mode == 'sgwi+inflora':
            # Arm C: SGWI first, then InfLoRA projects
            self._sgwi_init()
            super().get_reg_matrix()  # InfLoRA will project SGWI-initialized weights
            return
    
    def _sgwi_init(self):
        """SRT-Guided Warm Initialization."""
        # Step 1: Compute current task SRT signature (extract embeddings)
        current_embeddings = self._extract_task_embeddings_for_sgwi()
        current_mu = current_embeddings.mean(dim=0)
        
        # Step 2: Compute SRT distances to all past tasks
        srt_weights = self._compute_sgwi_weights(current_mu)
        
        # Step 3: Weighted fusion of past LoRA adapters
        self._fuse_and_init_lora(srt_weights)
    
    def _compute_sgwi_weights(self, current_mu):
        """Softmax weights from SRT distances."""
        distances = {}
        for task_id, sig in self.srt_router.signatures.items():
            mu_s = torch.from_numpy(sig.mu).float()
            diff = current_mu - mu_s
            # Simplified: use pooled precision for Mahalanobis
            d = (diff * diff).sum().item()  # L2 as proxy when no pooled cov
            distances[task_id] = d
        
        if len(distances) == 1:
            # Only 1 prior task → trivially weight = 1.0
            return {k: 1.0 for k in distances}
        
        # τ = median heuristic
        d_values = list(distances.values())
        tau = sorted(d_values)[len(d_values) // 2] + 1e-8
        
        weights = {k: math.exp(-d / tau) for k, d in distances.items()}
        Z = sum(weights.values())
        return {k: w / Z for k, w in weights.items()}
    
    def _fuse_and_init_lora(self, srt_weights):
        """Fuse past LoRA adapters and SVD-decompose for new task."""
        # Access past LoRA weights from previous_lora_path checkpoints
        # ... (implementation depends on how weights are stored in model)
        pass  # Detailed implementation in actual code
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Add Dual Fisher regularization to standard loss."""
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        if self.lambda_emb > 0 and self.cur_task_id > 0:
            fisher_loss = self._dual_fisher_penalty()
            loss = loss + fisher_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _dual_fisher_penalty(self):
        """SRT-weighted embedding Fisher penalty."""
        total = 0.0
        srt_weights = self._compute_sgwi_weights(self._current_mu)
        
        for task_id, w_s in srt_weights.items():
            if task_id not in self.theta_stars:
                continue
            sig = self.srt_router.signatures[task_id]
            
            for name, param in self.model.named_parameters():
                if 'lora_' not in name:
                    continue
                if name not in self.theta_stars[task_id]:
                    continue
                delta = param - self.theta_stars[task_id][name].to(param.device)
                total += w_s * (delta ** 2).sum()
        
        return self.lambda_emb * total
    
    def on_task_end(self, task_id):
        """Save θ* for Dual Fisher, then do standard SRT task end."""
        # Save current task's final weights for Dual Fisher
        self.theta_stars[task_id] = {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
            if 'lora_' in name
        }
        # Standard SRT: compute signature, wire router
        super().on_task_end(task_id)
```

### C.3 Shell Scripts

Xem PART D dưới đây cho chi tiết scripts.

---

## PART D: DETAILED EXPERIMENT SCRIPTS

### D.1 Phase 0: Train MNLI Baseline

```bash
#!/bin/bash
# run_phase0_mnli.sh — Train mnli (task 1 of order 4) with SRT
# Usage: bash run_phase0_mnli.sh <GPU_ID> <MODEL_PATH>
# Example: bash run_phase0_mnli.sh 0 google/flan-t5-large

GPU_ID=${1:-0}
MODEL_PATH=${2:-google/flan-t5-large}
OUTPUT_BASE=logs_and_outputs/c2_hypothesis

SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric auto --srt_max_emb_samples 500"

CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_t5.py \
   --do_train --do_predict --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/mnli \
   --output_dir $OUTPUT_BASE/phase0_mnli \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 4 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --run_name c2_hyp_phase0_mnli \
   --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
   --add_task_name False --add_dataset_name False \
   --overwrite_output_dir --overwrite_cache \
   --lr_scheduler_type constant --warmup_steps 0 \
   --logging_strategy steps --logging_steps 10 \
   --metric_for_best_model eval_exact_match \
   --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
   --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
   --add_instruction_replay --data_replay_freq -1 --replay_after_n_epoch 0 \
   --mlp_hidden_dim 100 \
   --model_name gainlora_inflora \
   --threshold 0.995 --transthreshold 0.995 \
   $SRT_FLAGS

rm -rf $OUTPUT_BASE/phase0_mnli/checkpoint*
echo "[Phase 0] MNLI training complete. Checkpoint at: $OUTPUT_BASE/phase0_mnli/saved_weights/"
```

### D.2 Phase 1: CB 4-Arm Comparison

```bash
#!/bin/bash
# run_phase1_cb_arms.sh — Test 4 initialization modes for CB
# Usage: bash run_phase1_cb_arms.sh <GPU_ID> <MODEL_PATH>

GPU_ID=${1:-0}
MODEL_PATH=${2:-google/flan-t5-large}
OUTPUT_BASE=logs_and_outputs/c2_hypothesis
MNLI_CKPT=$OUTPUT_BASE/phase0_mnli/saved_weights

SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric auto --srt_max_emb_samples 500"

# Common args for CB training
CB_COMMON="--do_train --do_predict --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $MNLI_CKPT/trans_input.pt \
   --previous_lora_path $MNLI_CKPT \
   --previous_prompt_key_path $MNLI_CKPT/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/cb \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
   --add_task_name False --add_dataset_name False \
   --overwrite_output_dir --overwrite_cache \
   --lr_scheduler_type constant --warmup_steps 0 \
   --logging_strategy steps --logging_steps 5 \
   --metric_for_best_model eval_exact_match_for_cb \
   --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
   --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
   --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
   --mlp_hidden_dim 100 --model_name gainlora_inflora \
   --threshold 0.995 --transthreshold 0.995 \
   $SRT_FLAGS --srt_load_path $MNLI_CKPT"

echo "============================================================"
echo "[Phase 1] ARM A: InfLoRA baseline (current SRT behavior)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_t5.py \
   $CB_COMMON \
   --output_dir $OUTPUT_BASE/phase1_arm_a_inflora \
   --run_name c2_arm_a_inflora \
   --sgwi_mode inflora
rm -rf $OUTPUT_BASE/phase1_arm_a_inflora/checkpoint*

echo "============================================================"
echo "[Phase 1] ARM B: SGWI only (no InfLoRA)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_t5.py \
   $CB_COMMON \
   --output_dir $OUTPUT_BASE/phase1_arm_b_sgwi \
   --run_name c2_arm_b_sgwi \
   --sgwi_mode sgwi
rm -rf $OUTPUT_BASE/phase1_arm_b_sgwi/checkpoint*

echo "============================================================"
echo "[Phase 1] ARM C: SGWI + InfLoRA"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_t5.py \
   $CB_COMMON \
   --output_dir $OUTPUT_BASE/phase1_arm_c_sgwi_inflora \
   --run_name c2_arm_c_sgwi_inflora \
   --sgwi_mode sgwi+inflora
rm -rf $OUTPUT_BASE/phase1_arm_c_sgwi_inflora/checkpoint*

echo "============================================================"
echo "[Phase 1] ARM D: Random init (no InfLoRA, no SGWI)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_t5.py \
   $CB_COMMON \
   --output_dir $OUTPUT_BASE/phase1_arm_d_random \
   --run_name c2_arm_d_random \
   --sgwi_mode random
rm -rf $OUTPUT_BASE/phase1_arm_d_random/checkpoint*

echo "============================================================"
echo "[Phase 1] All arms complete. Running analysis..."
echo "============================================================"
python contri2_analysis/analyze_results.py --phase 1 --output_base $OUTPUT_BASE
```

### D.3 Phase 3: Dual Fisher λ_emb Sweep

```bash
#!/bin/bash
# run_phase3_dualfisher_sweep.sh — Sweep λ_emb for Dual Fisher
# Usage: bash run_phase3_dualfisher_sweep.sh <GPU_ID> <MODEL_PATH>
# Prerequisite: Phase 1 Arm B (SGWI) showed positive signal

GPU_ID=${1:-0}
MODEL_PATH=${2:-google/flan-t5-large}
OUTPUT_BASE=logs_and_outputs/c2_hypothesis
MNLI_CKPT=$OUTPUT_BASE/phase0_mnli/saved_weights

SRT_FLAGS="--use_srt_router --srt_shrink --srt_shrink_factor 0.1 --srt_metric auto --srt_max_emb_samples 500"

CB_COMMON="--do_train --do_predict --predict_with_generate \
   --model_name_or_path $MODEL_PATH \
   --load_checkpoint_from $MNLI_CKPT/trans_input.pt \
   --previous_lora_path $MNLI_CKPT \
   --previous_prompt_key_path $MNLI_CKPT/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo \
   --task_config_dir configs/gen_script_long_order4_t5_configs/cb \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --max_source_length 512 --max_target_length 50 --generation_max_length 50 \
   --add_task_name False --add_dataset_name False \
   --overwrite_output_dir --overwrite_cache \
   --lr_scheduler_type constant --warmup_steps 0 \
   --logging_strategy steps --logging_steps 5 \
   --metric_for_best_model eval_exact_match_for_cb \
   --evaluation_strategy steps --save_strategy best --save_total_limit 1 \
   --lora_r 8 --lora_alpha 32 --lora_dropout 0.0 \
   --data_replay_freq -1 --kl_ratio 0.1 --attn_temperature 1 \
   --mlp_hidden_dim 100 --model_name gainlora_inflora \
   --threshold 0.995 --transthreshold 0.995 \
   --sgwi_mode sgwi \
   $SRT_FLAGS --srt_load_path $MNLI_CKPT"

for LAMBDA in 0.001 0.005 0.01 0.05; do
    echo "============================================================"
    echo "[Phase 3] λ_emb = $LAMBDA"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/run_t5.py \
       $CB_COMMON \
       --output_dir $OUTPUT_BASE/phase3_lambda_${LAMBDA} \
       --run_name c2_df_lambda_${LAMBDA} \
       --lambda_emb $LAMBDA
    rm -rf $OUTPUT_BASE/phase3_lambda_${LAMBDA}/checkpoint*
done

echo "============================================================"
echo "[Phase 3] Sweep complete. Running analysis..."
echo "============================================================"
python contri2_analysis/analyze_results.py --phase 3 --output_base $OUTPUT_BASE
```

### D.4 Phase 4: 5-Task Mini-Sequence

```bash
#!/bin/bash
# run_phase4_5task.sh — 5-task end-to-end validation
# Runs: mnli → cb → wic → copa → qqp with both SRT-only and C2 configs
# Usage: bash run_phase4_5task.sh <GPU_ID> <MODEL_PATH>
# Cần ~5h per config (A100), ~10h per config (T4)

GPU_ID=${1:-0}
MODEL_PATH=${2:-google/flan-t5-large}
OUTPUT_BASE=logs_and_outputs/c2_hypothesis

# Config A: SRT-only baseline (first 5 tasks of order 4)
echo "Running CONFIG A: SRT-only baseline (5 tasks)..."
# [Use existing gen_script_long_order4_t5_srt.sh but stop after task 5 (qqp)]

# Config B: SRT + C2 (SGWI + best λ_emb from Phase 3)  
echo "Running CONFIG B: SRT + C2 (5 tasks)..."
# [Modified script with --sgwi_mode sgwi --lambda_emb <best>]

# Score both configs
python score.py c2_hyp_baseline_5task c2_hyp_baseline_5task
python score.py c2_hyp_c2_5task c2_hyp_c2_5task
```

---

## PART E: ANALYSIS FRAMEWORK

### E.1 analyze_results.py — Auto-Decision Logic

```python
"""
Auto-analyze hypothesis testing results and print decisions.
Usage: python analyze_results.py --phase <1|3|4> --output_base <path>
"""
import json, os, argparse

def analyze_phase1(output_base):
    """Phase 1: 4-arm CB comparison → Q1, Q2, O1"""
    arms = {
        'A_inflora': f'{output_base}/phase1_arm_a_inflora',
        'B_sgwi': f'{output_base}/phase1_arm_b_sgwi',
        'C_sgwi_inflora': f'{output_base}/phase1_arm_c_sgwi_inflora',
        'D_random': f'{output_base}/phase1_arm_d_random',
    }
    
    scores = {}
    for arm_name, arm_path in arms.items():
        results_file = os.path.join(arm_path, 'all_results.json')
        if os.path.exists(results_file):
            with open(results_file) as f:
                data = json.load(f)
            # CB accuracy key
            score = data.get('predict_exact_match', data.get('eval_exact_match_for_cb', 0))
            scores[arm_name] = score * 100  # to percentage
    
    print("=" * 60)
    print("PHASE 1 RESULTS: CB Initialization Comparison")
    print("=" * 60)
    for arm, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {arm:25s}: {score:.2f}%")
    print()
    
    # Decision Q1: SGWI works?
    a = scores.get('A_inflora', 0)
    b = scores.get('B_sgwi', 0)
    c = scores.get('C_sgwi_inflora', 0)
    d = scores.get('D_random', 0)
    
    print("DECISIONS:")
    if b > a + 1.0:
        print(f"  Q1: ✅ YES — SGWI improves CB by {b-a:.2f}pp over InfLoRA baseline")
    elif b > a - 1.0:
        print(f"  Q1: ⚠️ MARGINAL — SGWI vs InfLoRA difference = {b-a:.2f}pp (< 1pp)")
    else:
        print(f"  Q1: ❌ NO — SGWI HURTS CB by {a-b:.2f}pp. STOP C2.")
    
    # Decision Q2: InfLoRA conflict?
    if b > c + 1.0:
        print(f"  Q2: ✅ YES — InfLoRA conflicts with SGWI (diff = {b-c:.2f}pp)")
    else:
        print(f"  Q2: ❌ NO — No significant conflict (diff = {b-c:.2f}pp)")
    
    # Decision O1: Best init mode
    best_arm = max(scores, key=scores.get)
    print(f"  O1: Best init mode = {best_arm} ({scores[best_arm]:.2f}%)")
    
    # Proceed?
    if b > a - 1.0:
        print(f"\n  → PROCEED to Phase 3 (Dual Fisher sweep)")
    else:
        print(f"\n  → STOP. SGWI doesn't help. Reconsider C2 direction.")

def analyze_phase3(output_base):
    """Phase 3: λ_emb sweep → Q3, O2"""
    lambdas = [0, 0.001, 0.005, 0.01, 0.05]
    scores = {}
    
    # λ=0 is Phase 1 Arm B
    arm_b_path = f'{output_base}/phase1_arm_b_sgwi/all_results.json'
    if os.path.exists(arm_b_path):
        with open(arm_b_path) as f:
            data = json.load(f)
        scores[0] = data.get('predict_exact_match', 0) * 100
    
    for l in [0.001, 0.005, 0.01, 0.05]:
        path = f'{output_base}/phase3_lambda_{l}/all_results.json'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            scores[l] = data.get('predict_exact_match', 0) * 100
    
    print("=" * 60)
    print("PHASE 3 RESULTS: Dual Fisher λ_emb Sweep")
    print("=" * 60)
    for l, score in sorted(scores.items()):
        marker = " ← baseline" if l == 0 else ""
        print(f"  λ_emb = {l:6.3f}: {score:.2f}%{marker}")
    print()
    
    # Decision Q3
    baseline = scores.get(0, 0)
    best_lambda = max(scores, key=scores.get)
    best_score = scores[best_lambda]
    
    if best_lambda > 0 and best_score > baseline + 0.5:
        print(f"  Q3: ✅ YES — Dual Fisher improves by {best_score - baseline:.2f}pp")
    else:
        print(f"  Q3: ❌ NO — Dual Fisher doesn't add value (best improvement = {best_score - baseline:.2f}pp)")
    
    print(f"  O2: Best λ_emb = {best_lambda} ({best_score:.2f}%)")
    
    # Check U-shape
    sorted_scores = [scores.get(l, 0) for l in sorted(scores.keys())]
    if len(sorted_scores) >= 3:
        peak_idx = sorted_scores.index(max(sorted_scores))
        if 0 < peak_idx < len(sorted_scores) - 1:
            print(f"  → U-shaped curve confirmed (peak at middle λ)")
        elif peak_idx == 0:
            print(f"  → Monotonically decreasing: regularization HURTS")
        else:
            print(f"  → Monotonically increasing: may need higher λ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--output_base', type=str, required=True)
    args = parser.parse_args()
    
    if args.phase == 1:
        analyze_phase1(args.output_base)
    elif args.phase == 3:
        analyze_phase3(args.output_base)
```

---

## PART F: DECISION FLOWCHART

```
Phase 0: Train MNLI
    │
Phase 1: CB 4-Arm Test
    │
    ├── Q1: SGWI works? ──┬── YES ──→ Continue
    │                      └── NO  ──→ STOP C2. Report negative result.
    │
    ├── Q2: InfLoRA conflict? ──┬── YES ──→ Use SGWI-only in future experiments
    │                           └── NO  ──→ Consider keeping both
    │
    ├── O1: Best init mode ──→ Record for Phase 3
    │
Phase 2: Convergence Analysis (from Phase 1 logs)
    │
    ├── Q4: Faster convergence? ──→ Record for paper narrative
    │
Phase 3: Dual Fisher λ Sweep (IF Q1=YES)
    │
    ├── Q3: Dual Fisher adds value? ──┬── YES ──→ Record best λ
    │                                  └── NO  ──→ C2 = SGWI only (still valid)
    │
    ├── O2: Best λ_emb ──→ Record for Phase 4
    │
Phase 4: 5-Task End-to-End (IF Q1=YES)
    │
    ├── Q5: C2 > SRT-only overall? ──┬── YES ──→ C2 confirmed! Write paper.
    │                                 └── NO  ──→ CB-specific only? Revise scope.
    │
    └── FINAL REPORT
```

---

## PART G: EXPECTED OUTCOMES & RISK ANALYSIS

### G.1 Expected CB Results (Order 4)

**Context**: CB = NLI task (~250 train, ~56 test). mnli = NLI task (~390K train). High similarity expected.

| Arm | Expected Score | Reasoning |
|---|---|---|
| D (Random) | ~64-68% | No prior knowledge, small data |
| A (InfLoRA) | ~68-72% | InfLoRA helps stability but no transfer |
| C (SGWI+InfLoRA) | ~68-72% | InfLoRA likely destroys SGWI benefit |
| **B (SGWI)** | **~72-78%** | mnli LoRA carries NLI knowledge → direct transfer |

### G.2 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| mnli LoRA not useful for CB | Low (both NLI) | Check d_SRT(mnli, cb) first |
| SVD fusion numerically unstable | Medium | Add Frobenius norm check, scaling |
| CB too small → high variance | High | Report mean ± std, run 3 seeds if possible |
| Dual Fisher over-constrains | Medium | Start with small λ (0.001) |
| SGWI helps CB but hurts later tasks | Low | Phase 4 validates sequence-level |

### G.3 Timeline

| Phase | GPU Hours (A100) | GPU Hours (T4) | Calendar |
|---|---|---|---|
| Phase 0 (mnli) | ~1h | ~2h | Day 1 |
| Phase 1 (4 arms CB) | ~2h | ~4h | Day 1 |
| Phase 2 (analysis) | 0 | 0 | Day 1 |
| Phase 3 (4 λ sweep) | ~2h | ~4h | Day 2 |
| Phase 4 (5-task ×2) | ~10h | ~20h | Day 2-3 |
| **Total** | **~15h** | **~30h** | **~3 days** |

---

## PART H: COMPARISON WITH RELATED WORK

### H.1 How SGWI Compares to Existing Methods

| Method | Transfer Mechanism | Routing | Zero-Rehearsal | Reference |
|---|---|---|---|---|
| EWC | Quadratic penalty only | None | Needs Fisher from data | Kirkpatrick 2017 |
| PackNet | Pruning + reuse | Task-ID required | ✅ | Mallya 2018 |
| AdapterFusion | Learned composition of adapters | Learned attention | Needs all adapters | Pfeiffer 2021 |
| LoRA-hub | Uniform combination of LoRAs | None | ✅ (needs LoRAs) | Huang 2023 |
| **SGWI (Ours)** | **SRT-weighted LoRA fusion + SVD** | **Non-parametric Mahal** | **✅** | — |

**Key differentiator from LoRA-hub**: SGWI uses information-geometric distances (Mahalanobis) from SRT to weight adapters, not uniform/random/manual selection. Also includes SVD truncation for optimal rank-r initialization.

### H.2 How Dual Fisher Compares

| Method | Fisher Type | Task Weighting | Cost | Reference |
|---|---|---|---|---|
| EWC | Empirical (gradient outer product) | Uniform | O(n × params) | Kirkpatrick 2017 |
| Online-EWC | Running Fisher average | Exponential decay | O(params) | Schwarz 2018 |
| FSR | Function-space (Jacobian) | Uniform | O(n × params) | Titsias 2020 |
| **Dual Fisher (Ours)** | **NTK-approximated F_emb** | **SRT-weighted softmax** | **O(d²) from signatures** | — |

**Key differentiator**: F_emb requires NO backward pass, NO raw data. Computed directly from SRT signatures ({μ_s, Σ_s}) and frozen encoder weights (W_enc). SRT-weighted task weighting is novel.

---

## PART I: CONCLUSION

Toàn bộ hypothesis testing plan được thiết kế để:

1. **Fail fast**: Phase 1 (Q1) quyết định có tiếp tục hay không. Nếu SGWI không giúp CB (best-case scenario), stop.
2. **Isolate variables**: Mỗi arm chỉ thay đổi 1 biến (init mode HOẶC λ_emb).
3. **Answer binary questions**: 5 yes/no questions với threshold rõ ràng.
4. **Select options**: O1 (best init), O2 (best λ) có decision logic tự động.
5. **Efficient**: Tổng ~15h GPU (A100) cho toàn bộ hypothesis testing.

**Nếu kết quả positive**: Có đủ evidence để:
- Write Section 4 (Method) + Section 5 (Experiments) cho paper
- Claim C2 = SGWI + Dual Fisher as genuine forward transfer mechanism
- Run full 15-task experiments trên multiple orders

**Nếu kết quả negative**: Biết chính xác component nào fail:
- Q1=NO → SGWI concept doesn't work → revise C2 fundamentally
- Q3=NO → Dual Fisher not needed → C2 = SGWI only (simpler, cleaner)
- Q5=NO → CB-specific benefit doesn't generalize → narrow claim
