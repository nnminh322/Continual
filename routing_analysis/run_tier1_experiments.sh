#!/usr/bin/env bash
# run_tier1_experiments.sh — Run GALA Tier-1 training experiments
#
# Runs selected phases across backbone × benchmark combinations.
# Each phase tests a specific GALA hypothesis with actual LoRA training.
#
# Phases:
#   t1  — Proxy Validation     (Σ_residual ≈ Σ_grad?)
#   t2  — GGI Init Training    (GGI > PCA > Random init?)
#   t3  — SGR vs Hard CL       (soft penalty > hard projection?)
#   e0  — TARA Rank Sweep      (accuracy saturates at r ≈ TGC_eff?)
#   e3  — BNG vs AdamW         (preconditioning + asymmetric LR helps?)
#
# Usage:
#   bash run_tier1_experiments.sh                                    # ALL phases × ALL backbones × ALL benchmarks
#   bash run_tier1_experiments.sh --phase t1                         # proxy validation, all combos
#   bash run_tier1_experiments.sh --backbone google/flan-t5-large    # one backbone, all phases
#   bash run_tier1_experiments.sh --benchmark Long_Sequence          # one benchmark, all phases
#   bash run_tier1_experiments.sh --phase t2 --backbone google/flan-t5-xl --benchmark SuperNI
#   bash run_tier1_experiments.sh --phase t3 --quick                 # SGR quick mode

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ═══════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════
PHASE=""
BACKBONE=""
BENCHMARK=""
DATA_DIR="../improve_gainlora/CL_Benchmark"
LORA_R=8
N_EPOCHS=3
BATCH_SIZE=4
LR="1e-4"
DEVICE="auto"
QUICK=false
OUTPUT_DIR="results_con2"
GRAD_ACCUM=4
FP16=false
MAX_LENGTH=256

# Phase-specific defaults
T1_N_BATCHES=100
T1_LAYERS="0 6 12"
T2_PROBE_BATCHES=50
T3_SGR_LAMBDAS="0.01,0.05,0.1,0.5,1.0"
T3_SOFT_INIT=0.7
E0_RANKS="2,4,8,16,32"
E3_N_EPOCHS=5
E3_LR_RATIO=4.0
E3_BETA_EMA=0.99
E3_LAMBDA_BAL=0.01

# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

  --phase P          Run only phase P (t1|t2|t3|e0|e3|e4). Default: all.
  --backbone B       Run only this backbone. Default: all 3.
                     Options: google/flan-t5-large, google/flan-t5-xl, meta-llama/Llama-2-7b-hf
  --benchmark B      Run only this benchmark. Default: both.
                     Options: Long_Sequence, SuperNI
  --data_dir D       Path to CL_Benchmark/ (default: ../improve_gainlora/CL_Benchmark)
  --lora_r R         LoRA rank (default: 8)
  --n_epochs N       Training epochs (default: 3, E3 uses 5)
  --batch_size B     Batch size per GPU (default: 4)
  --grad_accum N     Gradient accumulation steps (default: 4, effective_bs=16)
  --fp16             Enable fp16 mixed precision (strongly recommended for T4)
  --max_length N     Max source sequence length (default: 256; use 128 for tight VRAM)
  --lr L             Learning rate (default: 1e-4)
  --device D         auto|cuda|cpu|mps (default: auto)
  --quick            Quick mode: T3 fewer lambdas, E0 fewer ranks
  --output_dir D     Output directory (default: results_con2/)

Examples:
  $0 --phase t2 --backbone google/flan-t5-large --benchmark Long_Sequence --fp16
  $0 --phase t2 --fp16 --grad_accum 8 --max_length 128   # very tight VRAM
  $0 --phase t3 --quick --fp16
  $0 --phase e4                             # Full GALA vs GainLoRA vs InfLoRA
  $0 --backbone google/flan-t5-xl --fp16
  $0 --fp16                                # everything with fp16
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --phase)      PHASE="$2"; shift 2 ;;
    --backbone)   BACKBONE="$2"; shift 2 ;;
    --benchmark)  BENCHMARK="$2"; shift 2 ;;
    --data_dir)   DATA_DIR="$2"; shift 2 ;;
    --lora_r)     LORA_R="$2"; shift 2 ;;
    --n_epochs)   N_EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --grad_accum) GRAD_ACCUM="$2"; shift 2 ;;
    --fp16)       FP16=true; shift ;;
    --max_length) MAX_LENGTH="$2"; shift 2 ;;
    --lr)         LR="$2"; shift 2 ;;
    --device)     DEVICE="$2"; shift 2 ;;
    --quick)      QUICK=true; shift ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)    usage ;;
    *)            echo "Unknown option: $1"; usage ;;
  esac
done

# ═══════════════════════════════════════════════════════════════════════
# Build iteration lists
# ═══════════════════════════════════════════════════════════════════════
if [[ -n "$BACKBONE" ]]; then
  BACKBONES=("$BACKBONE")
else
  BACKBONES=("google/flan-t5-large" "google/flan-t5-xl" "meta-llama/Llama-2-7b-hf")
fi

if [[ -n "$BENCHMARK" ]]; then
  BENCHMARKS=("$BENCHMARK")
else
  BENCHMARKS=("Long_Sequence" "SuperNI")
fi

if [[ -n "$PHASE" ]]; then
  PHASES=("$PHASE")
else
  PHASES=(t1 t2 e0 e3 t3 e4)  # ordered by dependency: t1 first, t3/e4 last
fi

if $QUICK; then
  T3_SGR_LAMBDAS="0.1"
  E0_RANKS="4,8,16"
fi

# ═══════════════════════════════════════════════════════════════════════
# Helper: get task config per benchmark (no associative arrays for bash 3)
# ═══════════════════════════════════════════════════════════════════════
get_single_task() {
  local bm="$1"
  case "$bm" in
    Long_Sequence) echo "sst2" ;;
    SuperNI)       echo "task363_sst2_polarity_classification" ;;
    *)             echo "sst2" ;;
  esac
}

get_t2_tasks() {
  local bm="$1"
  case "$bm" in
    Long_Sequence) echo "sst2 mnli dbpedia" ;;
    SuperNI)       echo "task363_sst2_polarity_classification task073_commonsenseqa_answer_generation task1290_xsum_summarization" ;;
    *)             echo "sst2" ;;
  esac
}

get_t3_tasks() {
  local bm="$1"
  case "$bm" in
    Long_Sequence) echo "sst2,imdb,yelp,amazon,agnews" ;;
    SuperNI)       echo "task1687_sentiment140_classification,task363_sst2_polarity_classification,task073_commonsenseqa_answer_generation,task1290_xsum_summarization,task639_multi_woz_user_utterance_generation" ;;
    *)             echo "sst2,imdb,yelp" ;;
  esac
}

# ═══════════════════════════════════════════════════════════════════════
# Print config
# ═══════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════"
echo "  GALA (Contribution 2) — Tier-1 Training Experiments"
echo "════════════════════════════════════════════════════════════"
echo "  Phases:     ${PHASES[*]}"
echo "  Backbones:  ${BACKBONES[*]}"
echo "  Benchmarks: ${BENCHMARKS[*]}"
echo "  LoRA rank:  ${LORA_R}"
echo "  Epochs:     ${N_EPOCHS} (E3: ${E3_N_EPOCHS})"
echo "  Batch:      ${BATCH_SIZE} × grad_accum=${GRAD_ACCUM} (effective=$(( BATCH_SIZE * GRAD_ACCUM )))"
echo "  fp16:       ${FP16}"
echo "  max_length: ${MAX_LENGTH}"
echo "  Device:     ${DEVICE}"
echo "  Quick:      ${QUICK}"
echo "  Output:     ${OUTPUT_DIR}"
echo "════════════════════════════════════════════════════════════"

# Build fp16 flag string
FP16_FLAG=""
if $FP16; then FP16_FLAG="--fp16"; fi

N_COMBOS=$(( ${#PHASES[@]} * ${#BACKBONES[@]} * ${#BENCHMARKS[@]} ))
echo "  Total combos: ${N_COMBOS}"
echo ""

COMBO=0
FAILED=0

# ═══════════════════════════════════════════════════════════════════════
# Main loop: phase × backbone × benchmark
# ═══════════════════════════════════════════════════════════════════════
for PHASE_NAME in "${PHASES[@]}"; do
  for BB in "${BACKBONES[@]}"; do
    for BM in "${BENCHMARKS[@]}"; do
      COMBO=$((COMBO + 1))
      TASK="$(get_single_task "$BM")"

      echo ""
      echo "──────────────────────────────────────────────────────────"
      echo "  [${COMBO}/${N_COMBOS}] Phase=${PHASE_NAME}  Backbone=$(basename "$BB")  Benchmark=${BM}"
      echo "──────────────────────────────────────────────────────────"

      case "$PHASE_NAME" in

        # ─── T1: Proxy Validation ───
        t1)
          for LAYER in $T1_LAYERS; do
            echo "  → T1: Layer ${LAYER}"
            python3 exp_t1_proxy_validation.py \
              --model_name "$BB" \
              --data_dir "$DATA_DIR" \
              --task "$TASK" \
              --benchmark "$BM" \
              --n_batches "$T1_N_BATCHES" \
              --batch_size "$BATCH_SIZE" \
              --target_layer "$LAYER" \
              --device "$DEVICE" \
              --output_dir "$OUTPUT_DIR" \
              || { echo "  ✗ FAILED: T1 $(basename "$BB") ${BM} layer${LAYER}"; FAILED=$((FAILED+1)); }
          done
          ;;

        # ─── T2: GGI Init Training ───
        t2)
          for T in $(get_t2_tasks "$BM"); do
            echo "  → T2: Task=${T}"
            python3 exp_t2_ggi_init_training.py \
              --model_name "$BB" \
              --data_dir "$DATA_DIR" \
              --task "$T" \
              --benchmark "$BM" \
              --lora_r "$LORA_R" \
              --n_epochs "$N_EPOCHS" \
              --batch_size "$BATCH_SIZE" \
              --grad_accum "$GRAD_ACCUM" \
              --max_length "$MAX_LENGTH" \
              --lr "$LR" \
              --n_probe_batches "$T2_PROBE_BATCHES" \
              --device "$DEVICE" \
              --output_dir "$OUTPUT_DIR" \
              $FP16_FLAG \
              || { echo "  ✗ FAILED: T2 $(basename "$BB") ${BM} ${T}"; FAILED=$((FAILED+1)); }
          done
          ;;

        # ─── T3: SGR vs Hard CL Sequence ───
        t3)
          CL_TASKS="$(get_t3_tasks "$BM")"
          echo "  → T3: CL sequence=${CL_TASKS}"
          QUICK_FLAG_T3=""
          if $QUICK; then QUICK_FLAG_T3="--quick"; fi
          python3 exp_t3_sgr_vs_hard_training.py \
            --model_name "$BB" \
            --data_dir "$DATA_DIR" \
            --benchmark "$BM" \
            --tasks "$CL_TASKS" \
            --lora_r "$LORA_R" \
            --n_epochs "$N_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --lr "$LR" \
            --sgr_lambdas "$T3_SGR_LAMBDAS" \
            --soft_init_strength "$T3_SOFT_INIT" \
            --device "$DEVICE" \
            --output_dir "$OUTPUT_DIR" \
            $FP16_FLAG \
            $QUICK_FLAG_T3 \
            || { echo "  ✗ FAILED: T3 $(basename "$BB") ${BM}"; FAILED=$((FAILED+1)); }
          ;;

        # ─── E0: TARA Rank Sweep ───
        e0)
          echo "  → E0: Task=${TASK}, Ranks=${E0_RANKS}"
          python3 exp_e0_tara_rank_sweep.py \
            --model_name "$BB" \
            --data_dir "$DATA_DIR" \
            --task "$TASK" \
            --benchmark "$BM" \
            --ranks "$E0_RANKS" \
            --n_epochs "$N_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --lr "$LR" \
            --n_probe_batches "$T2_PROBE_BATCHES" \
            --device "$DEVICE" \
            --output_dir "$OUTPUT_DIR" \
            $FP16_FLAG \
            || { echo "  ✗ FAILED: E0 $(basename "$BB") ${BM}"; FAILED=$((FAILED+1)); }
          ;;

        # ─── E3: BNG vs AdamW ───
        e3)
          echo "  → E3: Task=${TASK}"
          python3 exp_e3_bng_vs_adamw.py \
            --model_name "$BB" \
            --data_dir "$DATA_DIR" \
            --task "$TASK" \
            --benchmark "$BM" \
            --lora_r "$LORA_R" \
            --n_epochs "$E3_N_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --lr "$LR" \
            --lr_ratio "$E3_LR_RATIO" \
            --beta_ema "$E3_BETA_EMA" \
            --lambda_bal "$E3_LAMBDA_BAL" \
            --n_probe_batches "$T2_PROBE_BATCHES" \
            --device "$DEVICE" \
            --output_dir "$OUTPUT_DIR" \
            $FP16_FLAG \
            || { echo "  ✗ FAILED: E3 $(basename "$BB") ${BM}"; FAILED=$((FAILED+1)); }
          ;;

        # ─── E4: Full GALA vs Baselines (capstone) ───
        e4)
          CL_TASKS="$(get_t3_tasks "$BM")"
          echo "  → E4: Full GALA pipeline, tasks=${CL_TASKS}"
          QUICK_FLAG_E4=""
          if $QUICK; then QUICK_FLAG_E4="--quick"; fi
          python3 exp_e4_full_gala.py \
            --model_name "$BB" \
            --data_dir "$DATA_DIR" \
            --benchmark "$BM" \
            --tasks "$CL_TASKS" \
            --lora_r "$LORA_R" \
            --n_epochs "$N_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --grad_accum "$GRAD_ACCUM" \
            --max_length "$MAX_LENGTH" \
            --lr "$LR" \
            --sgr_lambda 0.1 \
            --soft_init_strength 0.7 \
            --device "$DEVICE" \
            --output_dir "$OUTPUT_DIR" \
            $FP16_FLAG \
            $QUICK_FLAG_E4 \
            || { echo "  ✗ FAILED: E4 $(basename "$BB") ${BM}"; FAILED=$((FAILED+1)); }
          ;;

        *)
          echo "  Unknown phase: ${PHASE_NAME}"
          ;;
      esac

    done
  done
done

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
if [[ $FAILED -eq 0 ]]; then
  echo "  ✓ All ${COMBO} combos completed successfully!"
else
  echo "  Done: ${COMBO} combos, ${FAILED} FAILED."
fi
echo "  Results in ${OUTPUT_DIR}/"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Result files:"
echo "    ls ${OUTPUT_DIR}/t1_proxy_*.json       # Proxy validation"
echo "    ls ${OUTPUT_DIR}/t2_ggi_init_*.json    # GGI init training"
echo "    ls ${OUTPUT_DIR}/t3_sgr_vs_hard_*.json # SGR vs Hard CL"
echo "    ls ${OUTPUT_DIR}/e0_tara_rank_*.json   # TARA rank sweep"
echo "    ls ${OUTPUT_DIR}/e3_bng_vs_adamw_*.json   # BNG vs AdamW
    ls ${OUTPUT_DIR}/e4_full_gala_*.json      # Full GALA vs baselines"
