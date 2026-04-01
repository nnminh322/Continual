#!/usr/bin/env bash
# run_gala_validation.sh — Run GALA (Contribution 2) empirical validation.
#
# Runs all 4 phases across all backbone × benchmark combinations.
# Uses same embeddings directory as Contribution 1.
#
# Usage:
#   bash run_gala_validation.sh                              # all combos
#   bash run_gala_validation.sh --phase 1                    # TARA only
#   bash run_gala_validation.sh --backbone T5EncoderModel    # one backbone
#   bash run_gala_validation.sh --whiten                     # whitened space
#   bash run_gala_validation.sh --phase 2 --rank 16          # GGI at rank=16
#
# Individual phase runs:
#   bash run_gala_validation.sh --phase 1   # TARA: rank allocation
#   bash run_gala_validation.sh --phase 2   # GGI: init strategy
#   bash run_gala_validation.sh --phase 3   # SGR: hard vs soft projection
#   bash run_gala_validation.sh --phase 4   # BNG: preconditioning

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Defaults
PHASE=0
BACKBONE=""
BENCHMARK=""
WHITEN=false
RANK=8
DEVICE="auto"
N_TRIALS=50

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "  --phase N       Run only phase N (1=TARA, 2=GGI, 3=SGR, 4=BNG). 0=all."
  echo "  --backbone B    Run only this backbone (T5EncoderModel|LlamaForCausalLM)"
  echo "  --benchmark B   Run only this benchmark (Long_Sequence|SuperNI)"
  echo "  --whiten        Apply ZCA whitening"
  echo "  --rank R        Subspace rank (default: 8)"
  echo "  --device D      auto|cuda|cpu|mps (default: auto)"
  echo "  --n_trials N    Random trials for GGI (default: 50)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --phase)     PHASE="$2"; shift 2 ;;
    --backbone)  BACKBONE="$2"; shift 2 ;;
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --whiten)    WHITEN=true; shift ;;
    --rank)      RANK="$2"; shift 2 ;;
    --device)    DEVICE="$2"; shift 2 ;;
    --n_trials)  N_TRIALS="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *)           echo "Unknown option: $1"; usage ;;
  esac
done

# Build backbone and benchmark lists
if [[ -n "$BACKBONE" ]]; then
  BACKBONES=("$BACKBONE")
else
  BACKBONES=(T5EncoderModel LlamaForCausalLM)
fi

if [[ -n "$BENCHMARK" ]]; then
  BENCHMARKS=("$BENCHMARK")
else
  BENCHMARKS=(Long_Sequence SuperNI)
fi

echo "============================================================"
echo "GALA (Contribution 2) — Empirical Validation"
echo "============================================================"
echo "  Phase:     ${PHASE} (0=all)"
echo "  Backbones: ${BACKBONES[*]}"
echo "  Benchmarks: ${BENCHMARKS[*]}"
echo "  Whitened:  ${WHITEN}"
echo "  Rank:      ${RANK}"
echo "  Device:    ${DEVICE}"
echo "============================================================"

WHITEN_FLAG=""
if $WHITEN; then
  WHITEN_FLAG="--whiten"
fi

for BB in "${BACKBONES[@]}"; do
  for BM in "${BENCHMARKS[@]}"; do
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  Running: ${BB} × ${BM} ${WHITEN_FLAG}"
    echo "──────────────────────────────────────────────"

    EMB_DIR="embeddings/${BB}"
    if [[ ! -d "$EMB_DIR" ]]; then
      echo "  SKIP: ${EMB_DIR} not found"
      continue
    fi

    python3 validate_gala.py \
      --emb_dir "$EMB_DIR" \
      --benchmark "$BM" \
      --phase "$PHASE" \
      --rank "$RANK" \
      --device "$DEVICE" \
      --n_trials "$N_TRIALS" \
      $WHITEN_FLAG \
      || echo "  FAILED: ${BB} × ${BM}"
  done
done

# Also run whitened variants if not already whitened
if ! $WHITEN; then
  echo ""
  echo "============================================================"
  echo "  Re-running with --whiten for comparison"
  echo "============================================================"
  for BB in "${BACKBONES[@]}"; do
    for BM in "${BENCHMARKS[@]}"; do
      EMB_DIR="embeddings/${BB}"
      if [[ ! -d "$EMB_DIR" ]]; then
        continue
      fi
      echo ""
      echo "──────────────────────────────────────────────"
      echo "  Running: ${BB} × ${BM} --whiten"
      echo "──────────────────────────────────────────────"
      python3 validate_gala.py \
        --emb_dir "$EMB_DIR" \
        --benchmark "$BM" \
        --phase "$PHASE" \
        --rank "$RANK" \
        --device "$DEVICE" \
        --n_trials "$N_TRIALS" \
        --whiten \
        || echo "  FAILED: ${BB} × ${BM} whitened"
    done
  done
fi

echo ""
echo "============================================================"
echo "  Done! Results in results/gala_*.json"
echo "============================================================"
