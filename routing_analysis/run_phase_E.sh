#!/usr/bin/env bash
# Phase E — Theory Validation
# Tests: KL decomposition vs confusion (E1), Grassmann packing bound (E2),
#        Random Matrix Theory / Marchenko-Pastur (E3), shrinkage routing (E3b).
#
# Usage:
#   bash run_phase_E.sh --benchmark Long_Sequence --backbone T5EncoderModel
#   bash run_phase_E.sh --benchmark SuperNI       --backbone LlamaForCausalLM
#
# Output: results/theory_{backbone}_{benchmark}.json
#         results/kl_matrix_{tag}.npy
#         results/overlap_matrix_{tag}.npy

set -euo pipefail

BENCHMARK=""
BACKBONE=""
SUBSPACE_K=8
WHITEN=false
LAYER=""
DEVICE="auto"

usage() {
  echo "Usage: $0 --benchmark <Long_Sequence|SuperNI> --backbone <backbone_dir_name> [--k INT] [--whiten] [--layer embedding] [--device auto|cpu|cuda]"
  echo ""
  echo "  --benchmark  Long_Sequence or SuperNI"
  echo "  --backbone   Name of backbone subdirectory under embeddings/"
  echo "  --k          Subspace rank (default: 8)"
  echo "  --whiten     Apply ZCA whitening"
  echo "  --layer      embedding = use _wordemb extraction dir"
  echo "  --device     cpu | cuda | auto (default: auto)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --backbone)  BACKBONE="$2";  shift 2 ;;
    --k)         SUBSPACE_K="$2"; shift 2 ;;
    --whiten)    WHITEN=true; shift ;;
    --layer)     LAYER="$2"; shift 2 ;;
    --device)    DEVICE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

[[ -z "$BENCHMARK" || -z "$BACKBONE" ]] && usage

if [[ "$LAYER" == "embedding" ]]; then
  EMB_DIR="embeddings/${BACKBONE}_wordemb"
else
  EMB_DIR="embeddings/${BACKBONE}"
fi

cd "$(dirname "$0")"

echo "========================================================"
echo "  Phase E — Theory Validation"
echo "  Backbone  : ${BACKBONE}  (${LAYER:-encoder/hidden})"
echo "  Benchmark : ${BENCHMARK}"
echo "  k         : ${SUBSPACE_K}"
echo "  Whiten    : ${WHITEN}"
echo "========================================================"

CMD="python validate_theory.py \
  --emb_dir    ${EMB_DIR} \
  --benchmark  ${BENCHMARK} \
  --subspace_k ${SUBSPACE_K} \
  --device     ${DEVICE}"

[[ "$WHITEN" == "true" ]] && CMD+=" --whiten"

echo "CMD: $CMD"
echo ""
eval $CMD
