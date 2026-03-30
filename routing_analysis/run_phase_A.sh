#!/usr/bin/env bash
# Phase A — Geometric EDA
# Checks: effective dimensionality, Gaussianity, anisotropy, centroid distances,
#         Grassmannian alignment, few-shot stability, multi-modality (GMM BIC).
#
# Usage:
#   bash run_phase_A.sh --benchmark Long_Sequence --backbone T5EncoderModel
#   bash run_phase_A.sh --benchmark SuperNI       --backbone LlamaForCausalLM
#
# Output: results/geometry_{backbone}_{benchmark}.json

set -euo pipefail

BENCHMARK=""
BACKBONE=""
SUBSPACE_K=8
WHITEN=false
LAYER=""    # optional: "" = default (encoder/hidden), "embedding" = word embedding layer

usage() {
  echo "Usage: $0 --benchmark <Long_Sequence|SuperNI> --backbone <backbone_dir_name> [--k INT] [--whiten] [--layer embedding]"
  echo ""
  echo "  --benchmark   Long_Sequence or SuperNI"
  echo "  --backbone    Name of the backbone subdirectory under embeddings/"
  echo "                e.g. T5EncoderModel, flan-t5-large, LlamaForCausalLM"
  echo "  --k           Subspace rank (default: 8)"
  echo "  --whiten      Apply ZCA whitening"
  echo "  --layer       embedding = use word-embedding-layer extraction (adds _wordemb suffix)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --backbone)  BACKBONE="$2";  shift 2 ;;
    --k)         SUBSPACE_K="$2"; shift 2 ;;
    --whiten)    WHITEN=true; shift ;;
    --layer)     LAYER="$2"; shift 2 ;;
    -h|--help)   usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

[[ -z "$BENCHMARK" || -z "$BACKBONE" ]] && usage

# Appply word-embedding suffix if layer=embedding
if [[ "$LAYER" == "embedding" ]]; then
  EMB_DIR="embeddings/${BACKBONE}_wordemb"
else
  EMB_DIR="embeddings/${BACKBONE}"
fi

# Navigate to the script directory
cd "$(dirname "$0")"

echo "========================================================"
echo "  Phase A — Geometric EDA"
echo "  Backbone  : ${BACKBONE}  (${LAYER:-encoder/hidden})"
echo "  Benchmark : ${BENCHMARK}"
echo "  k         : ${SUBSPACE_K}"
echo "  Whiten    : ${WHITEN}"
echo "========================================================"

CMD="python analyze_geometry.py \
  --emb_dir  ${EMB_DIR} \
  --benchmark ${BENCHMARK} \
  --subspace_k ${SUBSPACE_K}"

[[ "$WHITEN" == "true" ]] && CMD+=" --whiten"

echo "CMD: $CMD"
echo ""
eval $CMD
