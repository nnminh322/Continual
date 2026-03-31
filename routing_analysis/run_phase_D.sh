#!/usr/bin/env bash
# Phase D — PSR Ablation
# Tests: component ablation (5 PSR configs), rank sweep (k=2..64),
#        domain breakdown, incremental simulation (PSR vs RLS_inc vs RLS_batch).
#
# Usage:
#   bash run_phase_D.sh --benchmark Long_Sequence --backbone T5EncoderModel
#   bash run_phase_D.sh --benchmark SuperNI       --backbone LlamaForCausalLM
#   bash run_phase_D.sh --benchmark Long_Sequence --compare_backbones  # T5 vs LLaMA
#
# Output: results/ablation_{backbone}_{benchmark}.json

set -euo pipefail

BENCHMARK=""
BACKBONE=""
SUBSPACE_K=8
WHITEN=false
COMPARE_BACKBONES=false
LAYER=""
DEVICE="auto"
FORCE=false

usage() {
  echo "Usage: $0 --benchmark <Long_Sequence|SuperNI> --backbone <backbone_dir_name> [--k INT] [--whiten] [--compare_backbones] [--layer embedding] [--device auto|cpu|cuda]"
  echo ""
  echo "  --benchmark         Long_Sequence or SuperNI"
  echo "  --backbone          Name of backbone subdirectory under embeddings/"
  echo "  --k                 Subspace rank (default: 8)"
  echo "  --whiten            Apply ZCA whitening"
  echo "  --compare_backbones Pass embeddings/ (parent dir) and compare all backbone subdirs"
  echo "  --layer             embedding = use _wordemb extraction dir"
  echo "  --device            cpu | cuda | auto (default: auto)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --benchmark)         BENCHMARK="$2"; shift 2 ;;
    --backbone)          BACKBONE="$2";  shift 2 ;;
    --k)                 SUBSPACE_K="$2"; shift 2 ;;
    --whiten)            WHITEN=true; shift ;;
    --compare_backbones) COMPARE_BACKBONES=true; shift ;;
    --layer)             LAYER="$2"; shift 2 ;;
    --device)            DEVICE="$2"; shift 2 ;;
    --force)             FORCE=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

[[ -z "$BENCHMARK" ]] && usage
if [[ "$COMPARE_BACKBONES" == "false" && -z "$BACKBONE" ]]; then usage; fi

cd "$(dirname "$0")"

if [[ "$COMPARE_BACKBONES" == "true" ]]; then
  EMB_DIR="embeddings"
else
  if [[ "$LAYER" == "embedding" ]]; then
    EMB_DIR="embeddings/${BACKBONE}_wordemb"
  else
    EMB_DIR="embeddings/${BACKBONE}"
  fi
fi

echo "========================================================"
echo "  Phase D — PSR Ablation"
echo "  Backbone  : ${BACKBONE:-ALL}  (${LAYER:-encoder/hidden})"
echo "  Benchmark : ${BENCHMARK}"
echo "  k         : ${SUBSPACE_K}"
echo "  Whiten    : ${WHITEN}"
echo "  Compare   : ${COMPARE_BACKBONES}"
echo "========================================================"

CMD="python ablation_psr.py \
  --emb_dir    ${EMB_DIR} \
  --benchmark  ${BENCHMARK} \
  --subspace_k ${SUBSPACE_K} \
  --device     ${DEVICE}"

[[ "$WHITEN"            == "true" ]] && CMD+=" --whiten"
[[ "$COMPARE_BACKBONES" == "true" ]] && CMD+=" --compare_backbones"
[[ "$FORCE"             == "true" ]] && CMD+=" --force"

echo "CMD: $CMD"
echo ""
eval $CMD
