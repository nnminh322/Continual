#!/usr/bin/env bash
# Phase B+C — Distance Metrics & Classifier Comparison
# Phase B: L2, Cosine, NormL2, SpectralAffinity, SubspaceResidual,
#          WeightedSpectral, PSR_full/no_mean/no_subspace/no_penalty, Mahalanobis
# Phase C: LDA, QDA, LinearSVM, RidgeClassifier (batch oracle)
#
# Usage:
#   bash run_phase_BC.sh --benchmark Long_Sequence --backbone T5EncoderModel
#   bash run_phase_BC.sh --benchmark SuperNI       --backbone LlamaForCausalLM --skip_sklearn
#
# Output: results/routing_{backbone}_{benchmark}.json
#         results/confusion_{backbone}_{benchmark}_{method}.npy

set -euo pipefail

BENCHMARK=""
BACKBONE=""
SUBSPACE_K=8
WHITEN=false
SKIP_SKLEARN=false
LAYER=""
DEVICE="auto"
FORCE=false

usage() {
  echo "Usage: $0 --benchmark <Long_Sequence|SuperNI> --backbone <backbone_dir_name> [--k INT] [--whiten] [--skip_sklearn] [--layer embedding] [--device auto|cpu|cuda]"
  echo ""
  echo "  --benchmark    Long_Sequence or SuperNI"
  echo "  --backbone     Name of backbone subdirectory under embeddings/"
  echo "  --k            Subspace rank (default: 8)"
  echo "  --whiten       Apply ZCA whitening"
  echo "  --skip_sklearn Skip Phase C (sklearn classifiers). Useful for quick distance sweep."
  echo "  --layer        embedding = use _wordemb extraction dir"
  echo "  --device       cpu | cuda | auto (default: auto)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --benchmark)   BENCHMARK="$2"; shift 2 ;;
    --backbone)    BACKBONE="$2";  shift 2 ;;
    --k)           SUBSPACE_K="$2"; shift 2 ;;
    --whiten)      WHITEN=true; shift ;;
    --skip_sklearn) SKIP_SKLEARN=true; shift ;;
    --layer)       LAYER="$2"; shift 2 ;;
    --device)      DEVICE="$2"; shift 2 ;;
    --force)       FORCE=true; shift ;;
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
echo "  Phase B+C — Distance Metrics & Classifiers"
echo "  Backbone  : ${BACKBONE}  (${LAYER:-encoder/hidden})"
echo "  Benchmark : ${BENCHMARK}"
echo "  k         : ${SUBSPACE_K}"
echo "  Whiten    : ${WHITEN}"
echo "  Skip C    : ${SKIP_SKLEARN}"
echo "========================================================"

CMD="python compare_routing.py \
  --emb_dir    ${EMB_DIR} \
  --benchmark  ${BENCHMARK} \
  --subspace_k ${SUBSPACE_K} \
  --device     ${DEVICE}"

[[ "$WHITEN"       == "true" ]] && CMD+=" --whiten"
[[ "$SKIP_SKLEARN" == "true" ]] && CMD+=" --skip_sklearn"
[[ "$FORCE"        == "true" ]] && CMD+=" --force"

echo "CMD: $CMD"
echo ""
eval $CMD
