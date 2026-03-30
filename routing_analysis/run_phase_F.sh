#!/usr/bin/env bash
# Phase F — Learned Routing (GPM/ROOT vs RLS/SpecRoute vs baselines)
# Runs incremental simulation: tasks are added one by one,
# accuracy reported at each step for GPM_ROOT, RLS_Woodbury, PSR,
# NearestCentroid, CosineNearestCentroid.
#
# Usage:
#   bash run_phase_F.sh --benchmark Long_Sequence --backbone T5EncoderModel
#   bash run_phase_F.sh --benchmark SuperNI       --backbone LlamaForCausalLM
#
# GPM/ROOT parameters (match training script defaults):
#   --mlp_hidden_dim  100      (from --mlp_hidden_dim 100 in gen_scripts)
#   --transthreshold  0.995    (from --transthreshold 0.995 in gen_scripts)
#
# Output: results/learned_routing_{backbone}_{benchmark}.json

set -euo pipefail

BENCHMARK=""
BACKBONE=""
SUBSPACE_K=8
WHITEN=false
LAYER=""

# GPM parameters
MLP_HIDDEN=100
TRANSTHRESHOLD=0.995
LR=1e-3
EPOCHS=30
BATCH_SIZE=256
DEVICE="cpu"

# RLS parameters
RLS_EXPANSION=2048
RLS_LAMBDA=0.1

usage() {
  echo "Usage: $0 --benchmark <Long_Sequence|SuperNI> --backbone <backbone_dir_name> [OPTIONS]"
  echo ""
  echo "  --benchmark        Long_Sequence or SuperNI"
  echo "  --backbone         Name of backbone subdirectory under embeddings/"
  echo "  --k                PSR subspace rank (default: 8)"
  echo "  --whiten           Apply ZCA whitening"
  echo "  --layer            embedding = use _wordemb extraction dir"
  echo ""
  echo "  GPM/ROOT options:"
  echo "  --mlp_hidden_dim   trans_input MLP hidden dim (default: 100)"
  echo "  --transthreshold   GPM energy threshold (default: 0.995)"
  echo "  --lr               Proxy training LR (default: 1e-3)"
  echo "  --epochs           Proxy training epochs (default: 30)"
  echo "  --device           cpu or cuda (default: cpu)"
  echo ""
  echo "  RLS options:"
  echo "  --rls_expansion    Random feature dim (default: 2048)"
  echo "  --rls_lambda       Ridge lambda (default: 0.1)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --benchmark)       BENCHMARK="$2"; shift 2 ;;
    --backbone)        BACKBONE="$2";  shift 2 ;;
    --k)               SUBSPACE_K="$2"; shift 2 ;;
    --whiten)          WHITEN=true; shift ;;
    --layer)           LAYER="$2"; shift 2 ;;
    --mlp_hidden_dim)  MLP_HIDDEN="$2"; shift 2 ;;
    --transthreshold)  TRANSTHRESHOLD="$2"; shift 2 ;;
    --lr)              LR="$2"; shift 2 ;;
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --device)          DEVICE="$2"; shift 2 ;;
    --rls_expansion)   RLS_EXPANSION="$2"; shift 2 ;;
    --rls_lambda)      RLS_LAMBDA="$2"; shift 2 ;;
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
echo "  Phase F — Learned Routing Comparison"
echo "  Backbone  : ${BACKBONE}  (${LAYER:-encoder/hidden})"
echo "  Benchmark : ${BENCHMARK}"
echo "  k (PSR)   : ${SUBSPACE_K}"
echo "  Whiten    : ${WHITEN}"
echo "  GPM: mlp_hidden=${MLP_HIDDEN}, threshold=${TRANSTHRESHOLD}, lr=${LR}, epochs=${EPOCHS}, device=${DEVICE}"
echo "  RLS: expansion=${RLS_EXPANSION}, lambda=${RLS_LAMBDA}"
echo "========================================================"

CMD="python simulate_gpm_routing.py \
  --emb_dir         ${EMB_DIR} \
  --benchmark       ${BENCHMARK} \
  --subspace_k      ${SUBSPACE_K} \
  --mlp_hidden_dim  ${MLP_HIDDEN} \
  --transthreshold  ${TRANSTHRESHOLD} \
  --lr              ${LR} \
  --epochs          ${EPOCHS} \
  --batch_size      ${BATCH_SIZE} \
  --device          ${DEVICE} \
  --rls_expansion   ${RLS_EXPANSION} \
  --rls_lambda      ${RLS_LAMBDA}"

[[ "$WHITEN" == "true" ]] && CMD+=" --whiten"

echo "CMD: $CMD"
echo ""
eval $CMD
