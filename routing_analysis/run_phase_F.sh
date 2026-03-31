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

# Reduce CUDA memory fragmentation (helps Llama d=4096 on 16 GB GPUs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BENCHMARK=""
BACKBONE=""
SUBSPACE_K=8
WHITEN=false
LAYER=""

# GPM parameters (auto-detected from backbone if not set)
MLP_HIDDEN=""       # auto: T5=100, Llama=50
TRANSTHRESHOLD=0.995
CHUNK=""            # auto: T5=1, Llama=4
BACKBONE_TYPE="auto"  # auto-detect from backbone name
LR=1e-3
EPOCHS=30
BATCH_SIZE=256
DEVICE="auto"   # auto = cuda if available, else cpu
FORCE=false

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
  echo "  --mlp_hidden_dim   trans_input hidden dim (auto: T5=100, Llama=50)"
  echo "  --transthreshold   GPM energy threshold (default: 0.995)"
  echo "  --chunk            GPM chunking factor (auto: T5=1, Llama=4)"
  echo "  --backbone_type    t5 | llama | auto (default: auto-detect)"
  echo "  --lr               Proxy training LR (default: 1e-3)"
  echo "  --epochs           Proxy training epochs (default: 30)"
  echo "  --device           cpu | cuda | cuda:0 | auto (default: auto — uses GPU if available)"
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
    --chunk)           CHUNK="$2"; shift 2 ;;
    --backbone_type)   BACKBONE_TYPE="$2"; shift 2 ;;
    --lr)              LR="$2"; shift 2 ;;
    --epochs)          EPOCHS="$2"; shift 2 ;;
    --device)          DEVICE="$2"; shift 2 ;;
    --force)           FORCE=true; shift ;;
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
# Resolve "auto" device before printing and passing to python
if [[ "$DEVICE" == "auto" ]]; then
  DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo 'cpu')
fi

echo "  GPM: backbone_type=${BACKBONE_TYPE}, mlp_hidden=${MLP_HIDDEN:-auto}, chunk=${CHUNK:-auto}, threshold=${TRANSTHRESHOLD}, lr=${LR}, epochs=${EPOCHS}, device=${DEVICE}"
echo "  RLS: expansion=${RLS_EXPANSION}, lambda=${RLS_LAMBDA}"
echo "========================================================"

CMD="python simulate_gpm_routing.py \
  --emb_dir         ${EMB_DIR} \
  --benchmark       ${BENCHMARK} \
  --subspace_k      ${SUBSPACE_K} \
  --transthreshold  ${TRANSTHRESHOLD} \
  --backbone_type   ${BACKBONE_TYPE} \
  --lr              ${LR} \
  --epochs          ${EPOCHS} \
  --batch_size      ${BATCH_SIZE} \
  --device          ${DEVICE} \
  --rls_expansion   ${RLS_EXPANSION} \
  --rls_lambda      ${RLS_LAMBDA}"

[[ -n "$MLP_HIDDEN" ]] && CMD+=" --mlp_hidden_dim ${MLP_HIDDEN}"
[[ -n "$CHUNK" ]]      && CMD+=" --chunk ${CHUNK}"

[[ "$WHITEN" == "true" ]] && CMD+=" --whiten"
[[ "$FORCE"  == "true" ]] && CMD+=" --force"

echo "CMD: $CMD"
echo ""
eval $CMD
