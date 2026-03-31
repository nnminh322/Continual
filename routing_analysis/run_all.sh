#!/usr/bin/env bash
# run_all.sh — Run all routing geometry experiment phases sequentially.
#
# Runs Phase A → B+C → D → E → F for the given benchmark and backbone.
# Each phase's output feeds the next (all results go to results/).
#
# Usage:
#   bash run_all.sh --benchmark Long_Sequence --backbone T5EncoderModel
#   bash run_all.sh --benchmark SuperNI       --backbone LlamaForCausalLM --whiten
#   bash run_all.sh --benchmark Long_Sequence --backbone T5EncoderModel --layer embedding
#
# Skip individual phases:
#   bash run_all.sh --benchmark Long_Sequence --backbone T5EncoderModel --skip_F
#
# To run all benchmarks × all backbones:
#   for bench in Long_Sequence SuperNI; do
#     for bb in T5EncoderModel LlamaForCausalLM; do
#       bash run_all.sh --benchmark $bench --backbone $bb
#     done
#   done

set -euo pipefail

# Reduce CUDA memory fragmentation (helps Llama d=4096 on 16 GB GPUs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BENCHMARK=""
BACKBONE=""
SUBSPACE_K=8
WHITEN=false
LAYER=""
SKIP_F=false        # Phase F (GPM) requires torch; set --skip_F to skip if unavailable
DEVICE="auto"       # auto = cuda if available, else cpu (used by Phase F)
FORCE=false         # --force = re-run even if output already exists

usage() {
  echo "Usage: $0 --benchmark <Long_Sequence|SuperNI> --backbone <backbone_dir_name> [OPTIONS]"
  echo ""
  echo "  --benchmark   Long_Sequence or SuperNI"
  echo "  --backbone    Name of backbone subdirectory under embeddings/"
  echo "  --k           Subspace rank (default: 8)"
  echo "  --whiten      Apply ZCA whitening to all phases"
  echo "  --layer       embedding = use word-embedding-layer extractions (_wordemb suffix)"
  echo "  --skip_F      Skip Phase F (requires PyTorch)"
  echo "  --device      cpu | cuda | cuda:0 | auto (default: auto, passed to all phases)"
  echo "  --force       Force re-run even if output already exists"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --backbone)  BACKBONE="$2";  shift 2 ;;
    --k)         SUBSPACE_K="$2"; shift 2 ;;
    --whiten)    WHITEN=true; shift ;;
    --layer)     LAYER="$2"; shift 2 ;;
    --skip_F)    SKIP_F=true; shift ;;
    --device)    DEVICE="$2"; shift 2 ;;
    --force)     FORCE=true; shift ;;
    -h|--help)   usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

[[ -z "$BENCHMARK" || -z "$BACKBONE" ]] && usage

SCRIPT_DIR="$(dirname "$0")"
COMMON="--benchmark ${BENCHMARK} --backbone ${BACKBONE} --k ${SUBSPACE_K} --device ${DEVICE}"
[[ "$WHITEN" == "true" ]] && COMMON+=" --whiten"
[[ -n "$LAYER" ]] && COMMON+=" --layer ${LAYER}"
[[ "$FORCE" == "true" ]] && COMMON+=" --force"

echo "###################################################"
echo "  Routing Geometry — Full Experiment Pipeline"
echo "  Backbone  : ${BACKBONE}  (layer: ${LAYER:-encoder/hidden})"
echo "  Benchmark : ${BENCHMARK}"
echo "  k         : ${SUBSPACE_K}"
echo "  Whiten    : ${WHITEN}"
echo "  Skip F    : ${SKIP_F}"
echo "  Force     : ${FORCE}"
echo "  Device    : ${DEVICE}"
echo "###################################################"

# ── Phase A: Geometric EDA ──────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  PHASE A: Geometric EDA"
echo "══════════════════════════════════════════════════"
bash "${SCRIPT_DIR}/run_phase_A.sh" ${COMMON}

# ── Phase B+C: Distance Metrics & Classifiers ──────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  PHASE B+C: Distance Metrics & Classifiers"
echo "══════════════════════════════════════════════════"
bash "${SCRIPT_DIR}/run_phase_BC.sh" ${COMMON}

# ── Phase D: PSR Ablation ──────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  PHASE D: PSR Ablation"
echo "══════════════════════════════════════════════════"
bash "${SCRIPT_DIR}/run_phase_D.sh" ${COMMON}

# ── Phase E: Theory Validation ─────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  PHASE E: Theory Validation"
echo "══════════════════════════════════════════════════"
bash "${SCRIPT_DIR}/run_phase_E.sh" ${COMMON}

# ── Phase F: Learned Routing (GPM/RLS) ────────────────
if [[ "$SKIP_F" == "false" ]]; then
  echo ""
  echo "══════════════════════════════════════════════════"
  echo "  PHASE F: Learned Routing (GPM vs RLS)"
  echo "══════════════════════════════════════════════════"
  bash "${SCRIPT_DIR}/run_phase_F.sh" ${COMMON} --device ${DEVICE}
else
  echo ""
  echo "[SKIP] Phase F — skipped (--skip_F)"
fi

echo ""
echo "###################################################"
echo "  All phases complete! Results in: ${SCRIPT_DIR}/results/"
echo "###################################################"
