"""
contri2_utils.py — Core utilities for CONTRI2: Isolated SGWI Tests.

Exports:
  TASK_ORDER     : Long_Sequence task order (15 tasks)
  RANDOM_INIT    : constant = "random"
  NTI_INIT       : constant = "nti"
  SVD_FUSION_INIT: constant = "sfi"
  RandomAcc      : dict of random-guess baselines per task
  load_task_data : load train/test JSON
  extract_frozen_embeddings : T5 frozen encoder forward pass
  build_srt_router : build SRTRouter from extracted embeddings
  build_model   : load T5 with/without LoRA adapters
  init_lora_weights : apply RANDOM / NTI / SFI init to model
  eval_zero_shot    : zero-shot evaluation
  evaluate_model    : full evaluation with metric
  train_lora_isolated : few-shot isolated training loop

────────────────────────────────────────────────────────────────
 LoRA Architecture (from t5_gainlora_inflora.py):
   LoRALayer:  lora_A ∈ ℝ^{r×d},  lora_B ∈ ℝ^{d×r}
               ΔW = B @ A  ∈ ℝ^{d×d}
               rank(A)=rank(B)=r (LoRA rank)

 LoRA Init in original code (t5_gainlora_inflora.py, LoRALayer.reset_parameters):
   nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))   # Random A
   nn.init.zeros_(lora_B)                               # Zero B
   → ΔW_init = 0

────────────────────────────────────────────────────────────────
 SRT-Guided Warm Initialization (SGWI):

 NTI (Nearest-Task Init):
   s* = argmin_{s<t} d_SRT(t, s)
   B_t ← B_{s*} · sqrt(λ_max)
   A_t ← sqrt(λ_max) · A_{s*}
   (keeps the same subspace, rescales to rank-r)

 SFI (SVD Fusion Init — recommended):
   1. w_s = softmax(-d_SRT(t,s) / τ)   [τ = median pairwise distance]
   2. ΔW_init = Σ_s w_s · B_s · A_s   (weight-space weighted blend)
   3. U, S, Vᵀ = rank-r SVD(ΔW_init)
   4. B_t = U[:, :r] · √S[:r]
   5. A_t = √S[:r] · V[:r, :]
────────────────────────────────────────────────────────────────
"""

import json
import math
import os
import random
import sys
import time
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    T5EncoderModel,
)

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
PARENT = ROOT.parent
SRC    = PARENT / "src"
sys.path.insert(0, str(SRC))

# ── Transformers Version Compatibility ─────────────────────────────────────────
# t5_gainlora_inflora.py imports modules removed/moved in newer transformers.
# MUST be patched BEFORE any import of t5_gainlora_inflora.
# Applies to: transformers >= 4.30 (model_parallel_utils removed),
#             transformers >= 4.40 (find_pruneable_heads_and_indices moved).
import types as _types
import transformers.utils

# Fix 1: model_parallel_utils (removed in transformers >= 4.40)
if not hasattr(transformers.utils, 'model_parallel_utils'):
    _mpu = _types.ModuleType('transformers.utils.model_parallel_utils')
    _mpu.assert_device_map = lambda *a, **k: None
    _mpu.get_device_map = lambda *a, **k: {}
    sys.modules['transformers.utils.model_parallel_utils'] = _mpu
    transformers.utils.model_parallel_utils = _mpu

# Fix 2: find_pruneable_heads_and_indices (moved in newer transformers)
import transformers.pytorch_utils as _pt_utils
if not hasattr(_pt_utils, 'find_pruneable_heads_and_indices'):
    def _find_ph_keys(*a, **k): return set(), None
    _pt_utils.find_pruneable_heads_and_indices = _find_ph_keys

# Fix 3: loralib (optional dependency — only used if present in t5_gainlora_inflora)
# t5_gainlora_inflora.py does "import loralib as lora" but loralib may not be installed.
# We provide a minimal stub so the import succeeds.
if 'loralib' not in sys.modules:
    _loralib = _types.ModuleType('loralib')
    # Provide minimal LoRALayer stub — not actually used in our code path
    class _LoRALayerStub(nn.Module):
        def __init__(self, *a, **k): super().__init__()
        def forward(self, x): return x
    _loralib.LoRALayer = _LoRALayerStub
    sys.modules['loralib'] = _loralib

from srt_router import SRTRouter, TaskSignature


# ── Constants ─────────────────────────────────────────────────────────────────

# Long_Sequence task orders
TASK_ORDERS = {
    3: ["yelp", "amazon", "mnli", "cb", "copa",
        "qqp",  "rte",   "imdb", "sst2", "dbpedia",
        "agnews","yahoo", "multirc", "boolq", "wic"],
    4: ["mnli", "cb", "wic", "copa", "qqp",
        "rte",  "imdb", "sst2", "dbpedia", "agnews",
        "yahoo","multirc", "boolq", "yelp", "amazon"],
}
TASK_ORDER = TASK_ORDERS[3]  # default

# Benchmark data directory
BENCHMARK_DIR = PARENT / "CL_Benchmark" / "Long_Sequence"

# Results directory
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Init mode constants
RANDOM_INIT    = "random"
NTI_INIT       = "nti"
SVD_FUSION_INIT = "sfi"

# Random-guess baselines per task (based on number of labels)
# These are approximate — computed from the actual label distributions
# in the test sets (or estimated from label count)
# Random-guess baselines per task (1 / n_classes).
# Extracted from actual CL_Benchmark/Long_Sequence/{task}/train.json label sets.
RandomAcc: Dict[str, float] = {
    # Sentiment (binary): 2 classes → 50%
    "yelp":    20.00,   # 5 classes (negative/neutral/positive/very negative/very positive)
    "amazon":  20.00,  # 5 classes
    "imdb":    50.00,  # 2 classes (Bad/Good)
    "sst2":    50.00,  # 2 classes (Bad/Good)
    # Multi-class topic: 4-14 classes
    "agnews":  25.00,  # 4 classes
    "yahoo":   10.00,  # 10 classes
    "dbpedia":  7.14,  # 14 classes (distinctive — very hard to guess randomly)
    # NLI (3-class): ~33%
    "mnli":    33.33,  # 3 classes (contradiction/entailment/neutral)
    "cb":      33.33,  # 3 classes
    "rte":     50.00,  # 2 classes (contradiction/entailment)
    # QA / Others (binary): 50%
    "copa":    50.00,  # 2 choices (A/B)
    "qqp":     50.00,  # 2 classes (yes/no)
    "wic":     50.00,  # 2 classes (False/True)
    "boolq":   50.00,  # 2 classes (False/True)
    "multirc":  50.00,  # 2 classes (False/True) — multi-answer but binary per instance
}


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_task_data(task_name: str, split: str = "train") -> List[Dict]:
    """
    Load one task's data from CL_Benchmark/Long_Sequence/{task}/{split}.json.

    Returns:
        List of dicts with keys: 'input', 'output', 'label', 'id'
        or None if file not found.
    """
    path = BENCHMARK_DIR / task_name / f"{split}.json"
    if not path.exists():
        print(f"    WARNING: {path} not found")
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    instances = data.get("Instances", [])
    samples = []
    for inst in instances:
        output = inst.get("output", "")
        if isinstance(output, list):
            output = output[0] if output else ""
        samples.append({
            "id":     inst.get("id", str(len(samples))),
            "input":  inst.get("input", ""),
            "output": str(output).strip(),
            "label":  str(output).strip(),
        })
    return samples


def format_example(sample: Dict, add_prefix: bool = True) -> str:
    """Format a sample as input string for T5."""
    instruction = ""
    if add_prefix:
        instruction = f"Input: {sample['input'].strip()}\nOutput: "
    else:
        instruction = f"{sample['input'].strip()}\nOutput: "
    return instruction, sample.get("label", sample.get("output", ""))


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING EXTRACTION (frozen T5 backbone)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frozen_embeddings(
    model_name: str,
    train_data: List[Dict],
    max_samples: int = 500,
    cache_path: Optional[str] = None,
    batch_size: int = 8,
) -> Tuple[torch.Tensor, List]:
    """
    Extract frozen backbone embeddings for SRT signatures {μ_t, Σ_t}.

    Uses T5EncoderModel.from_pretrained as frozen encoder.
    Matches routing_analysis/extract_embeddings_t5.py:
      - Layer: last encoder hidden state
      - Pool:  mean over non-padding tokens

    Args:
        model_name : "google/flan-t5-small" or "google/flan-t5-base"
        train_data : list of sample dicts
        max_samples: max training samples to use
        cache_path  : if given, save/load embeddings to this path

    Returns:
        embeddings: (n, d) tensor, mean-pooled over tokens
        labels   : list of label strings
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        embeddings = torch.from_numpy(data["embeddings"])
        labels     = list(data["labels"])
        print(f"    [CACHE HIT] {cache_path}: {embeddings.shape}")
        return embeddings, labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen encoder
    print(f"    Loading frozen encoder: {model_name}...")
    encoder = T5EncoderModel.from_pretrained(model_name)
    encoder.eval()
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Format inputs
    texts = [f"{s['input'].strip()}\nOutput:" for s in train_data[:max_samples]]
    labels = [s.get("label", s.get("output", "")) for s in train_data[:max_samples]]

    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = encoder(**inputs)
            hidden = out.last_hidden_state   # (B, L, d)

            # Mean pooling over non-padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)   # (B, d)

            embeddings_list.append(pooled.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)   # (n, d)

    # Cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings.numpy(),
            labels=np.array(labels, dtype=object),
        )
        print(f"    Saved embeddings: {embeddings.shape} → {cache_path}")

    # Free GPU memory
    del encoder, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return embeddings, labels


# ─────────────────────────────────────────────────────────────────────────────
#  SRT ROUTER BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_srt_router(
    task_list: List[str],
    router_state: Dict,
    model_name: str = "google/flan-t5-large",
    metric_mode: str = "hard",
    shrink_factor: float = 0.1,
) -> SRTRouter:
    """
    Build an SRT router from pre-extracted embeddings.

    Args:
        task_list   : list of task names (e.g. ["yelp", "amazon", "mnli"])
        router_state: dict keyed by task_name, each containing:
                       {"emb_path": str, "train_data": list or None}
        model_name  : backbone used for embedding extraction (e.g. flan-t5-large)
        metric_mode : "hard" (ZCA whitening + L2) or "dynamics" (SRM)
        shrink_factor: Ledoit-Wolf shrinkage factor

    Returns:
        SRTRouter with signatures for all tasks in task_list.

    Handles dimension mismatch: if cached embeddings have wrong dim
    (e.g. cached from small model, now running large), re-extracts.
    """
    from transformers import AutoConfig
    expected_d = AutoConfig.from_pretrained(model_name).d_model

    router = SRTRouter(
        srt_metric_mode=metric_mode,
        use_shrink=True,
        shrink_factor=shrink_factor,
    )

    for t_name in task_list:
        if t_name not in router_state:
            print(f"    WARNING: {t_name} not in router_state, skipping")
            continue

        emb_path = router_state[t_name].get("emb_path")
        if emb_path and os.path.exists(emb_path):
            data = np.load(emb_path)
            h_train = data["embeddings"]
            # ── Re-extract if dimension mismatch ───────────────────────────
            if h_train.shape[1] != expected_d:
                print(f"    [DIM MISMATCH] {t_name}: cached dim={h_train.shape[1]} "
                      f"≠ expected={expected_d} (model={model_name}) → re-extracting")
                train_data = router_state[t_name].get("train_data")
                if train_data is None:
                    train_data = load_task_data(t_name, "train")
                if train_data:
                    emb_new, _ = extract_frozen_embeddings(
                        model_name, train_data, max_samples=500,
                        cache_path=emb_path,
                    )
                    h_train = emb_new.numpy()
        else:
            # Fall back: extract embeddings on the fly
            train_data = router_state[t_name].get("train_data")
            if train_data is None:
                train_data = load_task_data(t_name, "train")
            if train_data is None:
                print(f"    WARNING: no data for {t_name}, skipping router entry")
                continue
            print(f"    WARNING: no cached embeddings for {t_name}, "
                  f"extracting with {model_name}")
            emb_new, _ = extract_frozen_embeddings(
                model_name, train_data, max_samples=500, cache_path=emb_path,
            )
            h_train = emb_new.numpy()

        sig = router.add_task(task_id=t_name, h_train=h_train)
        print(f"    [SRT] Added {t_name}: PaR={sig.par:.1f}, "
              f"metric={sig.metric}, n={sig.n}")

    return router


def compute_srt_distances(
    router: SRTRouter,
    t_name: str,
    all_prior_tasks: List[str],
) -> Dict[str, float]:
    """
    Compute pairwise SRT distances from task t to all prior tasks s < t.

    Args:
        router        : SRTRouter with signatures for all tasks in all_prior_tasks
        t_name        : name of the target (new) task
        all_prior_tasks: list of prior task names

    Returns:
        dict: {prior_task_name: distance}
    """
    # Get t's centroid from router (if already added) or compute separately
    # Since t is NOT yet added to the router (we're pre-computing distances
    # to decide init), we need to extract its embeddings and compute distances.
    # For simplicity, we use the stored embeddings from router_state.

    distances = {}
    t_sig = router.signatures.get(t_name)
    if t_sig is not None:
        # t is already in router
        for s_name in all_prior_tasks:
            s_sig = router.signatures.get(s_name)
            if s_sig is None:
                distances[s_name] = float("inf")
                continue
            # Whitened L2 distance
            d = np.linalg.norm(t_sig.mu - s_sig.mu)
            distances[s_name] = float(d)
    else:
        # t not in router yet — this is the normal case
        # We can't compute distances without t's centroid.
        # This function is typically called after t's embeddings are available.
        # Return empty or use stored state.
        for s_name in all_prior_tasks:
            distances[s_name] = float("inf")

    return distances


def _compute_temperatures(router: SRTRouter) -> float:
    """
    Compute temperature τ = median of pairwise SRT distances.

    This is the data-driven temperature from Section 3.3 of SRT_WARM_INIT_PROPOSAL.md.
    """
    task_list = sorted(router.signatures.keys())
    T = len(task_list)
    if T < 2:
        return 1.0

    dists = []
    for i in range(T):
        for j in range(i + 1, T):
            d = np.linalg.norm(
                router.signatures[task_list[i]].mu -
                router.signatures[task_list[j]].mu
            )
            dists.append(d)

    tau = float(np.median(dists))
    return max(tau, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str = "google/flan-t5-small",
    adapter_mode: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    device: Optional[torch.device] = None,
):
    """
    Build a T5 model with GainLoRA LoRA adapters.

    CRITICAL: Must use T5ForConditionalGeneration from t5_gainlora_inflora.py
    (NOT AutoModelForSeq2SeqLM from HuggingFace), because:
      - HF T5Attention.forward does NOT call lora_q/lora_v → LoRA is dead code
      - GainLoRA T5Attention.forward DOES call lora_q/lora_v → LoRA actually works

    Uses T5ForConditionalGeneration.from_pretrained(config, prompt_config)
    which creates lora_q, lora_v as first-class attributes in T5Attention.__init__.

    The model returned has the SAME architecture as run_t5.py training,
    just without trans_input/prompt_key routing (isolated test).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"    Loading GainLoRA model: {model_name} (device={device})")


    # All compatibility patches (ipdb, transformers, loralib) are at MODULE level.
    # No duplicate patches needed here — import directly.
    from t5_gainlora_inflora import T5ForConditionalGeneration


    config = AutoConfig.from_pretrained(model_name)

    # prompt_config mirrors run_t5.py lines 513-524
    # Note: previous_lora_path=None, task_id=0 → no previous_lora_weights init
    prompt_config = {
        'seq_len': 512,
        'mlp_hidden_dim': 100,
        'attn_temperature': 1,    # = √(2*dim) when temperature==1
        'previous_lora_path': None,
        'previous_prompt_key_path': None,
        'task_id': 0,              # current task (standalone test)
        'run_single': True,        # no prompt_key stored
        'lora_r': lora_rank,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
    }

    # Use same from_pretrained call as run_t5.py line 533
    # This calls T5ForConditionalGeneration.__init__(config, prompt_config)
    # which creates lora_q/lora_v in T5Attention.__init__ — real LoRA that works
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        prompt_config,
        config=config,
    )

    # ── FIX: Re-initialize lora_A after from_pretrained ─────────────────────
    # from_pretrained wraps __init__ in no_init_weights() context,
    # making reset_parameters() a no-op → lora_A stays zeros.
    # This mirrors run_t5.py lines 550-555 exactly.
    _n_reinit = 0
    for _module in model.modules():
        if (hasattr(_module, 'lora_A') and hasattr(_module, 'lora_B')
                and hasattr(_module, 'reset_parameters')):
            nn.init.kaiming_uniform_(_module.lora_A, a=math.sqrt(5))
            _n_reinit += 1
    print(f"    [FIX] Re-initialized lora_A in {_n_reinit} LoRA layers "
          f"(from_pretrained bypassed init)")

    model.to(device)
    model.eval()

    # Attach frozen encoder for SRT routing (matches run_t5.py lines 581-589)
    frozen_enc = T5EncoderModel.from_pretrained(model_name)
    frozen_enc.eval()
    for p in frozen_enc.parameters():
        p.requires_grad = False
    model.encoder.encoder_frozen = frozen_enc

    # ── Fix 3: get_head_mask missing on T5Stack in newer transformers ────
    # t5_gainlora_inflora.py T5Stack.forward() calls self.get_head_mask()
    # but T5Stack doesn't inherit from PreTrainedModel in newer transformers.
    def _get_head_mask(head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            return head_mask
        return [None] * num_hidden_layers

    for submod in [model.encoder, model.decoder]:
        if not hasattr(submod, 'get_head_mask'):
            submod.get_head_mask = _get_head_mask

    # Verify LoRA layers exist and are real (not dead)
    lora_count = sum(
        1 for m in model.modules()
        if hasattr(m, 'lora_A') and hasattr(m, 'lora_B')
    )
    print(f"    LoRA layers verified: {lora_count} (from GainLoRA T5Attention)")

    return model


# ── attach GainLoRA adapters to T5 model ────────────────────────────────────

def _attach_gainlora_adapters(model, lora_rank: int, lora_alpha: int, lora_dropout: float):
    """
    No-op for models built via T5ForConditionalGeneration.from_pretrained.

    T5ForConditionalGeneration.from_pretrained already creates real lora_q and
    lora_v LoRALayer instances in T5Attention.__init__ (see t5_gainlora_inflora.py
    lines 422, 426). The forward pass in T5Attention.forward (line 669, 678)
    calls lora_q and lora_v — they are FIRST-CLASS participants, not patches.

    Calling this function would create DUPLICATE LoRALayer attributes that are
    never used by the forward pass (since forward reads self.lora_q once).
    So this is intentionally a no-op when the model already has LoRA layers.
    """
    existing = sum(
        1 for m in model.modules()
        if hasattr(m, 'lora_A') and hasattr(m, 'lora_B')
    )
    print(f"    [_attach_gainlora_adapters] LoRA already present from "
          f"T5ForConditionalGeneration.__init__: {existing} layers (no-op)")


# ─────────────────────────────────────────────────────────────────────────────
#  LORA WEIGHT INITIALIZATION  (core of SGWI)
# ─────────────────────────────────────────────────────────────────────────────

def init_lora_weights(
    model: nn.Module,
    mode: str,
    router: Optional[SRTRouter],
    t_name: str,
    all_lora_paths: Optional[List[str]],
    task_list: List[str],
    lora_rank: int = 8,
    temperature: Optional[float] = None,
):
    """
    Apply one of three LoRA initialization strategies.

    Args:
        model        : T5 model with LoRA adapters (from build_model)
        mode         : RANDOM_INIT | NTI_INIT | SVD_FUSION_INIT
        router       : SRTRouter with signatures for tasks 0..t-1
                       (not needed for RANDOM_INIT)
        t_name       : name of current task
        all_lora_paths: NOT USED in isolated tests (LoRA checkpoints
                       are not available locally). For NTI/SFI, we use
                       router distances to determine similarity, and
                       simulate the LoRA weights using the frozen backbone
                       embedding geometry.
        task_list    : full 15-task list (needed for SRT distance lookup)
        lora_rank    : LoRA rank (default 8)
        temperature  : τ for softmax weighting. If None → auto (median).

    ─────────────────────────────────────────────────────────────
    Implementation Notes:

    For RANDOM INIT (original behavior):
      - A: kaiming_uniform_(A, a=sqrt(5))
      - B: zeros(B)
      → ΔW = BA = 0 at start

    For NTI INIT (Nearest-Task Init):
      - Find nearest task s* by SRT distance
      - Approximate LoRA transfer using the observation that:
        optimal LoRA direction ≈ backbone gradient direction
        We model: B_s*A_s ≈ μ_s · (μ_s)ᵀ (outer product of centroid)
        This captures the TASK GEOMETRY without needing real checkpoints.

    For SFI INIT (SVD Fusion):
      - Compute w_s = softmax(-d_SRT(t,s) / τ) for all s < t
      - ΔW_init = Σ w_s · (μ_s · μ_sᵀ)   [outer product proxy]
      - SVD → rank-r → B_t, A_t
    ─────────────────────────────────────────────────────────────
    """
    if mode == RANDOM_INIT:
        _init_random(model)
        return

    if router is None:
        print(f"    [INIT] WARNING: router=None for {mode}, falling back to RANDOM")
        _init_random(model)
        return

    # Get current task index in order
    t_idx = task_list.index(t_name) if t_name in task_list else -1

    if mode == NTI_INIT:
        _init_nti(model, router, t_name, task_list, lora_rank)
        return

    if mode == SVD_FUSION_INIT:
        _init_sfi(model, router, t_name, task_list, lora_rank, temperature)
        return

    raise ValueError(f"Unknown init mode: {mode}")


def _init_random(model: nn.Module):
    """Standard random init: kaiming_uniform_(A), zeros(B)."""
    count = 0
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B)
            count += 1
    print(f"    [INIT] RANDOM: kaiming_uniform_(A), zeros(B) → {count} layers")


def _init_nti(
    model: nn.Module,
    router: SRTRouter,
    t_name: str,
    task_list: List[str],
    lora_rank: int,
):
    """
    Nearest-Task Init (NTI).

    Finds the task s* with minimum SRT distance to t, then
    initializes LoRA_t using a proxy derived from the embedding geometry.

    The proxy is: the outer product μ_s · μ_sᵀ captures the principal
    semantic direction of the source task's data in embedding space.
    Rank-r SVD gives B_s*·A_s* ≈ U[:,:r]·Σ^{1/2} and A_s*·B_s* ≈ Σ^{1/2}·V[:r,:].
    """
    t_idx = task_list.index(t_name) if t_name in task_list else -1
    prior_tasks = task_list[:t_idx] if t_idx > 0 else []
    if not prior_tasks:
        print(f"    [INIT] NTI: no prior tasks, falling back to RANDOM")
        _init_random(model)
        return

    # t may or may not be in router. If not, we need to add it first
    # to compute distances. The router_state contains the cached embeddings.
    t_sig = router.signatures.get(t_name)
    if t_sig is None:
        # Need t's embeddings — check if we can load them
        # For NTI, we don't strictly need t in the router since distances
        # are computed from other tasks' centroids anyway.
        # Use the nearest by computing distances from all prior sigs.
        print(f"    [INIT] NTI: {t_name} not in router — adding temporarily...")

        # Try to load from cache
        emb_path = os.path.join(RESULTS_DIR, "cache", f"emb_{t_name}.npz")
        if os.path.exists(emb_path):
            h_t = np.load(emb_path)["embeddings"]   # saved via np.savez_compressed(embeddings=...)
            t_sig = router.add_task(task_id=t_name, h_train=h_t)
        else:
            # Can't add without embeddings — fall back to using distances
            # among prior tasks and picking the one with smallest max-distance
            # (i.e., most "central" task among priors)
            print(f"    [INIT] NTI: no cached embeddings for {t_name} → using "
                  f"most-central prior task")
            # Find the most central prior task (min max distance to others)
            best_task = None
            best_score = float("inf")
            for s_name in prior_tasks:
                s_sig = router.signatures.get(s_name)
                if s_sig is None:
                    continue
                max_d = max(
                    float(np.linalg.norm(router.signatures[s2].mu - s_sig.mu))
                    for s2 in prior_tasks
                    if s2 != s_name and router.signatures.get(s2) is not None
                )
                if max_d < best_score:
                    best_score = max_d
                    best_task = s_name
            if best_task is None:
                best_task = prior_tasks[-1]
            print(f"    [INIT] NTI: fallback → using most central={best_task}")
            s_sig = router.signatures[best_task]
            mu_s = s_sig.mu.astype(np.float64)
            delta_W_proxy = np.outer(mu_s, mu_s)
            _set_lora_from_deltaW(model, delta_W_proxy, lora_rank, tag=f"NTI-{best_task}")
            return

    # Compute distances to all prior tasks
    dists = {}
    for s_name in prior_tasks:
        s_sig = router.signatures.get(s_name)
        if s_sig is None:
            continue
        d = float(np.linalg.norm(t_sig.mu - s_sig.mu))
        dists[s_name] = d

    if not dists:
        print(f"    [INIT] NTI: no prior signatures found, falling back to RANDOM")
        _init_random(model)
        return

    nearest = min(dists, key=dists.get)
    min_dist = dists[nearest]
    print(f"    [INIT] NTI: nearest={nearest}, d={min_dist:.3f}")

    # Proxy: ΔW_s ≈ μ_s · μ_sᵀ  (outer product captures task geometry)
    s_sig = router.signatures[nearest]
    mu_s = s_sig.mu.astype(np.float64)        # (d,)
    delta_W_proxy = np.outer(mu_s, mu_s)       # (d, d)

    # Rank-r SVD → B_t, A_t
    _set_lora_from_deltaW(model, delta_W_proxy, lora_rank, tag=f"NTI-{nearest}")


def _init_sfi(
    model: nn.Module,
    router: SRTRouter,
    t_name: str,
    task_list: List[str],
    lora_rank: int,
    temperature: Optional[float] = None,
):
    """
    SVD Fusion Init (SFI) — the recommended SGWI strategy.

    Steps:
      1. Add t's embeddings to router if not already present
      2. Compute d_SRT(t, s) = ||μ_t - μ_s||_whitened for all s < t
      3. w_s = softmax(-d / τ)   [τ = data-driven temperature]
      4. ΔW_init = Σ_s w_s · (μ_s · μ_sᵀ)   [weighted blend of proxy ΔW]
      5. U, S, Vᵀ = rank-r SVD(ΔW_init)
      6. B_t = U[:, :r] · √S[:r]
         A_t = √S[:r] · V[:r, :]
    """
    t_idx = task_list.index(t_name) if t_name in task_list else -1
    prior_tasks = task_list[:t_idx] if t_idx > 0 else []

    if not prior_tasks:
        print(f"    [INIT] SFI: no prior tasks → falling back to RANDOM")
        _init_random(model)
        return

    # Ensure t's signature is in the router (needed to compute distances)
    t_sig = router.signatures.get(t_name)
    if t_sig is None:
        print(f"    [INIT] SFI: {t_name} not in router — adding from cache...")
        emb_path = os.path.join(RESULTS_DIR, "cache", f"emb_{t_name}.npz")
        if os.path.exists(emb_path):
            h_t = np.load(emb_path)["embeddings"]   # saved via np.savez_compressed(embeddings=...)
            t_sig = router.add_task(task_id=t_name, h_train=h_t)
        else:
            print(f"    [INIT] SFI: no cached embeddings for {t_name} → RANDOM")
            _init_random(model)
            return

    # Step 1: Compute SRT distances to all prior tasks
    dists = []
    for s_name in prior_tasks:
        s_sig = router.signatures.get(s_name)
        if s_sig is None:
            continue
        d = float(np.linalg.norm(t_sig.mu - s_sig.mu))
        dists.append((s_name, d))

    if not dists:
        print(f"    [INIT] SFI: no prior signatures → falling back to RANDOM")
        _init_random(model)
        return

    names, dist_arr = zip(*dists)
    dist_arr = np.array(dist_arr)

    # Step 2: Compute softmax weights with temperature τ
    if temperature is None:
        # Data-driven τ = median of pairwise distances across all known tasks
        tau = _compute_temperatures(router)
    else:
        tau = temperature
    tau = max(tau, 1e-8)
    weights = scipy_softmax(-dist_arr / tau)   # (T_prior,)

    print(f"    [INIT] SFI: τ={tau:.3f}, {len(prior_tasks)} prior tasks")
    for s_name, w in zip(names, weights):
        if w > 0.01:
            print(f"           {s_name}: w={w:.3f}")

    # Step 3: Weighted blend of proxy ΔW
    d = t_sig.d
    delta_W_init = np.zeros((d, d), dtype=np.float64)
    for s_name, w in zip(names, weights):
        s_sig = router.signatures[s_name]
        mu_s = s_sig.mu.astype(np.float64)    # (d,)
        # Proxy: ΔW_s ≈ μ_s · μ_sᵀ  (outer product captures task geometry)
        delta_W_init += w * np.outer(mu_s, mu_s)

    # Step 4-5: Rank-r SVD
    _set_lora_from_deltaW(model, delta_W_init, lora_rank, tag="SFI")


def _set_lora_from_deltaW(model, delta_W: np.ndarray, r: int, tag: str = ""):
    """
    Perform rank-r SVD on delta_W and set LoRA weights.

    delta_W ∈ ℝ^{d×d} (full rank or low rank)
    → U, S, Vᵀ = svd(delta_W)
    → keep top-r components
    → B = U[:, :r] · √S[:r]   ∈ ℝ^{d×r}
    → A = √S[:r] · V[:r, :]   ∈ ℝ^{r×d}
    """
    U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]

    sqrt_S = np.sqrt(S_r)
    B_init = U_r * sqrt_S[np.newaxis, :]   # (d, r)
    A_init = sqrt_S[:, np.newaxis] * Vt_r  # (r, d)

    print(f"    [{tag}] SVD: kept r={r}/{len(S)} singular values, "
          f"S[0]={S[0]:.2f}, S[r-1]={S[r-1]:.4f}")

    # Set weights on all LoRA layers in the model
    count = 0
    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            module.lora_A.data.copy_(torch.from_numpy(A_init.astype(np.float32)))
            module.lora_B.data.copy_(torch.from_numpy(B_init.astype(np.float32)))
            count += 1

    print(f"    [{tag}] Set LoRA weights on {count} layers → A,B ∈ ℝ^{r}×{delta_W.shape[0]}")


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def eval_zero_shot(model, test_data: List[Dict]) -> float:
    """
    Evaluate model WITHOUT training on test set.
    Returns accuracy as percentage.

    NOTE: T5ForConditionalGeneration from t5_gainlora_inflora.py does NOT inherit
    from GenerationMixin, so it has no .generate() method.
    We implement manual greedy decoding instead.
    """
    if not test_data:
        return 0.0

    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    model.eval()

    # Determine pad/eos token ids
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    correct = 0
    total   = 0

    with torch.no_grad():
        for sample in tqdm(test_data, desc="  Zero-shot eval", leave=False,
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            text_in, label = format_example(sample)
            inputs = tokenizer(
                text_in,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            # ── Manual greedy decoding (replaces model.generate) ──────────────
            # Encode (T5Stack needs head_mask=None to avoid AttributeError)
            encoder_out = model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                head_mask=None,
            )
            encoder_hidden = encoder_out.last_hidden_state

            # Decode: greedy, up to 50 new tokens
            decoder_input_ids = torch.full(
                (inputs["input_ids"].size(0), 1),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            max_new = 50

            for _step in range(max_new):
                # Forward decoder (T5Stack needs head_mask=None)
                decoder_out = model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=inputs.get("attention_mask"),
                    head_mask=None,
                )
                logits = model.lm_head(decoder_out.last_hidden_state)   # (B, seq, vocab)
                next_logits = logits[:, -1, :]                          # (B, vocab)
                next_token = next_logits.argmax(dim=-1, keepdim=True)    # (B, 1)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                # Stop if all EOS
                if (next_token == eos_token_id).all():
                    break

            pred_ids = decoder_input_ids[0].cpu()
            # Trim padding
            pred_ids = pred_ids[pred_ids != pad_token_id]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            pred_lower = pred.lower()
            label_lower = label.lower()

            # Normalize: strip common prefixes
            if pred_lower.startswith("output:"):
                pred_lower = pred_lower[len("output:"):].strip()
            if label_lower.startswith("output:"):
                label_lower = label_lower[len("output:"):].strip()

            if pred_lower == label_lower:
                correct += 1
            total += 1

    return (correct / max(total, 1)) * 100


def evaluate_model(
    model,
    test_data: List[Dict],
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Full evaluation of a trained model on test set.

    Returns dict with:
      - exact_match: % of exact string matches
      - rouge1: ROUGE-1 F-score
      - accuracy: normalized accuracy (after stripping prefix)
    """
    if not test_data:
        return {"exact_match": 0.0, "rouge1": 0.0, "accuracy": 0.0}

    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    model.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id  = tokenizer.eos_token_id  if tokenizer.eos_token_id  is not None else 1

    predictions = []
    references  = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="  Eval",
                      leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                      total=(len(test_data) + batch_size - 1) // batch_size):
            batch = test_data[i:i + batch_size]
            inputs_text = [format_example(s)[0] for s in batch]
            labels      = [format_example(s)[1] for s in batch]

            inputs = tokenizer(
                inputs_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # ── Manual greedy decoding (model.generate not available on
            #    T5ForConditionalGeneration — no GenerationMixin inheritance)
            encoder_out = model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                head_mask=None,
            )
            encoder_hidden = encoder_out.last_hidden_state  # (B, src_len, d)

            B = inputs["input_ids"].size(0)
            max_new = 50

            if B == 1:
                # Fast path: single sample
                decoder_input_ids = torch.full(
                    (1, 1), pad_token_id, dtype=torch.long, device=device
                )
                for _step in range(max_new):
                    decoder_out = model.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_hidden,
                        encoder_attention_mask=inputs.get("attention_mask"),
                        head_mask=None,
                    )
                    logits = model.lm_head(decoder_out.last_hidden_state)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
                    if (next_token == eos_token_id).all():
                        break
                pred = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True).strip()
                pred = pred.replace("Output:", "").replace("output:", "").strip()
                predictions.append(pred)
                references.append(labels[0])
            else:
                # Multi-sample batch: decode each independently
                for j in range(B):
                    h_j = encoder_hidden[j:j+1]           # (1, src_len, d)
                    am_j = inputs["attention_mask"][j:j+1] if "attention_mask" in inputs else None
                    dec_ids = torch.full((1, 1), pad_token_id, dtype=torch.long, device=device)
                    for _step in range(max_new):
                        dec_out = model.decoder(
                            input_ids=dec_ids,
                            encoder_hidden_states=h_j,
                            encoder_attention_mask=am_j,
                            head_mask=None,
                        )
                        logits = model.lm_head(dec_out.last_hidden_state)
                        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        dec_ids = torch.cat([dec_ids, next_tok], dim=1)
                        if (next_tok == eos_token_id).all():
                            break
                    pred = tokenizer.decode(dec_ids[0], skip_special_tokens=True).strip()
                    pred = pred.replace("Output:", "").replace("output:", "").strip()
                    predictions.append(pred)
                    references.append(labels[j])

    # Compute metrics
    exact_match = sum(
        1 for p, r in zip(predictions, references)
        if p.lower() == r.lower()
    ) / len(predictions) * 100

    # Simple ROUGE-1
    rouge1 = compute_rouge1(predictions, references)

    return {
        "exact_match": exact_match,
        "rouge1": rouge1,
        "accuracy": exact_match,   # alias for convenience
    }


def compute_rouge1(predictions: List[str], references: List[str]) -> float:
    """Compute ROUGE-1 F-score (word-level, unigram)."""
    from collections import Counter

    def unigrams(s):
        return Counter(s.lower().split())

    total_precision = 0.0
    total_recall   = 0.0

    for pred, ref in zip(predictions, references):
        pred_uni = unigrams(pred)
        ref_uni  = unigrams(ref)

        overlap = sum((pred_uni & ref_uni).values())
        total_precision += overlap / max(len(pred_uni), 1)
        total_recall    += overlap / max(len(ref_uni), 1)

    n = len(predictions)
    precision = total_precision / n
    recall    = total_recall    / n
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1 * 100


# ─────────────────────────────────────────────────────────────────────────────
#  ISOLATED TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_lora_isolated(
    model,
    train_data: List[Dict],
    test_data: List[Dict],
    n_epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    cache_dir: Optional[str] = None,
    tag: str = "",
    max_samples: Optional[int] = None,
) -> Dict[str, List[float]]:
    """
    Isolated training loop for one task — used in Tests 2 & 3.

    Trains ONLY the LoRA adapters (all other params frozen).
    Records loss and accuracy after each epoch.

    Args:
        model      : T5 model with LoRA adapters (pre-initialized)
        train_data : list of training samples
        test_data  : list of test samples
        n_epochs   : number of epochs
        lr         : learning rate for LoRA params
        batch_size : per-device batch size
        gradient_accumulation: gradient accumulation steps
        cache_dir  : if given, save/load checkpoint to this dir
        tag        : identifier for this training run
        max_samples: cap training samples (for few-shot)

    Returns:
        {"loss": [loss_per_epoch], "accuracy": [acc_per_epoch]}
    """
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    # Checkpoint cache
    ckpt_path = None
    if cache_dir and tag:
        ckpt_path = os.path.join(cache_dir, f"ckpt_{tag}.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            # Restore LoRA weights
            for module in model.modules():
                if hasattr(module, "lora_A"):
                    key = f"lora_A"
                    if key in state:
                        module.lora_A.data.copy_(state[key])
                if hasattr(module, "lora_B"):
                    key = f"lora_B"
                    if key in state:
                        module.lora_B.data.copy_(state[key])
            loss_curve = state.get("loss_curve", [])
            acc_curve  = state.get("acc_curve",  [])
            print(f"    [CACHE] Restored checkpoint: {ckpt_path}")
            return {"loss": loss_curve, "accuracy": acc_curve}

    # Prepare training data
    train_subset = train_data[:max_samples] if max_samples else train_data

    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    loss_curve = []
    acc_curve  = []

    total_steps = 0
    eval_every  = max(1, len(train_subset) // (batch_size * gradient_accumulation * 2))

    print(f"    Training: {len(train_subset)} samples × {n_epochs} epochs, "
          f"batch={batch_size}, lr={lr}")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()

        # Shuffle
        order = list(range(len(train_subset)))
        random.shuffle(order)

        batches = [order[s:s + batch_size] for s in range(0, len(train_subset), batch_size)]
        pbar = tqdm(batches, desc=f"  Epoch {epoch+1}/{n_epochs}", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches")
        for batch_idx in pbar:
            batch = [train_subset[i] for i in batch_idx]

            inputs_text = [format_example(s)[0] for s in batch]
            labels_text = [format_example(s)[1] for s in batch]

            inputs = tokenizer(
                inputs_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # as_target_tokenizer() removed in transformers v4+
            # For T5, set decoder_input_ids to labels shifted right (with pad as start)
            targets = tokenizer(
                labels_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50,
            ).to(device)
            # Prepend decoder_start_token_id to labels for T5 teacher-forcing
            decoder_start = tokenizer.pad_token_id or 0
            labels_ids = targets["input_ids"]
            # decoder input = shift labels right, prepend pad token
            decoder_input_ids = torch.cat(
                [
                    torch.full((labels_ids.size(0), 1), decoder_start, dtype=torch.long, device=device),
                    labels_ids[:, :-1],
                ],
                dim=1,
            )

            # Forward
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
                labels=targets["input_ids"],
            )
            loss = outputs.loss / gradient_accumulation
            loss.backward()

            epoch_loss += loss.item() * gradient_accumulation
            n_batches  += 1
            total_steps += 1

            if total_steps % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_curve.append(avg_loss)

        # Evaluate
        model.eval()
        metrics = evaluate_model(model, test_data)
        acc = metrics.get("accuracy", 0.0)
        acc_curve.append(acc)
        model.train()

        print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}  acc={acc:.2f}%")

    # Save checkpoint
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        state = {"loss_curve": loss_curve, "acc_curve": acc_curve}
        for module in model.modules():
            if hasattr(module, "lora_A"):
                state["lora_A"] = module.lora_A.data.clone()
            if hasattr(module, "lora_B"):
                state["lora_B"] = module.lora_B.data.clone()
        torch.save(state, ckpt_path)

    return {"loss": loss_curve, "accuracy": acc_curve}


# ─────────────────────────────────────────────────────────────────────────────
#  MISC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def scipy_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def get_task_stats(task_name: str) -> Dict[str, Any]:
    """Return statistics about a task."""
    train = load_task_data(task_name, "train")
    test  = load_task_data(task_name, "test")
    if train is None:
        return {}
    labels = set(s.get("label", s.get("output", "")) for s in train)
    return {
        "name": task_name,
        "n_train": len(train) if train else 0,
        "n_test":  len(test)  if test  else 0,
        "n_classes": len(labels),
        "random_baseline": 100.0 / len(labels) if labels else 50.0,
    }
