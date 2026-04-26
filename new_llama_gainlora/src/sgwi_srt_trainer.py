"""
SRTSGWITrainer
==============
Clean Continual Learning trainer for LLaMA GainLoRA.

Inherits from Trainer directly — NO GPM, NO replay, NO GainLoRATrainer.

Implements two CL hooks:
  get_reg_matrix()  — called BEFORE training each task
                      if --sgwi: SGWI warm-init lora_A + lora_B from past adapters
                      else:      no-op (kaiming-A / zero-B from re-init in main script)

  on_task_end(task_id)  — called AFTER training each task
                          extracts embeddings via frozen backbone,
                          adds {μ_t, Σ_t} to SRT router,
                          wires router into model for next task's inference routing

SRT routing convention (matches T5 standard):
  slot 0         = current task
  slot 1         = task_{T-1}  (most recent previous)
  slot T         = task_0      (oldest)
  (because previous_lora_list is reversed at load time in main script)
"""

import copy
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Trainer


# ── Inline metric helpers (avoid circular import with main script) ─────────────

def _tokenize_eval(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def _lcs_eval(left: list, right: list) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for lt in left:
        curr = [0]
        for idx, rt in enumerate(right, 1):
            curr.append(prev[idx - 1] + 1 if lt == rt else max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l_eval(pred: str, ref: str) -> float:
    pt, rt = _tokenize_eval(pred), _tokenize_eval(ref)
    if not pt or not rt:
        return 0.0
    lcs = _lcs_eval(pt, rt)
    p, r = lcs / len(pt), lcs / len(rt)
    if p + r == 0:
        return 0.0
    beta = 1.2
    b2 = beta * beta
    return (1 + b2) * p * r / (r + b2 * p)

# ── SRTRouter import: load from new_gainlora/src ─────────────────────────────
_NG_SRC = str(Path(__file__).resolve().parent.parent.parent / "new_gainlora" / "src")
if _NG_SRC not in sys.path:
    sys.path.insert(0, _NG_SRC)

from srt_router import SRTRouter  # noqa: E402 (path injection above)


class SRTSGWITrainer(Trainer):
    """
    Trainer with SRT routing + optional SGWI warm-initialization.

    Extra constructor args vs Trainer:
        cur_task_id         (int)  : 0-based index of current task in task_order
        task_order          (list) : full CL task sequence ['task1572_...', ...]
        sgwi                (bool) : enable SGWI warm-init (default False = full_lora)
        srt_shrinkage       (str)  : 'ridge' | 'oas' | 'lw' | 'none'  (default 'ridge')
        srt_max_emb_samples (int)  : max batches for embedding extraction (default 500)
        srt_load_path       (str)  : dir containing srt_signatures.npz to load
        srt_skip_forward    (bool) : load pre-extracted embeddings from disk

        # In-training evaluation (mirrors T5 load_best_model_at_end):
        dev_samples         (list) : raw dev sample dicts for RougeL eval during training
        tokenizer                   : tokenizer for encoding prompts + decoding preds
        max_source_length   (int)  : truncation length for prompts
        max_new_tokens      (int)  : max tokens to generate per example
        eval_batch_size     (int)  : batch size for generation during evaluate()
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        cur_task_id: int,
        task_order: List[str],
        sgwi: bool = False,
        srt_shrinkage: str = "ridge",
        srt_max_emb_samples: int = 500,
        srt_load_path: Optional[str] = None,
        srt_skip_forward: bool = False,
        # ── in-training evaluation (best-model selection) ──
        dev_samples: Optional[List[dict]] = None,
        tokenizer: Any = None,
        max_source_length: int = 1024,
        max_new_tokens: int = 50,
        eval_batch_size: int = 4,
        **kwargs,
    ):
        super().__init__(model=model, args=args, train_dataset=train_dataset, **kwargs)
        self.cur_task_id = cur_task_id
        self.task_order = task_order
        self.sgwi = sgwi
        self.srt_shrinkage = srt_shrinkage
        self.srt_max_emb_samples = srt_max_emb_samples
        self.srt_load_path = srt_load_path
        self.srt_skip_forward = srt_skip_forward
        self.srt_router: Optional[SRTRouter] = None
        # ── best-model tracking (replaces load_best_model_at_end for CausalLM) ──
        self.dev_samples = dev_samples or []
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
        self._best_eval_rougeL: float = -1.0
        self._best_lora_A_state: Optional[Dict[str, torch.Tensor]] = None
        self._best_lora_B_state: Optional[Dict[str, torch.Tensor]] = None

        self._srt_init()

    # ─────────────────────────────────────────────────────────────────────────
    # In-training evaluation (mirrors T5 Seq2SeqTrainer + load_best_model_at_end)
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Override Trainer.evaluate() to run generation-based RougeL eval.

        Seq2SeqTrainer (T5) does this automatically; for CausalLM we must
        override manually. Called by Trainer every eval_steps during training.

        Tracks best RougeL checkpoint in-memory (instead of saving 7B files).
        Call restore_best_model() after train() to reload the best weights.
        """
        if not self.dev_samples or self.tokenizer is None:
            return {}

        model = self.model
        model.eval()
        device = next(model.parameters()).device
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        predictions: List[str] = []
        references:  List[str] = []

        n_batches = math.ceil(len(self.dev_samples) / self.eval_batch_size)
        step = self.state.global_step
        for start in tqdm(
            range(0, len(self.dev_samples), self.eval_batch_size),
            total=n_batches,
            desc=f"Eval @ step {step}",
            leave=False,
        ):
            batch = self.dev_samples[start : start + self.eval_batch_size]
            prompts = [
                s["Instance"]["instruction"].format(s["Instance"]["sentence"])
                for s in batch
            ]
            refs = [s["Instance"]["label"] for s in batch]

            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_source_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            input_length = encoded["input_ids"].shape[1]

            gen_cfg = copy.deepcopy(model.generation_config)
            gen_cfg.max_length   = None
            gen_cfg.do_sample    = False
            gen_cfg.num_beams    = 1
            gen_cfg.temperature  = None
            gen_cfg.top_p        = None
            gen_cfg.top_k        = None
            gen_cfg.pad_token_id = pad_id
            gen_cfg.eos_token_id = self.tokenizer.eos_token_id
            gen_cfg.bos_token_id = self.tokenizer.bos_token_id
            gen_cfg.max_new_tokens = self.max_new_tokens

            with torch.no_grad():
                generated = model.generate(
                    input_ids          = encoded["input_ids"],
                    input_ids_wo_label = encoded["input_ids"],
                    attention_mask     = encoded["attention_mask"],
                    generation_config  = gen_cfg,
                )

            for gen_ids, ref in zip(generated, refs):
                pred = self.tokenizer.decode(
                    gen_ids[input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                predictions.append(pred)
                references.append(ref.strip())

        model.train()

        rouge_l = (
            sum(_rouge_l_eval(p, r) for p, r in zip(predictions, references))
            / max(len(references), 1)
        )
        metrics = {f"{metric_key_prefix}_rougeL": round(rouge_l, 4)}

        # ── Best-model tracking (in-memory, avoids saving 7B checkpoint files) ──
        if rouge_l > self._best_eval_rougeL:
            self._best_eval_rougeL = rouge_l
            self._best_lora_A_state = {
                k: v.detach().clone().cpu()
                for k, v in self.model.named_parameters()
                if "lora_A" in k and "previous_lora_weights" not in k
            }
            self._best_lora_B_state = {
                k: v.detach().clone().cpu()
                for k, v in self.model.named_parameters()
                if "lora_B" in k and "previous_lora_weights" not in k
            }
            print(
                f"  [EVAL] ★ New best eval_rougeL={rouge_l:.4f} "
                f"(step={self.state.global_step}) — snapshot saved in memory"
            )
        else:
            print(f"  [EVAL] eval_rougeL={rouge_l:.4f} (best={self._best_eval_rougeL:.4f})")

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def restore_best_model(self) -> float:
        """
        Restore in-memory best checkpoint weights back into the live model.
        Equivalent to HF Trainer's load_best_model_at_end=True, but without
        saving 7B model files.

        Returns best eval_rougeL (or -1.0 if no eval was run).
        Call this AFTER trainer.train() and BEFORE on_task_end() / saving LoRA.
        """
        if self._best_lora_A_state is None:
            print("  [EVAL] No best checkpoint found (eval never ran?). Using final model.")
            return self._best_eval_rougeL

        restored = 0
        for name, param in self.model.named_parameters():
            if name in self._best_lora_A_state:
                param.data.copy_(self._best_lora_A_state[name].to(param.device))
                restored += 1
            elif name in self._best_lora_B_state:
                param.data.copy_(self._best_lora_B_state[name].to(param.device))
                restored += 1

        print(
            f"  [EVAL] Restored best model (eval_rougeL={self._best_eval_rougeL:.4f}, "
            f"{restored} tensors)"
        )
        return self._best_eval_rougeL

    # ─────────────────────────────────────────────────────────────────────────
    # SRT initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _srt_init(self):
        if self.srt_router is not None:
            return
        self.srt_router = SRTRouter(shrinkage=self.srt_shrinkage)
        if self.srt_load_path is not None:
            self.load_srt_signatures(self.srt_load_path, wire_model=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-training hook
    # ─────────────────────────────────────────────────────────────────────────

    def get_reg_matrix(self):
        """
        Called BEFORE trainer.train().

        Task 0  → nothing (no prior adapters).
        Task >0 → SGWI warm-init if self.sgwi else no-op.
        """
        if self.cur_task_id == 0:
            print("[SGWI] Task 0: no prior adapters, skipping warm-init.")
            return

        if not self.sgwi:
            print(f"[SGWI] Task {self.cur_task_id}: sgwi=False → full_lora mode (kaiming-A / zero-B).")
            return

        print(f"[SGWI] Task {self.cur_task_id}: computing SRT weights for warm-init…")
        srt_weights = self._compute_sgwi_weights()
        if not srt_weights:
            print("[SGWI] No SRT weights available. Skipping warm-init.")
            return

        print(f"[SGWI] SRT weights: {srt_weights}")
        self._sgwi_init_a(srt_weights)
        self._fuse_past_lora_adapters(srt_weights)

    # ─────────────────────────────────────────────────────────────────────────
    # Post-training hook
    # ─────────────────────────────────────────────────────────────────────────

    def on_task_end(self, task_id: str):
        """Called AFTER trainer.train(): compute SRT signature and wire router."""
        self._compute_and_store_signature(task_id)
        self._replace_attention_routing()
        if self.srt_router:
            s = self.srt_router.summary()
            print(f"[SRT] Router: {s['n_tasks']} tasks, avg_PaR={s['avg_par']:.1f}, "
                  f"metrics={s['metrics']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_task_embeddings(
        self, max_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, List]:
        """
        Extract (B, d) embeddings for the current training task.

        Mode 1 (srt_skip_forward=True): load from pre-extracted .npz on disk.
        Mode 2 (default): forward through self.model.model.encoder_frozen.
        """
        if max_samples is None:
            max_samples = self.srt_max_emb_samples

        # ── Mode 1: disk cache ────────────────────────────────────────────
        if self.srt_skip_forward:
            cur_task = self.task_order[self.cur_task_id]
            emb_path = self._srt_emb_cache_path(cur_task)
            if emb_path and os.path.exists(emb_path):
                print(f"  [SRT] ★ LOAD FROM CACHE: {emb_path}")
                data = np.load(emb_path, allow_pickle=True)
                embs = torch.from_numpy(data["embeddings"]).float()
                embs = embs[:max_samples]
                print(f"  [SRT]   → {embs.shape[0]} embeddings, dim={embs.shape[1]}")
                return embs, []
            print(f"  [SRT] Cache miss for '{cur_task}' → falling back to forward pass")

        # ── Mode 2: forward pass through frozen backbone ──────────────────
        core = self.model.model  # LlamaModel
        encoder_frozen = getattr(core, "encoder_frozen", None)
        if encoder_frozen is None:
            print("  [SRT] WARNING: encoder_frozen not attached. Returning empty embeddings.")
            return torch.empty(0), []

        print(f"  [SRT] → Forward extraction (max {max_samples} batches)")
        train_dl = self.get_train_dataloader()
        h_list = []
        pad_id = self.model.config.pad_token_id or 0

        for step, inputs in enumerate(train_dl):
            if step >= max_samples:
                break
            inputs = self._prepare_inputs(inputs)

            # Use source-only ids for routing (no label tokens)
            src_ids = inputs.get("input_ids_wo_label", inputs["input_ids"])
            src_mask = (src_ids != pad_id).long()

            with torch.no_grad():
                h = encoder_frozen(src_ids, src_mask)  # (B, d)
            h_list.append(h.float().cpu())

        if not h_list:
            return torch.empty(0), []
        return torch.cat(h_list, dim=0), []

    def _srt_emb_cache_path(self, task_name: str) -> Optional[str]:
        model_name = getattr(self.model.config, "_name_or_path", None) or ""
        backbone = None
        if "llama-3" in model_name.lower():
            backbone = "Meta-Llama-3-8B"
        elif "llama-2" in model_name.lower() or "llama_2" in model_name.lower():
            backbone = "Llama-2-7b-hf"
        if backbone is None:
            return None
        split = "SuperNI" if task_name.startswith("task") else "Long_Sequence"
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "new_gainlora", "embeddings",
        )
        return os.path.join(root, backbone, split, task_name, "train.npz")

    # ─────────────────────────────────────────────────────────────────────────
    # SRT signature storage and routing
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_and_store_signature(self, task_id: str):
        if self.srt_router is None:
            self._srt_init()
        h_train, _ = self._extract_task_embeddings(self.srt_max_emb_samples)
        if h_train.shape[0] == 0:
            print(f"  [SRT] WARNING: no embeddings for task '{task_id}', skipping signature.")
            return
        sig = self.srt_router.add_task(
            task_id=task_id, h_train=h_train.float().cpu().numpy()
        )
        print(f"  [SRT] Task '{task_id}': PaR={sig.par:.1f}, metric={sig.metric}, n={sig.n}")

    def _replace_attention_routing(self):
        """Wire SRT router into LlamaModel (self.model.model)."""
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return

        core = self.model.model
        current_task = self.task_order[self.cur_task_id]

        # slot 0 = current; slot k = task_order[cur_task_id - k]  (reverse chron)
        task_id_to_idx: Dict[str, int] = {current_task: 0}
        for prev_idx in range(self.cur_task_id):
            task_id_to_idx[self.task_order[prev_idx]] = self.cur_task_id - prev_idx

        core.srt_router = self.srt_router
        core.srt_task_id_to_idx = task_id_to_idx
        core.use_srt_routing = True
        print(
            f"  [SRT] Wired router: {len(self.srt_router.signatures)} sigs, "
            f"cur={current_task}, mapping={task_id_to_idx}"
        )

    def save_srt_signatures(self, output_dir: str):
        if self.srt_router is None:
            return
        path = os.path.join(output_dir, "srt_signatures.npz")
        self.srt_router.save(path)
        print(f"  [SRT] Saved signatures → {path}")

    def load_srt_signatures(self, checkpoint_dir: str, wire_model: bool = False):
        path = os.path.join(checkpoint_dir, "srt_signatures.npz")
        if not os.path.exists(path):
            print(f"  [SRT] No signatures found at {path}")
            return
        self.srt_router.load(path)
        print(f"  [SRT] Loaded {len(self.srt_router.signatures)} signatures from {path}")
        if wire_model and len(self.srt_router.signatures) > 0:
            self._replace_attention_routing()

    # ─────────────────────────────────────────────────────────────────────────
    # SGWI: compute SRT-weighted softmax over past tasks
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_sgwi_weights(self) -> Dict[str, float]:
        """
        Softmax weights over past tasks based on Mahalanobis distance from
        current task centroid to each past task centroid {μ_s}.

        Returns: {task_id: weight}  where Σ weights ≈ 1
        """
        if self.srt_router is None or len(self.srt_router.signatures) == 0:
            return {}

        h_train, _ = self._extract_task_embeddings(self.srt_max_emb_samples)
        if h_train.shape[0] == 0:
            return {}

        current_mu = h_train.mean(dim=0).cpu().numpy()

        distances: Dict = {}
        # Use PooledMahalanobis Sinv for SGWI weight computation
        # _impl._Sigma_inv_t holds the shrunk pooled covariance inverse
        impl = getattr(self.srt_router, "_impl", None)
        Sinv_np = None
        if impl is not None:
            Sinv_t = getattr(impl, "_Sigma_inv_t", None)
            if Sinv_t is not None:
                Sinv_np = Sinv_t.cpu().numpy()

        for task_id, sig in self.srt_router.signatures.items():
            diff = current_mu - sig.mu
            if Sinv_np is not None:
                # Pooled Mahalanobis distance
                d = float(diff @ Sinv_np @ diff)
            else:
                # Fallback: L2 distance
                d = float(np.sum(diff ** 2))
            distances[task_id] = d

        if not distances:
            return {}
        if len(distances) == 1:
            return {k: 1.0 for k in distances}

        # Softmax with τ = median heuristic
        tau = float(np.median(list(distances.values()))) + 1e-8
        weights = {k: math.exp(-d / tau) for k, d in distances.items()}
        Z = sum(weights.values()) + 1e-12
        return {k: w / Z for k, w in weights.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # SGWI: warm-init lora_A via SVD of weighted ΔW
    # ─────────────────────────────────────────────────────────────────────────

    def _sgwi_init_a(self, srt_weights: Dict[str, float]):
        """
        For each LlamaAttention module with lora_q/v and previous_lora_weights_q/v:
          ΔW = Σ_s w_s · (B_s @ A_s)          [out_dim, in_dim]
          SVD(ΔW):  U, S, Vt
          A_new = sqrt(S[:r]) * Vt[:r, :]      captures important input directions
        """
        model = self.model
        device = next(model.parameters()).device
        lora_r = getattr(self.args, "lora_r", 8)

        fused, skipped = 0, 0
        for name, module in model.named_modules():
            if not (hasattr(module, "lora_q") and hasattr(module, "lora_v")):
                continue
            prev_q = getattr(module, "previous_lora_weights_q", None)
            prev_v = getattr(module, "previous_lora_weights_v", None)
            if not prev_q:
                continue

            for lora_cur, prev_list in [
                (module.lora_q, prev_q),
                (module.lora_v, prev_v),
            ]:
                if prev_list is None:
                    continue

                delta_W = self._weighted_delta_w(srt_weights, prev_list)
                if delta_W is None:
                    skipped += 1
                    continue

                try:
                    U, S, Vt = torch.linalg.svd(
                        delta_W.to(device), full_matrices=False
                    )
                    r = min(lora_r, len(S))
                    A_new = torch.sqrt(S[:r] + 1e-12).unsqueeze(1) * Vt[:r, :]
                    lora_cur.lora_A.data.copy_(A_new.to(lora_cur.lora_A.device))
                    fused += 1
                except Exception as exc:
                    print(f"[SGWI-A] SVD failed for {name}: {exc}")
                    skipped += 1

        print(f"[SGWI-A] Warm-init A done: {fused} modules, {skipped} skipped")

    # ─────────────────────────────────────────────────────────────────────────
    # SGWI: warm-init lora_B via least-squares
    # ─────────────────────────────────────────────────────────────────────────

    def _fuse_past_lora_adapters(self, srt_weights: Dict[str, float]):
        """
        For each LlamaAttention module:
          ΔW = Σ_s w_s · (B_s @ A_s)
          B_warm = ΔW @ A_cur^T @ (A_cur @ A_cur^T + εI)^{-1}

        A_cur is already warm-init'd from _sgwi_init_a (call that first).
        """
        model = self.model
        device = next(model.parameters()).device

        fused, skipped = 0, 0
        for name, module in model.named_modules():
            if not (hasattr(module, "lora_q") and hasattr(module, "lora_v")):
                continue
            prev_q = getattr(module, "previous_lora_weights_q", None)
            prev_v = getattr(module, "previous_lora_weights_v", None)
            if not prev_q:
                continue

            for lora_cur, prev_list in [
                (module.lora_q, prev_q),
                (module.lora_v, prev_v),
            ]:
                if prev_list is None:
                    continue

                delta_W = self._weighted_delta_w(srt_weights, prev_list)
                if delta_W is None:
                    skipped += 1
                    continue

                try:
                    A_cur = lora_cur.lora_A.data.float().to(device)   # (r, in)
                    AtA = A_cur @ A_cur.T                               # (r, r)
                    eps = 1e-4 * torch.eye(A_cur.shape[0], device=device)
                    B_warm = (
                        delta_W.to(device)
                        @ A_cur.T
                        @ torch.linalg.inv(AtA + eps)
                    )                                                   # (out, r)
                    lora_cur.lora_B.data.copy_(B_warm.to(lora_cur.lora_B.device))
                    fused += 1
                except Exception as exc:
                    print(f"[SGWI-B] lora_B warm-init failed for {name}: {exc}")
                    skipped += 1

        print(f"[SGWI-B] Warm-init B done: {fused} modules, {skipped} skipped")

    # ─────────────────────────────────────────────────────────────────────────
    # Shared helper: compute weighted ΔW = Σ_s w_s * (B_s @ A_s)
    # ─────────────────────────────────────────────────────────────────────────

    def _weighted_delta_w(
        self,
        srt_weights: Dict[str, float],
        prev_list: nn.ModuleList,
    ) -> Optional[torch.Tensor]:
        """
        Compute weighted ΔW from past LoRA adapters.

        srt_weights keys are task name strings (e.g. 'task1572_samsum_summary').
        Slot mapping: prev_list[0] = most recent task = task_order[cur_task_id-1],
                      prev_list[i] = task_order[cur_task_id-1-i].
        """
        delta_W = None
        for task_id, w_s in srt_weights.items():
            # Map task name → prev_list index
            if isinstance(task_id, str):
                try:
                    pos = self.task_order.index(task_id)
                    idx = (self.cur_task_id - 1) - pos
                except ValueError:
                    continue
            else:
                idx = int(task_id)

            if idx < 0 or idx >= len(prev_list):
                continue

            pl = prev_list[idx]
            # Move to float for precision; may be on CPU (saved there to spare VRAM)
            BA = pl.lora_B.data.float() @ pl.lora_A.data.float()  # (out, in)
            delta_W = w_s * BA if delta_W is None else delta_W + w_s * BA

        if delta_W is None or delta_W.norm().item() < 1e-10:
            return None
        return delta_W
