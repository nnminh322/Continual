#!/usr/bin/env python3
"""
exp_fgcl.py — FGCL C2 Experiment Framework
==========================================
Scope: Training dynamics (initialization, optimizer, CL regularization).
Không xét: CalAttention routing mechanism (thuộc C1).

CLI:
  python exp_fgcl.py                                    # ALL phases, all benchmarks
  python exp_fgcl.py --phase T4 --benchmark Long_Sequence
  python exp_fgcl.py --model google/flan-t5-large --benchmark SuperNI
  python exp_fgcl.py --phase T1 --benchmark Long_Sequence --tasks sst2 imdb mnli

Methods (trong scope C2):
  standard_lora  — LoRA + AdamW (no CL mechanism)
  gainlora_root  — GainLoRA root port: LoRA + GPM + trans_input + prev_branches
  inflora        — LoRA + GPM (no routing, no trans_input)
  fgcl_fsr       — LoRA + FSR (Fisher Subspace Regularization)
  fgcl_kfng      — LoRA + FSR + KF-FNG optimizer
  fgcl_taa       — LoRA + FSR + TAA (Task Arithmetic Alignment)
  fgcl_sgr       — LoRA + SGR (GALA baseline: Soft Grassmannian Reg)

Checkpoint policy: NO .pt checkpoints saved. All intermediate state is kept in
memory during training. Only results JSON and logs are persisted.
This is intentional — checkpoints are not needed for analysis; they waste disk.
"""

import sys, json, time, math, argparse, gc, warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARK TASKS
# ═══════════════════════════════════════════════════════════════════════

BENCHMARK_TASKS = {
    "Long_Sequence": [
        "sst2", "imdb", "yelp", "amazon", "dbpedia", "agnews", "yahoo",
        "mnli", "boolq", "wic", "cb", "copa", "qqp", "rte", "multirc",
    ],
    "SuperNI": [
        "task1687_sentiment140_classification",
        "task363_sst2_polarity_classification",
        "task875_emotion_classification",
        "task073_commonsenseqa_answer_generation",
        "task591_sciq_answer_generation",
        "task002_quoref_answer_generation",
        "task1290_xsum_summarization",
        "task1572_samsum_summary",
        "task511_reddit_tifu_long_text_summarization",
        "task181_outcome_extraction",
        "task748_glucose_reverse_cause_event_detection",
        "task1510_evalution_relation_extraction",
        "task639_multi_woz_user_utterance_generation",
        "task1590_diplomacy_text_generation",
        "task1729_personachat_generate_next",
    ],
}

ALL_PHASES = ["T1", "T2", "T3", "T4"]
ALL_PHASE_METHODS = {
    "T1": ["standard_lora", "inflora", "fgcl_fsr"],
    "T2": ["standard_lora", "fgcl_kfng"],
    "T3": ["standard_lora", "fgcl_fsr", "fgcl_taa", "fgcl_sgr"],
    "T4": ["standard_lora", "gainlora_root", "inflora", "fgcl_fsr", "fgcl_kfng", "fgcl_taa"],
}
PHASE_DESCS = {
    "T1": "FSR vs GPM isolation",
    "T2": "KF-FNG convergence",
    "T3": "TAA vs SGR ablation",
    "T4": "Full comparison",
}
PHASE_TASK_COUNTS = {"T1": 6, "T2": 1, "T3": 8, "T4": None}
DEFAULT_EPOCHS = {"T1": 20, "T2": 30, "T3": 20, "T4": 20}
DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 16
DEFAULT_LR = 1e-3
DEFAULT_ATTN_LR = 1e-4
DEFAULT_BATCH = 8
DEFAULT_GPM_STEPS = 1000
DEFAULT_GPM_THRESH = 0.98
DEFAULT_CHUNK = 8
DEFAULT_LAMBDA_FSR = 0.1
DEFAULT_LAMBDA_TAA = 0.05
DEFAULT_LAMBDA_SGR = 0.1

# ── Evaluation frequency: eval every N tasks ──────────────────────────
DEFAULT_EVAL_EVERY = 3  # 0 to disable intermediate eval (only final)


# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════

ROOT_SRC = Path("/Users/nnminh322/Desktop/personal/Continual/root_gainlora/src")


def _make_prompt_config(task_id: int = 0, rank: int = 4, lora_alpha: int = 16,
                        lora_dropout: float = 0.0, run_single: bool = False,
                        previous_lora_path: Optional[str] = None,
                        previous_prompt_key_path: Optional[str] = None,
                        mlp_hidden_dim: int = 100):
    return {
        "task_id": task_id,
        "lora_r": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "run_single": run_single,
        "previous_lora_path": previous_lora_path,
        "previous_prompt_key_path": previous_prompt_key_path,
        "mlp_hidden_dim": mlp_hidden_dim,
    }


def load_gainlora_model(model_name: str, rank: int = 4, lora_alpha: int = 16,
                         lora_dropout: float = 0.0, task_id: int = 0,
                         use_trans_input: bool = False, chunk: int = 8,
                         previous_lora_path: Optional[str] = None,
                         previous_prompt_key_path: Optional[str] = None,
                         device: str = DEVICE):
    if not ROOT_SRC.exists():
        raise RuntimeError(
            f"root_gainlora source not found at {ROOT_SRC}. "
            "Please copy or mount the source to this path on the server."
        )
    sys.path.insert(0, str(ROOT_SRC))
    from t5_gainlora_inflora import T5ForConditionalGeneration

    prompt_config = _make_prompt_config(
        task_id=task_id,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        run_single=not use_trans_input,
        previous_lora_path=previous_lora_path,
        previous_prompt_key_path=previous_prompt_key_path,
    )

    root_model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        prompt_config=prompt_config,
    )
    root_model.config.use_cache = False

    for block in root_model.encoder.block:
        attn = block.layer[0]
        if hasattr(attn, 'get_chunk'):
            attn.get_chunk(chunk)
    for block in root_model.decoder.block:
        attn = block.layer[0]
        if hasattr(attn, 'get_chunk'):
            attn.get_chunk(chunk)

    if use_trans_input and hasattr(root_model.encoder, 'get_chunk'):
        root_model.encoder.get_chunk(chunk)

    for name, param in root_model.named_parameters():
        if "lora_" not in name and "trans_input" not in name and "prompt_key" not in name:
            param.requires_grad = False

    return root_model


# ═══════════════════════════════════════════════════════════════════════
# GPM — Gradient Projection Memory
# ═══════════════════════════════════════════════════════════════════════

class GPMAccumulator:
    def __init__(self, threshold: float = 0.98, chunk: int = 8):
        self.threshold = threshold
        self.chunk = chunk
        self.feature_list: List[List[Dict[int, torch.Tensor]]] = []

    def collect(self, model, dataloader, n_steps: int = 1000):
        for module in model.modules():
            if hasattr(module, "get_feature"):
                module.get_feature = True
        if hasattr(model.encoder, "get_trans_feature"):
            model.encoder.get_trans_feature = True

        model.eval()
        with torch.no_grad():
            for step_idx, batch in enumerate(dataloader):
                if step_idx >= n_steps:
                    break
                device_batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                                for k, v in batch.items()}
                try:
                    model(**device_batch)
                except Exception:
                    break

        for module in model.modules():
            if hasattr(module, "get_feature"):
                module.get_feature = False
        if hasattr(model.encoder, "get_trans_feature"):
            model.encoder.get_trans_feature = False

    def build(
        self,
        model,
        task_id: int = 0,
        total_tasks: int = 15,
        prev_features: Optional[List] = None,
    ) -> List[List[Dict[int, torch.Tensor]]]:
        features: List[List[Dict[int, torch.Tensor]]] = []
        thresh = (
            self.threshold
            if task_id == 0
            else (1.0 - self.threshold) * task_id / total_tasks + self.threshold
        )

        for layer_idx, block in enumerate(model.encoder.block):
            attn = block.layer[0]
            if not hasattr(attn, "matrix"):
                continue
            layer_feat = {}
            step = getattr(attn, "step", None)
            if step is None:
                d_model = getattr(attn, "d_model", None)
                if d_model is None:
                    continue
                step = d_model // self.chunk
            for ci in range(self.chunk):
                cov = attn.matrix.get(ci, None)
                if cov is None or cov.numel() == 0:
                    layer_feat[ci] = torch.eye(step, step)
                    continue
                cov = cov.float().cpu()
                try:
                    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
                except Exception:
                    U = torch.eye(cov.shape[0])
                    S = torch.ones(cov.shape[0])

                s2 = S ** 2
                s2_ratio = s2 / (s2.sum() + 1e-12)
                cum = torch.cumsum(s2_ratio, dim=0)
                r = (cum >= thresh).nonzero(as_tuple=True)[0]
                r = (r[0].item() + 1) if len(r) > 0 else 1
                r = min(r, U.shape[1])

                if prev_features and layer_idx < len(prev_features) and ci in prev_features[layer_idx]:
                    P = prev_features[layer_idx][ci].cpu()
                    cov_hat = cov - P @ P.T @ cov
                    try:
                        U2, S2, _ = torch.linalg.svd(cov_hat.float(), full_matrices=False)
                    except Exception:
                        U2 = torch.zeros(cov.shape[0], 0)
                        S2 = torch.zeros(cov.shape[0])
                    s2h = S2 ** 2
                    s2h_ratio = s2h / (s2h.sum() + 1e-12)
                    cumh = torch.cumsum(s2h_ratio, dim=0)
                    r2 = (cumh >= thresh).nonzero(as_tuple=True)[0]
                    r2 = (r2[0].item() + 1) if len(r2) > 0 else 1
                    r2 = min(r2, U2.shape[1])
                    if U2.shape[1] >= r2:
                        combined = torch.cat([P, U2[:, :r2]], dim=1)
                        try:
                            Q, _ = torch.linalg.qr(combined)
                            layer_feat[ci] = Q
                        except Exception:
                            layer_feat[ci] = combined
                    else:
                        layer_feat[ci] = P
                else:
                    layer_feat[ci] = U[:, :r]
            features.append(layer_feat)

        self.feature_list.append(features)
        return features

    def project(self, model):
        if not self.feature_list:
            return
        cur = self.feature_list[-1]
        layer_idx = 0
        for block in model.encoder.block:
            attn = block.layer[0]
            if not hasattr(attn, "lora_q"):
                continue
            for ci in range(self.chunk):
                A_q = attn.lora_q.lora_A.data
                A_v = attn.lora_v.lora_A.data
                s0, s1 = ci * attn.step, (ci + 1) * attn.step

                if layer_idx < len(cur) and ci in cur[layer_idx]:
                    P = cur[layer_idx][ci].to(A_q)
                    A_q[:, s0:s1] = A_q[:, s0:s1] - (P @ P.T @ A_q[:, s0:s1])
                    A_v[:, s0:s1] = A_v[:, s0:s1] - (P @ P.T @ A_v[:, s0:s1])

                nq = A_q[:, s0:s1].norm(dim=1, keepdim=True).clamp(min=1e-12)
                nv = A_v[:, s0:s1].norm(dim=1, keepdim=True).clamp(min=1e-12)
                A_q[:, s0:s1] = A_q[:, s0:s1] / (math.sqrt(3) * nq)
                A_v[:, s0:s1] = A_v[:, s0:s1] / (math.sqrt(3) * nv)
            layer_idx += 1


# ═══════════════════════════════════════════════════════════════════════
# FSR — Fisher Subspace Regularization
# ═══════════════════════════════════════════════════════════════════════

class FSR:
    def __init__(self, lam: float = 0.1, threshold: float = 0.99):
        self.lam = lam
        self.threshold = threshold
        self.fisher_ema: Dict[str, torch.Tensor] = {}
        self.bases: Dict[str, torch.Tensor] = {}

    def update(self, grad_dict: Dict[str, torch.Tensor]):
        for name, g in grad_dict.items():
            if g is None:
                continue
            g_flat = g.detach().reshape(-1).float()
            outer = g_flat.unsqueeze(1) @ g_flat.unsqueeze(0)
            if name not in self.fisher_ema:
                self.fisher_ema[name] = outer.clone()
            else:
                self.fisher_ema[name] = 0.99 * self.fisher_ema[name].float() + 0.01 * outer

    def build_bases(self):
        for name, F_mat in self.fisher_ema.items():
            if F_mat.shape[0] < 10:
                continue
            try:
                U, S, _ = torch.linalg.svd(F_mat.float(), full_matrices=False)
                s2 = S ** 2
                ratio = s2 / (s2.sum() + 1e-12)
                cum = torch.cumsum(ratio, dim=0)
                k = (cum >= self.threshold).nonzero(as_tuple=True)[0]
                k = (k[0].item() + 1) if len(k) > 0 else 1
                k = min(k, U.shape[1])
                if name in self.bases:
                    combined = torch.cat([self.bases[name], U[:, :k]], dim=1)
                    try:
                        Q, _ = torch.linalg.qr(combined)
                        self.bases[name] = Q
                    except Exception:
                        self.bases[name] = torch.cat([self.bases[name], U[:, :k]], dim=1)
                else:
                    self.bases[name] = U[:, :k]
            except Exception:
                pass

    def loss(self, grad_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        if not self.bases:
            return torch.tensor(0.0, device=DEVICE, dtype=torch.float), 0.0

        total_proj_sq = torch.tensor(0.0, device=DEVICE, dtype=torch.float)
        total_norm_sq = torch.tensor(0.0, device=DEVICE, dtype=torch.float)
        for name, g in grad_dict.items():
            if g is None or name not in self.bases:
                continue
            g_f = g.reshape(-1).float()
            total_norm_sq = total_norm_sq + (g_f ** 2).sum()
            U = self.bases[name].to(g.device).float()
            proj_sq = (U.T @ g_f).pow(2).sum()
            total_proj_sq = total_proj_sq + proj_sq

        if total_norm_sq < 1e-12:
            return torch.tensor(0.0, device=DEVICE, dtype=torch.float), 0.0
        loss_val = self.lam * total_proj_sq / total_norm_sq.detach()
        return loss_val, loss_val.item()


# ═══════════════════════════════════════════════════════════════════════
# KF-FNG — Kronecker-Factored Fisher Natural Gradient
# ═══════════════════════════════════════════════════════════════════════

class KFFNGOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: Tuple = (0.9, 0.999),
                 f_ema: float = 0.99, eps: float = 1e-8):
        defaults = dict(lr=lr, betas=betas, f_ema=f_ema, eps=eps)
        super().__init__(params, defaults)
        self._fs: Dict[int, Dict] = {}

    @torch.no_grad()
    def update_fisher(self, named_params, grad_dict):
        for name, param in named_params:
            if name not in grad_dict or grad_dict[name] is None:
                continue
            pid = id(param)
            g = grad_dict[name].detach().reshape(-1).float()
            g_sq = (g ** 2).mean()
            if pid not in self._fs:
                self._fs[pid] = {"ema": None, "cnt": 0}
            s = self._fs[pid]
            s["cnt"] += 1
            if s["ema"] is None:
                s["ema"] = g_sq.clone()
            else:
                f_ema_val = self.param_groups[0].get("f_ema", 0.99)
                s["ema"] = f_ema_val * s["ema"] + (1 - f_ema_val) * g_sq

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["betas"][0]
            b2 = group["betas"][1]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if not state:
                    state.update(
                        step=0,
                        exp_avg=torch.zeros_like(p),
                        exp_avg_sq=torch.zeros_like(p),
                    )

                state["step"] += 1
                st = state["step"]

                bias1 = 1.0 - b1 ** st
                bias2 = 1.0 - b2 ** st

                state["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)

                denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias2)).add_(eps)
                adam_up = state["exp_avg"] / denom * (lr / bias1)

                pid = id(p)
                fs = self._fs.get(pid, {})
                if fs.get("ema") is not None and fs["cnt"] > 10:
                    G = fs["ema"].clamp(min=1e-8).to(p)
                    ng_up = adam_up / G.clamp(min=1e-4)
                    update = 0.5 * adam_up + 0.5 * ng_up
                else:
                    update = adam_up

                p.data.add_(update, alpha=-1)

        return loss


# ═══════════════════════════════════════════════════════════════════════
# TAA — Task Arithmetic Alignment
# ═══════════════════════════════════════════════════════════════════════

class TAA:
    def __init__(self, lam: float = 0.05):
        self.lam = lam
        self.task_vectors: Dict[int, Dict[str, torch.Tensor]] = {}

    def register(self, task_id: int, model: torch.nn.Module):
        self.task_vectors[task_id] = {}
        a_params = {}
        b_params = {}
        scaling_by_layer = {}

        for name, param in model.named_parameters():
            if "lora_A" in name:
                layer = name.replace(".lora_A", "")
                a_params[layer] = param.detach().float().cpu()
            elif "lora_B" in name:
                layer = name.replace(".lora_B", "")
                parts = name.split(".")
                module = model
                for p in parts[:-1]:
                    module = getattr(module, p, None)
                    if module is None:
                        break
                scaling = getattr(module, "scaling", 1.0) if module is not None else 1.0
                b_params[layer] = param.detach().float().cpu()
                scaling_by_layer[layer] = float(scaling)

        for layer in a_params:
            if layer in b_params:
                A = a_params[layer]
                B = b_params[layer]
                s = scaling_by_layer.get(layer, 1.0)
                tau = B @ A * s
                self.task_vectors[task_id][layer] = tau

    def loss(self, model: torch.nn.Module,
             grad_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        if not self.task_vectors:
            return torch.tensor(0.0, device=DEVICE, dtype=torch.float), 0.0

        mags = {}
        total_mag = 0.0
        for tid, tv in self.task_vectors.items():
            mag = sum((v ** 2).sum().item() for v in tv.values())
            mags[tid] = math.sqrt(mag + 1e-8)
            total_mag += mag
        total_mag = max(total_mag, 1e-8)

        all_grads = []
        for name in sorted(grad_dict.keys()):
            g = grad_dict[name]
            if g is not None:
                all_grads.append(g.reshape(-1).detach().float())
        grad_cat = torch.cat(all_grads) if all_grads else torch.tensor([])

        total_align = 0.0
        for tid, tv in self.task_vectors.items():
            w = (mags[tid] ** 2) / total_mag
            tau_parts = []
            for name in sorted(tv.keys()):
                tau_parts.append(tv[name].reshape(-1).float())
            if not tau_parts:
                continue
            tau_cat = torch.cat(tau_parts).to(grad_cat.device)

            min_len = min(len(grad_cat), len(tau_cat))
            dot_sq = (grad_cat[:min_len] * tau_cat[:min_len]).sum().item() ** 2
            total_align += w * dot_sq

        g_norm = sum(
            (grad_dict[n] ** 2).sum().item()
            for n in grad_dict if grad_dict[n] is not None
        ) + 1e-8

        loss_val = self.lam * total_align / g_norm
        return torch.tensor(loss_val, device=DEVICE, dtype=torch.float), loss_val


# ═══════════════════════════════════════════════════════════════════════
# SGR — Soft Grassmannian Regularization
# ═══════════════════════════════════════════════════════════════════════

class SGR:
    def __init__(self, lam: float = 0.1):
        self.lam = lam
        self.prev_subspaces: Dict[str, torch.Tensor] = {}

    def register(self, name: str, lora_A: torch.Tensor):
        try:
            U, _, _ = torch.linalg.svd(lora_A.float(), full_matrices=False)
            k = min(lora_A.shape[0], lora_A.shape[1])
            self.prev_subspaces[name] = U[:, :k]
        except Exception:
            pass

    def loss(self, model: torch.nn.Module) -> Tuple[torch.Tensor, float]:
        if not self.prev_subspaces:
            return torch.tensor(0.0, device=DEVICE, dtype=torch.float), 0.0
        total = torch.tensor(0.0, device=DEVICE, dtype=torch.float)
        for name, param in model.named_parameters():
            if "lora_A" not in name:
                continue
            try:
                A = param.detach().float()
                U, _, _ = torch.linalg.svd(A, full_matrices=False)
                k = min(A.shape[0], A.shape[1])
                U_new = U[:, :k]
                if name in self.prev_subspaces:
                    U_prev = self.prev_subspaces[name].to(U_new.device)
                    kp = min(U_new.shape[1], U_prev.shape[1])
                    overlap = torch.norm(U_new[:, :kp].T @ U_prev[:, :kp], "fro") ** 2
                    total = total + overlap
            except Exception:
                pass
        loss_val = self.lam * total
        return loss_val, loss_val.item()


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_task_texts_labels(data_root: Path, benchmark: str, task: str):
    """Load texts and string labels for one task."""
    import json
    path = data_root / benchmark / task / "train.json"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    defs = data.get("Definition", [])
    definition = defs[0].strip() if isinstance(defs, list) and defs else (
        defs.strip() if isinstance(defs, str) else ""
    )

    if benchmark == "Long_Sequence":
        template = f"{definition}\n{{0}}\nOutput: " if definition else "{{0}}"
    else:
        template = (
            f"Definition: {definition}\n\n"
            "Now complete the following example -\n"
            "Input: {{0}}\nOutput: "
        ) if definition else "{{0}}"

    texts, labels = [], []
    for inst in data.get("Instances", []):
        if isinstance(inst, dict):
            inp = inst.get("input", "") or inst.get("text", "")
            out = inst.get("output", "")
        else:
            inp, out = str(inst), ""
        texts.append(template.format(inp))
        labels.append(out if isinstance(out, str) else str(out))
    return texts, labels


class TaskDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer,
                 max_in: int = 512, max_out: int = 64):
        self.enc = tokenizer(
            texts, max_length=max_in, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        self.dec = tokenizer(
            labels, max_length=max_out, truncation=True,
            padding="max_length", return_tensors="pt",
        )

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.dec["input_ids"][i],
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics_cls(preds_str: List[str], labels_str: List[str]) -> Dict[str, float]:
    correct = sum(
        p.strip().lower() == l.strip().lower()
        for p, l in zip(preds_str, labels_str)
    )
    return {"accuracy": correct / max(len(preds_str), 1)}


def compute_metrics_gen(preds_str: List[str], labels_str: List[str]) -> Dict[str, float]:
    try:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(preds_str, labels_str, avg=True)
        return {
            "rouge_l": scores["rouge-l"]["f"],
            "rouge_1": scores["rouge-1"]["f"],
            "rouge_2": scores["rouge-2"]["f"],
        }
    except Exception:
        def tokens(s):
            return set(s.strip().lower().split())
        em = sum(len(tokens(p) & tokens(l)) / max(len(tokens(l)), 1)
                 for p, l in zip(preds_str, labels_str))
        return {"exact_match": em / max(len(preds_str), 1)}


def _load_lora_into_model(model, lora_state: Dict[str, torch.Tensor]):
    for full_name, tensor in lora_state.items():
        parts = full_name.rsplit(".", 1)
        if len(parts) != 2:
            continue
        parent_path, param_name = parts
        module = model
        for part in parent_path.split("."):
            if not part:
                continue
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and hasattr(module, param_name):
            getattr(module, param_name).data.copy_(tensor)


def evaluate_on_tasks(
    model,
    tokenizer,
    data_root: Path,
    benchmark: str,
    tasks: List[str],
    batch_size: int = 8,
    max_in: int = 512,
    max_out: int = 64,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Evaluate model on ALL given tasks.
    Returns: (n×n score matrix, per-task metric dicts)
      mat[t, s] = performance on task s after training on task t
      mat entries for s > t are 0 (not yet trained)
    """
    n = len(tasks)
    mat = np.zeros((n, n))
    task_metrics = []

    CLS_TASKS = {"sst2", "imdb", "yelp", "amazon", "dbpedia", "agnews", "yahoo",
                 "boolq", "mnli", "wic", "cb", "copa", "qqp", "rte", "multirc"}

    for tid, tname in enumerate(tasks):
        texts, labels = load_task_texts_labels(data_root, benchmark, tname)
        dataset = TaskDataset(texts, labels, tokenizer, max_in=max_in, max_out=max_out)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                device_batch = {
                    k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                out_ids = model.generate(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    max_new_tokens=max_out,
                )
                preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                labs = tokenizer.batch_decode(device_batch["labels"], skip_special_tokens=True)
                all_preds.extend(preds)
                all_labels.extend(labs)

        is_cls = (benchmark == "Long_Sequence" and tname in CLS_TASKS) or \
                 (benchmark == "SuperNI" and "classification" in tname)

        if is_cls:
            metrics = compute_metrics_cls(all_preds, all_labels)
        else:
            metrics = compute_metrics_gen(all_preds, all_labels)

        main_metric = metrics.get("accuracy", metrics.get("rouge_l", metrics.get("exact_match", 0)))
        mat[tid, tid] = main_metric
        task_metrics.append(metrics)
        log.info(f"      Eval [{tname}@{benchmark}]: {main_metric:.4f}  {metrics}")

    return mat, task_metrics


# ═══════════════════════════════════════════════════════════════════════
# FULL CL EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_cl(
    method: str,
    method_dir: Path,
    model_name: str,
    tokenizer,
    data_root: Path,
    benchmark: str,
    tasks: List[str],
    rank: int,
    lora_alpha: int,
    use_trans_input: bool,
    chunk: int,
    batch_size: int,
    max_in: int = 512,
    max_out: int = 64,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Evaluate CL metrics: AP, FT, BWT by reloading LoRA checkpoints.

    BWT = mat[t,s] - mat[s,s]: performance on task s after training on t,
    minus performance on task s after training on s (positive = no forgetting).

    NOTE: This function reads lora_state.pt checkpoints from method_dir.
    If no checkpoints exist (cleaned up), returns zeros.
    """
    n = len(tasks)
    mat = np.zeros((n, n))
    all_task_metrics: List[Dict[str, float]] = []

    for t in range(n):
        eval_model = load_gainlora_model(
            model_name=model_name,
            rank=rank,
            lora_alpha=lora_alpha,
            task_id=t,
            use_trans_input=use_trans_input,
            chunk=chunk,
        )

        if method == "gainlora_root":
            for load_t in range(t + 1):
                ckpt = method_dir / f"task_{load_t:02d}_{tasks[load_t]}"
                lora_path = ckpt / "lora_state.pt"
                if not lora_path.exists():
                    continue
                state = torch.load(lora_path, map_location="cpu", weights_only=False)
                if load_t < t:
                    for full_name, tensor in state.items():
                        parts = full_name.rsplit(".", 1)
                        if len(parts) != 2:
                            continue
                        parent_path, param_name = parts
                        parent_parts = parent_path.rsplit(".", 2)
                        if len(parent_parts) != 3:
                            continue
                        base_path, attype, param_n = parent_parts
                        if attype not in ("previous_lora_weights_q", "previous_lora_weights_v"):
                            continue
                        try:
                            idx = int(base_path.rsplit(".", 1)[1])
                            attn_path = base_path.rsplit(".", 1)[0]
                            module = eval_model
                            for mp in attn_path.split("."):
                                if not mp:
                                    continue
                                module = getattr(module, mp, None)
                                if module is None:
                                    break
                            if module is not None:
                                layer = getattr(module, attype, None)
                                if layer is not None and idx < len(layer):
                                    getattr(layer[idx], param_n).data.copy_(tensor)
                        except (ValueError, IndexError, AttributeError):
                            pass
                else:
                    _load_lora_into_model(eval_model, state)
        else:
            ckpt = method_dir / f"task_{t:02d}_{tasks[t]}"
            lora_path = ckpt / "lora_state.pt"
            if lora_path.exists():
                state = torch.load(lora_path, map_location="cpu", weights_only=False)
                _load_lora_into_model(eval_model, state)

        eval_mat, eval_metrics = evaluate_on_tasks(
            eval_model, tokenizer, data_root, benchmark,
            tasks[:t+1], batch_size=batch_size,
            max_in=max_in, max_out=max_out,
        )
        for j in range(t + 1):
            mat[t, j] = eval_mat[t, j]
        all_task_metrics.append(eval_metrics[t])

        del eval_model
        gc.collect()
        torch.cuda.empty_cache()

    return mat, all_task_metrics


def _compute_cl_metrics(mat: np.ndarray, tasks: List[str]
                         ) -> Tuple[float, float, float, np.ndarray]:
    """
    Compute AP, FT, BWT from score matrix.
    Also returns intermediate_APs: list of AP after each task t.

    AP_t = mean(mat[t, :t+1]) — average performance after training task t
    AP   = AP_{last}         — AP after all tasks
    FT   = mean([mat[t,t] for t]) — forward transfer (diagonal mean)
    BWT  = mean([mat[t,s] - mat[s,s] for s < t < n]) — backward transfer
    """
    n = len(tasks)
    ap_per_task = []
    for t in range(n):
        # Only tasks 0..t have non-zero entries in row t
        ap_t = float(np.mean(mat[t, :t+1]))
        ap_per_task.append(ap_t)

    ap = ap_per_task[-1] if ap_per_task else 0.0
    ft = float(np.mean([mat[t, t] for t in range(n)]))

    bwt_scores = []
    for t in range(1, n):
        for s in range(t):
            bwt_scores.append(mat[t, s] - mat[s, s])
    bwt = float(np.mean(bwt_scores)) if bwt_scores else 0.0

    return ap, ft, bwt, np.array(ap_per_task)


def _log_score_matrix(mat: np.ndarray, tasks: List[str], title: str = ""):
    """Pretty-print the CL score matrix with task names and per-task AP."""
    n = len(tasks)
    if n == 0:
        return

    # Header
    task_cols = [f"{t[:8]:>8}" for t in tasks]
    header = " " * 10 + "".join(task_cols) + f"{'AP_t':>8}"
    log.info(f"    Score matrix {title}:")
    log.info(f"    {'Trained\\Seen':>10}" + "".join(task_cols) + f"{'AP_t':>8}")
    log.info(f"    {'─'*10}' + '─'*9*n + '─'*8")

    for t in range(n):
        row_label = f"after {tasks[t][:8]}"
        entries = []
        for s in range(n):
            if s <= t:
                entries.append(f"{mat[t,s]:>8.4f}")
            else:
                entries.append(f"{'—':>8}")
        ap_t = float(np.mean(mat[t, :t+1]))
        entries.append(f"{ap_t:>8.4f}")
        log.info(f"    {row_label:>10}" + "".join(entries))


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def _get_lora_named_params(model) -> List[Tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def _build_grad_dict(model) -> Dict[str, torch.Tensor]:
    return {
        n: p.grad.detach().clone()
        for n, p in _get_lora_named_params(model)
        if p.grad is not None
    }


def train_and_eval_method(
    method: str,
    method_dir: Path,
    model_name: str,
    tokenizer,
    data_root: Path,
    benchmark: str,
    tasks: List[str],
    epochs: int,
    rank: int,
    lora_alpha: int,
    lr: float,
    attn_lr: float,
    batch_size: int,
    gpm_steps: int,
    gpm_threshold: float,
    chunk: int,
    lambda_fsr: float,
    lambda_taa: float,
    lambda_sgr: float,
    eval_every: int,
    max_in: int = 512,
    max_out: int = 64,
) -> Dict:
    """
    Train all tasks sequentially for one method, then evaluate.
    Returns result dict with score matrix, AP/FT/BWT, and per-task AP progression.
    NO checkpoints are saved.
    """
    total_tasks = len(tasks)
    t0 = time.time()

    log.info(f"\n{'='*60}")
    log.info(f"  METHOD: {method}")
    log.info(f"  TASKS:  {' → '.join(tasks)}")
    log.info(f"  EPOCHS: {epochs}  |  RANK: {rank}  |  LR: {lr}")
    log.info(f"{'='*60}")

    # ── Init CL components ──────────────────────────────────────────────
    use_trans = (method == "gainlora_root")
    use_gpm   = method in ("gainlora_root", "inflora")
    use_fsr   = "fgcl_fsr" in method or method == "fgcl_taa"
    use_taa   = (method == "fgcl_taa")
    use_sgr   = (method == "fgcl_sgr")
    use_kffng = (method == "fgcl_kfng")

    gpm   = GPMAccumulator(threshold=gpm_threshold, chunk=chunk) if use_gpm else None
    fsr   = FSR(lam=lambda_fsr) if use_fsr else None
    taa   = TAA(lam=lambda_taa) if use_taa else None
    sgr   = SGR(lam=lambda_sgr) if use_sgr else None

    # SGR: accumulate previous subspaces from LoRA states (loaded from disk)
    if sgr is not None:
        for task_id, task_name in enumerate(tasks):
            ckpt = method_dir / f"task_{task_id:02d}_{task_name}"
            lora_path = ckpt / "lora_state.pt"
            if lora_path.exists():
                state = torch.load(lora_path, map_location="cpu", weights_only=False)
                for name, sd in state.items():
                    if "lora_A" in name:
                        sgr.register(name, sd)

    prev_dirs: List[Path] = []

    # ── Per-task training ───────────────────────────────────────────────
    for task_id, task_name in enumerate(tasks):
        log.info(f"\n  [Task {task_id+1}/{total_tasks}] {task_name} ({method})")
        base_model = load_gainlora_model(
            model_name=model_name,
            rank=rank,
            lora_alpha=lora_alpha,
            task_id=task_id,
            use_trans_input=use_trans,
            chunk=chunk,
        )

        tp = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        tt = sum(p.numel() for p in base_model.parameters())
        log.info(f"    Trainable: {tp:,}/{tt:,} ({100*tp/tt:.2f}%)")

        lora_params = [
            p for n, p in base_model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]
        extra_params = [
            p for n, p in base_model.named_parameters()
            if ("trans_input" in n or "prompt_key" in n) and p.requires_grad
        ]
        if use_kffng:
            all_params = [{"params": lora_params, "lr": lr}]
            if extra_params:
                all_params.append({"params": extra_params, "lr": attn_lr})
            optimizer = KFFNGOptimizer(all_params, lr=lr, f_ema=0.99)
        else:
            optimizer = torch.optim.AdamW([
                {"params": lora_params, "lr": lr},
                *([{"params": extra_params, "lr": attn_lr}] if extra_params else []),
            ], weight_decay=0.01)

        texts, labels = load_task_texts_labels(data_root, benchmark, task_name)
        dataset = TaskDataset(texts, labels, tokenizer)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=False,
        )
        n_cls = len(set(labels))
        log.info(f"    {len(texts)} samples, {n_cls} classes, {epochs} epochs")

        # ── GPM collection ───────────────────────────────────────────────
        if gpm is not None and gpm_steps > 0:
            log.info(f"    GPM: collecting ({gpm_steps} steps)")
            gpm.collect(base_model, loader, n_steps=gpm_steps)
            prev_feats = gpm.feature_list[-1] if gpm.feature_list else None
            gpm.build(base_model, task_id=task_id, total_tasks=total_tasks,
                      prev_features=prev_feats)

        # ── Load accumulated CL state from last checkpoint ───────────────
        if prev_dirs:
            last_ckpt = prev_dirs[-1]
            if gpm is not None:
                gp = last_ckpt / "gpm_state.pt"
                if gp.exists():
                    # Rebuild GPM from saved features
                    data = torch.load(gp, map_location="cpu", weights_only=False)
                    gpm.feature_list = data.get("features", [])

        # ── Training ─────────────────────────────────────────────────────
        base_model.train()
        step = 0

        for epoch in range(epochs):
            epoch_ce_loss, n_batches = 0.0, 0
            for batch in loader:
                optimizer.zero_grad()

                device_batch = {
                    k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }

                out = base_model(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    labels=device_batch["labels"],
                )
                loss_ce = out.loss
                total_loss = loss_ce

                if sgr is not None:
                    sgr_l, _ = sgr.loss(base_model)
                    if sgr_l.item() > 0:
                        total_loss = total_loss + sgr_l

                total_loss.backward()
                grad_dict = _build_grad_dict(base_model)

                if fsr is not None and grad_dict:
                    fsr.update(grad_dict)
                    if step > 0 and step % 200 == 0:
                        fsr.build_bases()
                    fsr_l, fsr_l_val = fsr.loss(grad_dict)
                    if fsr_l_val > 0:
                        optimizer.zero_grad()
                        out2 = base_model(
                            input_ids=device_batch["input_ids"],
                            attention_mask=device_batch["attention_mask"],
                            labels=device_batch["labels"],
                        )
                        (out2.loss + fsr_l).backward()
                        grad_dict2 = _build_grad_dict(base_model)
                        if grad_dict2:
                            fsr.update(grad_dict2)
                            if use_kffng:
                                KFFNGOptimizer.update_fisher(
                                    optimizer, _get_lora_named_params(base_model), grad_dict2
                                )

                if use_kffng and grad_dict:
                    KFFNGOptimizer.update_fisher(
                        optimizer, _get_lora_named_params(base_model), grad_dict
                    )

                if taa is not None and taa.task_vectors:
                    taa_l, taa_l_val = taa.loss(base_model, grad_dict)
                    if taa_l_val > 0:
                        optimizer.zero_grad()
                        out3 = base_model(
                            input_ids=device_batch["input_ids"],
                            attention_mask=device_batch["attention_mask"],
                            labels=device_batch["labels"],
                        )
                        (out3.loss + taa_l).backward()
                        grad_dict3 = _build_grad_dict(base_model)
                        if grad_dict3 and fsr is not None:
                            fsr.update(grad_dict3)

                if gpm is not None and gpm.feature_list and task_id > 0:
                    gpm.project(base_model)

                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_ce_loss += loss_ce.item()
                n_batches += 1
                step += 1

            avg_loss = epoch_ce_loss / max(n_batches, 1)
            if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                log.info(f"    Epoch {epoch+1}/{epochs} loss: {avg_loss:.4f}")

        # ── Save LoRA checkpoint (kept for eval only) ─────────────────────
        ckpt_dir = method_dir / f"task_{task_id:02d}_{task_name}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        lora_state = {
            n: p.detach().cpu()
            for n, p in base_model.named_parameters()
            if "lora_" in n
        }
        torch.save(lora_state, ckpt_dir / "lora_state.pt")

        # Register TAA task vector
        if taa is not None:
            taa.register(task_id, base_model)

        log.info(f"    ✓ {task_name} done ({epochs} epochs, loss={avg_loss:.4f})")

        # ── INTERMEDIATE EVAL (every eval_every tasks or last task) ──────
        if eval_every > 0 and (task_id % eval_every == eval_every - 1 or task_id == total_tasks - 1):
            log.info(f"\n    [Intermediate eval after task {task_id+1}/{total_tasks}]")
            interm_mat, _ = evaluate_cl(
                method=method, method_dir=method_dir,
                model_name=model_name, tokenizer=tokenizer,
                data_root=data_root, benchmark=benchmark,
                tasks=tasks, rank=rank, lora_alpha=lora_alpha,
                use_trans_input=use_trans, chunk=chunk,
                batch_size=batch_size,
            )
            interm_ap, interm_ft, interm_bwt, interm_aps = _compute_cl_metrics(interm_mat, tasks)
            _log_score_matrix(interm_mat, tasks, f"{method} after {task_name}")
            log.info(f"    Intermediate: AP={interm_ap:.4f}  FT={interm_ft:.4f}  BWT={interm_bwt:+.4f}  "
                     f"(tasks 0..{task_id})")

        prev_dirs.append(ckpt_dir)

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    # ── FINAL EVALUATION ─────────────────────────────────────────────────
    log.info(f"\n  Final CL evaluation for {method}...")
    score_mat, _ = evaluate_cl(
        method=method, method_dir=method_dir,
        model_name=model_name, tokenizer=tokenizer,
        data_root=data_root, benchmark=benchmark,
        tasks=tasks, rank=rank, lora_alpha=lora_alpha,
        use_trans_input=use_trans, chunk=chunk,
        batch_size=batch_size,
    )

    ap, ft, bwt, ap_per_task = _compute_cl_metrics(score_mat, tasks)
    elapsed = time.time() - t0

    # ── Score matrix summary ─────────────────────────────────────────────
    _log_score_matrix(score_mat, tasks, f"{method} FINAL")

    log.info(f"\n{'='*60}")
    log.info(f"  RESULTS ({method})  |  {total_tasks} tasks  |  {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log.info(f"  AP   = {ap:.4f}   (mean perf after last task, tasks 0..{total_tasks-1})")
    log.info(f"  FT   = {ft:.4f}   (mean diagonal = forward transfer)")
    log.info(f"  BWT  = {bwt:+.4f}  (backward transfer, 0 = no forgetting)")
    log.info(f"  AP per task: {' '.join(f'{x:.3f}' for x in ap_per_task)}")
    log.info(f"{'='*60}")

    per_task_rows = [
        {"task_id": t, "task_name": tasks[t], "score_row": score_mat[t].tolist()}
        for t in range(total_tasks)
    ]

    result = {
        "method": method,
        "tasks": tasks,
        "epochs": epochs,
        "elapsed_seconds": elapsed,
        "AP": ap,
        "FT": ft,
        "BWT": bwt,
        "AP_per_task": ap_per_task.tolist(),
        "score_matrix": score_mat.tolist(),
        "per_task": per_task_rows,
        "config": {
            "rank": rank, "lora_alpha": lora_alpha, "lr": lr,
            "gpm_steps": gpm_steps, "gpm_threshold": gpm_threshold,
            "lambda_fsr": lambda_fsr, "lambda_taa": lambda_taa, "lambda_sgr": lambda_sgr,
        },
    }

    # Save JSON
    with open(method_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # Clean up checkpoints (save disk — we only needed them for eval)
    for task_id, task_name in enumerate(tasks):
        ckpt = method_dir / f"task_{task_id:02d}_{task_name}"
        if not ckpt.exists():
            continue
        for fname in [
            "gpm_state.pt",
            "fsr_state.pt",
            "taa_state.pt",
            "sgr_state.pt",
            "trans_input.pt",
            "prompt_key.pt",
        ]:
            fp = ckpt / fname
            if fp.exists():
                fp.unlink()

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FGCL C2 Experiment Framework — Training Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python exp_fgcl.py                                    # ALL phases, all benchmarks
  python exp_fgcl.py --phase T4 --benchmark Long_Sequence
  python exp_fgcl.py --model google/flan-t5-large --benchmark SuperNI
  python exp_fgcl.py --phase T1 --benchmark Long_Sequence --tasks sst2 imdb mnli
  python exp_fgcl.py --eval_every 1                    # eval after EVERY task
  python exp_fgcl.py --eval_every 0                    # skip intermediate eval
        """,
    )
    p.add_argument("--phase", nargs="+", default=None,
                   help="Phase(s): T1 T2 T3 T4. Default: all.")
    p.add_argument("--model", default="google/flan-t5-large",
                   help="HF model name. Default: google/flan-t5-large.")
    p.add_argument("--benchmark", nargs="+", default=None,
                   help="Benchmark(s): Long_Sequence SuperNI. Default: all.")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Specific tasks (subset of benchmark). Default: all.")
    p.add_argument("--data_root", type=Path, default=Path("CL_Benchmark"),
                   help="Path to CL_Benchmark/ directory.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override default epochs for all phases.")
    p.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    p.add_argument("--alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--attn_lr", type=float, default=DEFAULT_ATTN_LR)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--gpm_steps", type=int, default=DEFAULT_GPM_STEPS)
    p.add_argument("--gpm_threshold", type=float, default=DEFAULT_GPM_THRESH)
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--lambda_fsr", type=float, default=DEFAULT_LAMBDA_FSR)
    p.add_argument("--lambda_taa", type=float, default=DEFAULT_LAMBDA_TAA)
    p.add_argument("--lambda_sgr", type=float, default=DEFAULT_LAMBDA_SGR)
    p.add_argument("--output_dir", default="results_fgcl",
                   help="Output root directory. Default: results_fgcl/")
    p.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY,
                   help="Eval every N tasks during training (0=final only). Default: 3.")
    return p.parse_args()


def main():
    args = parse_args()

    phases = args.phase or ALL_PHASES
    benchmarks = args.benchmark or list(BENCHMARK_TASKS.keys())

    runs = []
    for bm in benchmarks:
        avail = BENCHMARK_TASKS.get(bm, [])
        if not avail:
            log.warning(f"Unknown benchmark: {bm}")
            continue
        for phase in phases:
            n = PHASE_TASK_COUNTS.get(phase)
            tasks = avail[:n] if n else avail
            if args.tasks:
                tasks = [t for t in args.tasks if t in avail]
            if tasks:
                runs.append((phase, bm, tasks))

    if not runs:
        log.error("No valid runs. Check --phase and --benchmark arguments.")
        sys.exit(1)

    log.info(f"\nRun matrix: {len(runs)} experiment(s)")
    for phase, bm, tasks in runs:
        log.info(f"  {phase} | {bm} | {len(tasks)} tasks: {' → '.join(tasks[:3])}"
                 f"{' ...' if len(tasks) > 3 else ''}")

    output_root = Path(args.output_dir)
    all_results = {}

    for phase, benchmark, tasks in runs:
        key = f"{args.model}/{benchmark}/{phase}"
        methods = ALL_PHASE_METHODS.get(phase, ["standard_lora"])
        epochs = args.epochs if args.epochs else DEFAULT_EPOCHS.get(phase, 3)

        log.info(f"\n{'#'*70}")
        log.info(f"# PHASE {phase}: {PHASE_DESCS[phase]}")
        log.info(f"# MODEL: {args.model}")
        log.info(f"# BENCHMARK: {benchmark}")
        log.info(f"# TASKS ({len(tasks)}): {' → '.join(tasks)}")
        log.info(f"# METHODS: {methods}")
        log.info(f"# EPOCHS: {epochs}  RANK: {args.rank}  LR: {args.lr}  EVAL_EVERY: {args.eval_every}")
        log.info(f"{'#'*70}")

        phase_dir = output_root / f"phase_{phase}" / args.model.split("/")[-1] / benchmark
        phase_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        phase_results = {}

        for method in methods:
            method_dir = phase_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            result = train_and_eval_method(
                method=method,
                method_dir=method_dir,
                model_name=args.model,
                tokenizer=tokenizer,
                data_root=args.data_root,
                benchmark=benchmark,
                tasks=tasks,
                epochs=epochs,
                rank=args.rank,
                lora_alpha=args.alpha,
                lr=args.lr,
                attn_lr=args.attn_lr,
                batch_size=args.batch_size,
                gpm_steps=args.gpm_steps,
                gpm_threshold=args.gpm_threshold,
                chunk=args.chunk,
                lambda_fsr=args.lambda_fsr,
                lambda_taa=args.lambda_taa,
                lambda_sgr=args.lambda_sgr,
                eval_every=args.eval_every,
            )
            phase_results[method] = result

        # ── Phase summary ──────────────────────────────────────────────
        log.info(f"\n{'═'*70}")
        log.info(f"PHASE {phase} RESULTS — {args.model} / {benchmark}")
        log.info(f"  {'Method':<20} {'AP':>8} {'FT':>8} {'BWT':>10} {'Time':>10}")
        log.info(f"  {'─'*60}")
        for m, r in phase_results.items():
            t = r.get("elapsed_seconds", 0)
            log.info(f"  {m:<20} {r['AP']:>8.4f} {r['FT']:>8.4f} "
                     f"{r['BWT']:>10.4f}  {t:>7.0f}s")

        combined = {
            "phase": phase, "model": args.model, "benchmark": benchmark,
            "tasks": tasks, "methods": list(phase_results.keys()),
            "description": PHASE_DESCS[phase],
            "results": phase_results,
        }
        out_path = phase_dir / "all_results.json"
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)
        log.info(f"\n  Saved: {out_path}")

        all_results[key] = phase_results

    # ── Grand summary ─────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"GRAND SUMMARY — {len(runs)} experiment(s)")
    print(f"{'═'*70}")
    for key, results in all_results.items():
        print(f"\n  [{key}]")
        print(f"  {'Method':<20} {'AP':>8} {'FT':>8} {'BWT':>10}")
        print(f"  {'─'*50}")
        for m, r in sorted(results.items()):
            print(f"  {m:<20} {r['AP']:>8.4f} {r['FT']:>8.4f} {r['BWT']:>10.4f}")
    print(f"\n{'═'*70}")
    print(f"Results: {output_root}/phase_{{T1,T2,T3,T4}}/{{model}}/{{benchmark}}/all_results.json")


if __name__ == "__main__":
    main()
