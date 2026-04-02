#!/usr/bin/env python3
"""
Experiment E0: TARA Rank Sweep — Validate Adaptive Rank Allocation
===================================================================
Tests whether LoRA accuracy saturates at r ≈ TGC_eff (gradient PaR).

Protocol:
1. Gradient probing: compute actual Σ_grad and TGC_eff = PaR(Σ_grad)
2. Train LoRA with varying ranks r ∈ {2, 4, 8, 16, 32}
3. Measure: final loss, eval loss, accuracy per rank
4. Verify: accuracy improvement from r to 2r becomes negligible when r ≥ TGC_eff
5. Compare TARA's predicted rank vs empirical optimal rank

Usage:
  python exp_e0_tara_rank_sweep.py \
    --model_name google/flan-t5-large \
    --data_dir ../improve_gainlora/CL_Benchmark \
    --task mnli --benchmark Long_Sequence \
    --ranks 2,4,8,16,32 --n_epochs 3

Output: results/e0_tara_rank_<model>_<task>.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, math, copy, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def participation_ratio(eigvals):
    eigvals = np.maximum(np.asarray(eigvals, dtype=np.float64), 0)
    s = eigvals.sum()
    if s < 1e-15:
        return 0.0
    return float(s ** 2 / (eigvals ** 2).sum())


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_task_data_simple(data_dir, benchmark, task, max_samples=5000):
    json_path = Path(data_dir) / benchmark / task / "train.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Data not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    instances = data.get("Instances", [])
    defn_list = data.get("Definition", [])
    definition = defn_list[0] if defn_list else ""
    samples = []
    for inst in instances[:max_samples]:
        text = inst.get("input", "")
        label = inst.get("output", "")
        if isinstance(label, list):
            label = label[0] if label else ""
        instruction = f"{definition}\n{text}\nOutput: "
        samples.append({"input": instruction, "label": label})
    return samples


def load_eval_data(data_dir, benchmark, task, max_samples=500):
    for split_name in ["dev", "test"]:
        json_path = Path(data_dir) / benchmark / task / f"{split_name}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            instances = data.get("Instances", [])
            defn_list = data.get("Definition", [])
            definition = defn_list[0] if defn_list else ""
            samples = []
            for inst in instances[:max_samples]:
                text = inst.get("input", "")
                label = inst.get("output", "")
                if isinstance(label, list):
                    label = label[0] if label else ""
                instruction = f"{definition}\n{text}\nOutput: "
                samples.append({"input": instruction, "label": label})
            if samples:
                return samples
    return None


class SimpleTextDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn_t5(batch, tokenizer, max_source_length=512, max_target_length=50):
    inputs_text = [s["input"] for s in batch]
    labels_text = [s["label"] for s in batch]
    model_inputs = tokenizer(inputs_text, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_source_length)
    labels = tokenizer(labels_text, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_target_length)
    label_ids = labels.input_ids
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = label_ids
    return model_inputs


# ═══════════════════════════════════════════════════════════════════════
# LoRA injection
# ═══════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def inject_lora(model, r, alpha=1.0, target_modules=None):
    if target_modules is None:
        target_modules = ["q", "v"]
    lora_modules = []
    is_t5 = hasattr(model, 'encoder')
    if is_t5:
        for block in model.encoder.block:
            attn = block.layer[0].SelfAttention
            for name in target_modules:
                if hasattr(attn, name):
                    base_linear = getattr(attn, name)
                    lora_linear = LoRALinear(base_linear, r, alpha)
                    setattr(attn, name, lora_linear)
                    lora_modules.append(lora_linear)
        for block in model.decoder.block:
            attn = block.layer[0].SelfAttention
            for name in target_modules:
                if hasattr(attn, name):
                    base_linear = getattr(attn, name)
                    lora_linear = LoRALinear(base_linear, r, alpha)
                    setattr(attn, name, lora_linear)
                    lora_modules.append(lora_linear)
    else:
        for layer in model.model.layers:
            attn = layer.self_attn
            for name in ["q_proj", "v_proj"]:
                if hasattr(attn, name):
                    base_linear = getattr(attn, name)
                    lora_linear = LoRALinear(base_linear, r, alpha)
                    setattr(attn, name, lora_linear)
                    lora_modules.append(lora_linear)
    return lora_modules


# ═══════════════════════════════════════════════════════════════════════
# Gradient probing for TARA
# ═══════════════════════════════════════════════════════════════════════

def probe_gradient_covariance(model, tokenizer, samples, device,
                               n_batches=50, batch_size=8, target_layer_idx=0):
    """Probe gradient covariance at target layer's Q projection weight."""
    model.eval()
    is_t5 = hasattr(model, 'encoder')

    if is_t5:
        n_layers = len(model.encoder.block)
        if target_layer_idx >= n_layers:
            target_layer_idx = n_layers - 1
        q_module = model.encoder.block[target_layer_idx].layer[0].SelfAttention.q
    else:
        n_layers = len(model.model.layers)
        if target_layer_idx >= n_layers:
            target_layer_idx = n_layers - 1
        q_module = model.model.layers[target_layer_idx].self_attn.q_proj

    weight_param = q_module.weight
    d_out, d_in = weight_param.shape

    grad_outer = np.zeros((d_in, d_in), dtype=np.float64)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(samples))
    batch_count = 0
    i = 0

    while batch_count < n_batches and i < len(indices):
        batch_idx = indices[i:i + batch_size]
        i += batch_size
        if len(batch_idx) == 0:
            break

        batch_texts = [samples[j]["input"] for j in batch_idx]
        batch_labels = [samples[j]["label"] for j in batch_idx]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        if is_t5:
            labels = tokenizer(batch_labels, return_tensors="pt", padding=True,
                             truncation=True, max_length=50).input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        wg = weight_param.grad
        if wg is not None:
            wg_np = wg.float().cpu().numpy()  # (d_out, d_in)
            grad_outer += wg_np.T @ wg_np

        batch_count += 1

    cov_grad = grad_outer / max(batch_count, 1)
    print(f"  Probed {batch_count} batches, d_in={d_in}")
    return cov_grad, d_in


# ═══════════════════════════════════════════════════════════════════════
# Training + Evaluation
# ═══════════════════════════════════════════════════════════════════════

def train_with_rank(model_name, tokenizer, samples, eval_samples, device,
                    rank, n_epochs=3, batch_size=8, lr=1e-4):
    """Train LoRA with a specific rank and return metrics."""
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

    is_t5 = "t5" in model_name.lower()
    if is_t5:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    target_modules = ["q", "v"] if is_t5 else ["q_proj", "v_proj"]
    lora_modules = inject_lora(model, rank, alpha=1.0, target_modules=target_modules)
    model.to(device)

    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    for lm in lora_modules:
        lm.lora_A.requires_grad_(True)
        lm.lora_B.requires_grad_(True)

    trainable_params = []
    for lm in lora_modules:
        trainable_params.extend([lm.lora_A, lm.lora_B])

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  r={rank}: {n_trainable:,} trainable params")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    dataset = SimpleTextDataset(samples)
    collate = partial(collate_fn_t5, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate, drop_last=True)

    loss_curve = []
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)
        loss_curve.append(round(avg_loss, 4))
        print(f"    Epoch {epoch}: loss={avg_loss:.4f}")

    # Eval loss
    eval_loss = None
    if eval_samples:
        model.eval()
        total_loss = 0.0
        n_eval = 0
        for i in range(0, len(eval_samples), batch_size):
            batch_data = eval_samples[i:i + batch_size]
            if not batch_data:
                break
            inputs_text = [s["input"] for s in batch_data]
            labels_text = [s["label"] for s in batch_data]
            inputs = tokenizer(inputs_text, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(device)
            if is_t5:
                labels = tokenizer(labels_text, return_tensors="pt", padding=True,
                                 truncation=True, max_length=50).input_ids.to(device)
                labels[labels == tokenizer.pad_token_id] = -100
                inputs["labels"] = labels
            with torch.no_grad():
                outputs = model(**inputs)
                total_loss += outputs.loss.item()
            n_eval += 1
        eval_loss = total_loss / max(n_eval, 1)

    # Accuracy
    accuracy = None
    if eval_samples:
        model.eval()
        correct = 0
        total = 0
        for i in range(0, len(eval_samples), batch_size):
            batch_data = eval_samples[i:i + batch_size]
            if not batch_data:
                break
            inputs_text = [s["input"] for s in batch_data]
            gold_labels = [s["label"].strip().lower() for s in batch_data]
            inputs = tokenizer(inputs_text, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(device)
            with torch.no_grad():
                if is_t5:
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=50, do_sample=False,
                    )
                    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                else:
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                    input_len = inputs["input_ids"].shape[1]
                    preds = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            for pred, gold in zip(preds, gold_labels):
                if pred.strip().lower() == gold:
                    correct += 1
                total += 1
        accuracy = correct / max(total, 1)

    # Effective rank of ΔW
    eff_ranks = []
    for lm in lora_modules:
        dW = (lm.lora_B @ lm.lora_A).detach().cpu().numpy()
        _, S, _ = np.linalg.svd(dW, full_matrices=False)
        eff_ranks.append(participation_ratio(S ** 2))

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "rank": rank,
        "n_trainable_params": n_trainable,
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1],
        "eval_loss": round(eval_loss, 4) if eval_loss is not None else None,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "mean_effective_rank": round(float(np.mean(eff_ranks)), 2),
        "effective_ranks": [round(r, 2) for r in eff_ranks],
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E0: TARA Rank Sweep")
    parser.add_argument("--model_name", default="google/flan-t5-large")
    parser.add_argument("--data_dir", required=True, help="Path to CL_Benchmark/")
    parser.add_argument("--task", default="mnli")
    parser.add_argument("--benchmark", default="Long_Sequence")
    parser.add_argument("--ranks", default="2,4,8,16,32",
                       help="Comma-separated LoRA ranks to sweep")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_probe_batches", type=int, default=50)
    parser.add_argument("--target_layer", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_train_samples", type=int, default=5000)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    is_t5 = "t5" in args.model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print(f"\nLoading data: {args.benchmark}/{args.task}")
    samples = load_task_data_simple(args.data_dir, args.benchmark, args.task, args.max_train_samples)
    eval_samples = load_eval_data(args.data_dir, args.benchmark, args.task)
    print(f"  Train: {len(samples)}, Eval: {len(eval_samples) if eval_samples else 0}")

    # ---- Phase 0: Gradient Probing for TARA prediction ----
    print(f"\n{'='*60}")
    print(f"Phase 0: Gradient Probing for TARA Rank Prediction")
    print(f"{'='*60}")

    if is_t5:
        model_probe = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    else:
        model_probe = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model_probe.to(device)

    t0 = time.time()
    cov_grad, d_in = probe_gradient_covariance(
        model_probe, tokenizer, samples, device,
        n_batches=args.n_probe_batches, batch_size=args.batch_size,
        target_layer_idx=args.target_layer
    )
    t_probe = time.time() - t0
    print(f"  Probing time: {t_probe:.1f}s")

    eigvals = np.linalg.eigvalsh(cov_grad)
    eigvals = np.sort(eigvals)[::-1]
    eigvals_pos = np.maximum(eigvals, 0)

    tgc_eff = participation_ratio(eigvals_pos)
    tara_rank = max(2, min(32, int(np.ceil(tgc_eff))))

    # Cumulative variance thresholds
    cumvar = np.cumsum(eigvals_pos)
    total = cumvar[-1] if cumvar[-1] > 1e-15 else 1.0
    r90 = int(np.searchsorted(cumvar / total, 0.90)) + 1
    r95 = int(np.searchsorted(cumvar / total, 0.95)) + 1
    r99 = int(np.searchsorted(cumvar / total, 0.99)) + 1

    print(f"\n  TARA Analysis:")
    print(f"    TGC_eff (gradient PaR)     = {tgc_eff:.2f}")
    print(f"    TARA recommended rank      = {tara_rank}")
    print(f"    r for 90% variance capture = {r90}")
    print(f"    r for 95% variance capture = {r95}")
    print(f"    r for 99% variance capture = {r99}")

    del model_probe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Phase 1: Rank Sweep ----
    ranks = sorted([int(r) for r in args.ranks.split(",")])
    print(f"\n{'='*60}")
    print(f"Phase 1: Rank Sweep — r ∈ {ranks}")
    print(f"{'='*60}")

    rank_results = {}
    for rank in ranks:
        print(f"\n--- Rank r={rank} ---")
        t0 = time.time()
        result = train_with_rank(
            args.model_name, tokenizer, samples, eval_samples, device,
            rank=rank, n_epochs=args.n_epochs,
            batch_size=args.batch_size, lr=args.lr
        )
        result["time"] = round(time.time() - t0, 1)
        rank_results[str(rank)] = result

        print(f"  Final loss={result['final_loss']:.4f}  "
              f"Eval loss={result['eval_loss']}  "
              f"Acc={result['accuracy']}  "
              f"Eff rank={result['mean_effective_rank']:.2f}  "
              f"Time={result['time']:.0f}s")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY: Rank Sweep Results")
    print(f"{'='*60}")
    print(f"  TARA predicted optimal rank: {tara_rank} (TGC_eff={tgc_eff:.2f})")
    print(f"\n  {'Rank':<6} {'Loss':<10} {'Eval Loss':<12} {'Accuracy':<10} {'Eff Rank':<10} {'Time':<8}")
    print(f"  {'-'*56}")

    for rank in ranks:
        res = rank_results[str(rank)]
        acc_str = f"{res['accuracy']:.4f}" if res['accuracy'] is not None else "N/A"
        eval_str = f"{res['eval_loss']:.4f}" if res['eval_loss'] is not None else "N/A"
        print(f"  {rank:<6} {res['final_loss']:<10.4f} {eval_str:<12} {acc_str:<10} "
              f"{res['mean_effective_rank']:<10.2f} {res['time']:.0f}s")

    # Find saturation point
    accuracies = [(int(r), res["accuracy"]) for r, res in rank_results.items()
                  if res["accuracy"] is not None]
    if len(accuracies) >= 2:
        accuracies.sort()
        best_rank, best_acc = max(accuracies, key=lambda x: x[1])
        # Find smallest rank within 1% of best accuracy
        threshold = best_acc - 0.01
        saturation_rank = min(r for r, a in accuracies if a >= threshold)

        print(f"\n  Best rank: r={best_rank} (acc={best_acc:.4f})")
        print(f"  Saturation rank (within 1% of best): r={saturation_rank}")
        print(f"  TARA predicted: r={tara_rank}")
        print(f"  Match: {'YES ✓' if abs(saturation_rank - tara_rank) <= tara_rank * 0.5 else 'PARTIAL' if abs(saturation_rank - tara_rank) <= tara_rank else 'NO ✗'}")
    else:
        saturation_rank = None
        best_rank = None
        best_acc = None

    # Marginal accuracy gain analysis
    print(f"\n  Marginal accuracy gain (r → 2r):")
    for i in range(len(ranks) - 1):
        r1, r2 = ranks[i], ranks[i + 1]
        a1 = rank_results[str(r1)].get("accuracy")
        a2 = rank_results[str(r2)].get("accuracy")
        if a1 is not None and a2 is not None:
            gain = a2 - a1
            marker = " ← diminishing" if gain < 0.005 else ""
            print(f"    r={r1}→{r2}: Δacc = {gain:+.4f}{marker}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model_name.split("/")[-1]
    out_path = os.path.join(args.output_dir, f"e0_tara_rank_{model_short}_{args.task}.json")

    all_results = {
        "experiment": "E0_tara_rank_sweep",
        "model": args.model_name,
        "task": args.task,
        "benchmark": args.benchmark,
        "target_layer": args.target_layer,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "tara_analysis": {
            "tgc_eff": round(tgc_eff, 2),
            "tara_recommended_rank": tara_rank,
            "r90_variance": r90,
            "r95_variance": r95,
            "r99_variance": r99,
            "eigvals_top32": eigvals_pos[:32].tolist(),
        },
        "rank_results": rank_results,
        "verdict": {
            "saturation_rank": saturation_rank,
            "best_rank": best_rank,
            "best_accuracy": round(best_acc, 4) if best_acc is not None else None,
            "tara_rank": tara_rank,
            "tara_match": (abs(saturation_rank - tara_rank) <= tara_rank * 0.5
                          if saturation_rank is not None else None),
        },
    }

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
