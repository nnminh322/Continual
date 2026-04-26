#!/usr/bin/env python3
"""Add comprehensive SRT debug logging to LLaMA GainLoRA codebase."""

import os, re, sys

# ──────────────────────────────────────────────────────────────────────────────
# 1. ENHANCE srt_router.py: add route_debug()
# ──────────────────────────────────────────────────────────────────────────────
router_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_gainlora/src/srt_router.py"

with open(router_path, "r") as f:
    content = f.read()

old_route_end = """        del H
        nearest_idx = np.argmin(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])
        return nearest_task, dists

    @property"""

route_debug_method = '''
    def route_debug(self, h: np.ndarray) -> dict:
        """
        Route with full debug info: all distances, top-2 predictions, confidence.

        Returns:
            dict with keys: task_ids, dists, nearest_task, confidence_ratio,
                            second_task, n_tasks, task_id_list
        """
        if h.ndim == 1:
            h = h.reshape(1, -1)

        n_sample, d_h = h.shape
        T = len(self.centroids)

        if T == 0:
            raise RuntimeError("PooledMahalanobisRouter: no tasks registered.")

        # Move to GPU
        H = torch.from_numpy(h.astype(np.float32)).to(self.dev)
        Sinv = self._Sigma_inv_t
        dists = np.zeros((n_sample, T), dtype=np.float64)

        for i, mu_t_np in enumerate(self.centroids):
            mu_t_t = torch.from_numpy(mu_t_np.astype(np.float32)).to(self.dev)
            diff = H - mu_t_t
            diff_Sinv = diff @ Sinv
            dists[:, i] = (diff * diff_Sinv).sum(dim=1).cpu().numpy()

        del H
        nearest_idx = np.argmin(dists, axis=1)
        task_ids_ordered = list(self._signatures_by_id.keys())
        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])

        # Confidence: ratio of 2nd-best distance to best distance
        sorted_idx = np.argsort(dists, axis=1)
        best_idx = sorted_idx[:, 0]
        second_idx = sorted_idx[:, 1]
        best_d = dists[np.arange(n_sample), best_idx]
        second_d = dists[np.arange(n_sample), second_idx]
        # Confidence = (second - first) / first = margin / nearest distance
        with np.errstate(divide='ignore', invalid='ignore'):
            conf = (second_d - best_d) / (best_d + 1e-10)
            conf = np.where(np.isfinite(conf), conf, 999.0)

        return {
            "task_ids": nearest_task,
            "dists": dists,
            "nearest_task": nearest_task,
            "second_task": np.array([task_ids_ordered[i] for i in second_idx]),
            "best_dist": best_d,
            "second_dist": second_d,
            "confidence_ratio": conf,
            "n_tasks": T,
            "task_id_list": task_ids_ordered,
        }

    @property'''

content = content.replace(old_route_end, "        del H\n        nearest_idx = np.argmin(dists, axis=1)\n        task_ids_ordered = list(self._signatures_by_id.keys())\n        nearest_task = np.array([task_ids_ordered[i] for i in nearest_idx])\n        return nearest_task, dists\n" + route_debug_method)

with open(router_path, "w") as f:
    f.write(content)
print("srt_router.py: route_debug() added")

# ──────────────────────────────────────────────────────────────────────────────
# 2. ENHANCE llama_gainlora.py: richer SRT debug in LlamaModel.forward()
# ──────────────────────────────────────────────────────────────────────────────
llama_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_gainlora/src/llama_gainlora.py"

with open(llama_path, "r") as f:
    content = f.read()

# Replace the existing SRT DEBUG block with a much richer version
old_debug_block = """                # ── SRT DEBUG LOGGING ────────────────────────────────────
                if self.srt_debug:
                    tok = getattr(self, '_tokenizer', None)
                    pad_id = getattr(self, '_pad_token_id', 0)
                    # Decode source tokens of first sample in batch
                    sid = source_input_ids[0].cpu()
                    mask = (sid != pad_id).cpu()
                    nonpad = sid[mask]
                    seq_len = nonpad.size(0)
                    decoded = ''
                    if tok is not None and seq_len > 0:
                        decoded = tok.decode(nonpad[:50], skip_special_tokens=True)[:150].replace('\\n', ' ')
                    self._srt_debug_log.append({
                        'batch_size': batch_size,
                        'n_slots': n_slots,
                        'srt_preds': srt_preds.tolist() if hasattr(srt_preds, 'tolist') else list(srt_preds),
                        'slot_idxs': [
                            min(self.srt_task_id_to_idx.get(tid, 0), n_slots - 1)
                            for tid in (srt_preds.tolist() if hasattr(srt_preds, 'tolist') else list(srt_preds))
                        ],
                        'task_id_to_idx': dict(self.srt_task_id_to_idx),
                        'decoded_first': decoded,
                        'seq_len': int(seq_len),
                    })
                # ── END SRT DEBUG ────────────────────────────────────"""

new_debug_block = '''                # ── SRT DEBUG LOGGING (enriched) ──────────────────────────
                if self.srt_debug:
                    tok = getattr(self, '_tokenizer', None)
                    pad_id = getattr(self, '_pad_token_id', 0)

                    # Full route_debug: all Mahalanobis distances + confidence
                    route_info = self.srt_router.route_debug(route_inputs)

                    # Decode source tokens of first sample in batch
                    sid = source_input_ids[0].cpu()
                    mask = (sid != pad_id).cpu()
                    nonpad = sid[mask]
                    seq_len = nonpad.size(0)
                    decoded = ""
                    if tok is not None and seq_len > 0:
                        decoded = tok.decode(nonpad[:60], skip_special_tokens=True)[:150].replace("\\n", " ")

                    # Per-sample slot decisions
                    slot_idxs = [
                        min(self.srt_task_id_to_idx.get(tid, 0), n_slots - 1)
                        for tid in route_info["task_ids"]
                    ]
                    idx_to_tid = {v: k for k, v in self.srt_task_id_to_idx.items()}

                    self._srt_debug_log.append({
                        "batch_size": batch_size,
                        "n_slots": n_slots,
                        "n_tasks": route_info["n_tasks"],
                        "task_id_list": route_info["task_id_list"],
                        "srt_preds": route_info["task_ids"].tolist() if hasattr(route_info["task_ids"], "tolist") else list(route_info["task_ids"]),
                        "slot_idxs": slot_idxs,
                        "best_dist": route_info["best_dist"].tolist() if hasattr(route_info["best_dist"], "tolist") else list(route_info["best_dist"]),
                        "second_dist": route_info["second_dist"].tolist() if hasattr(route_info["second_dist"], "tolist") else list(route_info["second_dist"]),
                        "confidence": route_info["confidence_ratio"].tolist() if hasattr(route_info["confidence_ratio"], "tolist") else list(route_info["confidence_ratio"]),
                        "dists": {
                            tid: route_info["dists"][:, i].tolist()
                            for i, tid in enumerate(route_info["task_id_list"])
                        },
                        "task_id_to_idx": dict(self.srt_task_id_to_idx),
                        "decoded_first": decoded,
                        "seq_len": int(seq_len),
                        "encoder_type": type(self.encoder_frozen).__name__ if self.encoder_frozen is not None else "None",
                    })
                # ── END SRT DEBUG ────────────────────────────────────'''

content = content.replace(old_debug_block, new_debug_block)

with open(llama_path, "w") as f:
    f.write(content)
print("llama_gainlora.py: enriched SRT debug block replaced")

# ──────────────────────────────────────────────────────────────────────────────
# 3. ENHANCE run_llama_gainlora_cl.py: richer debug print block
# ──────────────────────────────────────────────────────────────────────────────
run_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_llama_gainlora/run_llama_gainlora_cl.py"

with open(run_path, "r") as f:
    content = f.read()

old_print_block = """        # ── SRT DEBUG LOGGING ────────────────────────────────────────────
        srt_log = model.get_srt_debug_log()
        for entry_idx, entry in enumerate(srt_log):
            batch_start = start + entry_idx * batch_size
            for sample_in_batch in range(entry['batch_size']):
                global_idx = batch_start + sample_in_batch
                if global_idx >= len(samples):
                    break
                sample = samples[global_idx]
                # Decode first 40 tokens of source prompt
                tok_ids = entry['first_tokens'][:40]
                decoded_text = tokenizer.decode(tok_ids, skip_special_tokens=True)[:120].replace('\\n', ' ')
                srt_preds = entry['srt_preds']
                slot_idxs = entry['slot_idxs']
                tid_map = entry['task_id_to_idx']
                # Build reverse map: slot_idx → task_id
                idx_to_tid = {v: k for k, v in tid_map.items()}
                pred_tid = srt_preds[sample_in_batch]
                pred_slot = slot_idxs[sample_in_batch]
                pred_task = idx_to_tid.get(pred_slot, f'unknown_slot{pred_slot}')
                gt_task = sample.get('Dataset', 'unknown')
                print(
                    f"  [SRT-DEBUG] idx={global_idx:3d}  "
                    f"slot={pred_slot}  "
                    f"task={pred_task[:40]:40s}  "
                    f"text={decoded_text[:80]!r}"
                )
        # ── END SRT DEBUG ─────────────────────────────────────────────

        for generated_ids, reference in zip(generated, refs):"""

new_print_block = """        # ── SRT DEBUG LOGGING ─────────────────────────────────────────────
        srt_log = model.get_srt_debug_log()
        for entry_idx, entry in enumerate(srt_log):
            batch_start = start + entry_idx * batch_size
            n_tasks = entry.get("n_tasks", 0)
            task_list = entry.get("task_id_list", [])

            for sample_in_batch in range(entry["batch_size"]):
                global_idx = batch_start + sample_in_batch
                if global_idx >= len(samples):
                    break
                sample = samples[global_idx]
                gt_task = sample.get("Dataset", sample.get("Instance", {}).get("task", "unknown"))

                # Predicted routing decision
                srt_preds = entry["srt_preds"]
                slot_idxs = entry["slot_idxs"]
                pred_task_str = srt_preds[sample_in_batch]
                pred_slot = slot_idxs[sample_in_batch]
                tid_map = entry["task_id_to_idx"]
                idx_to_tid = {v: k for k, v in tid_map.items()}
                slot_task = idx_to_tid.get(pred_slot, f"unknown_slot{pred_slot}")

                # Distances
                dists = entry.get("dists", {})
                best_d = entry["best_dist"][sample_in_batch]
                second_d = entry["second_dist"][sample_in_batch]
                conf = entry["confidence"][sample_in_batch]

                # Print top-2 nearest tasks
                sorted_tasks = sorted(
                    [(tid, dists.get(tid, [0])[sample_in_batch]) for tid in task_list],
                    key=lambda x: x[1]
                )
                top2 = sorted_tasks[:2]

                correct_flag = "✓" if (slot_task == gt_task or pred_task_str == gt_task) else "✗"

                # Format distances compactly
                dist_str = "  ".join([f"{t[:20]:20s}:{d:.1f}" for t, d in sorted_tasks[:4]])

                print(
                    f"  [SRT-D] idx={global_idx:3d}  "
                    f"n_tasks={n_tasks}  "
                    f"slot={pred_slot}  "
                    f"1st={pred_task_str[:25]:25s}(d={best_d:.2f})  "
                    f"2nd={top2[1][0][:25] if len(top2)>1 else 'N/A':25s}(d={top2[1][1]:.2f} if len(top2)>1 else 0)  "
                    f"conf={conf:.3f}  "
                    f"GT={gt_task[:25]:25s}  "
                    f"{correct_flag}  "
                    f"txt={entry['decoded_first'][:60]!r}"
                )
                print(f"        dists: {dist_str}")
        # ── END SRT DEBUG ─────────────────────────────────────────────

        for generated_ids, reference in zip(generated, refs):"""

content = content.replace(old_print_block, new_print_block)

with open(run_path, "w") as f:
    f.write(content)
print("run_llama_gainlora_cl.py: enriched SRT debug print block replaced")

print("\nAll debug logging additions complete.")
