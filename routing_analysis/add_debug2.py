#!/usr/bin/env python3
"""Add additional critical debug traces to catch root cause."""

import os

# ──────────────────────────────────────────────────────────────────────────────
# 1. Add router-slot mapping debug at wiring time in sgwi_srt_trainer.py
# ──────────────────────────────────────────────────────────────────────────────
trainer_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_llama_gainlora/src/sgwi_srt_trainer.py"

with open(trainer_path, "r") as f:
    content = f.read()

# Add print when wiring router into model
old_wire = """        core.srt_router = self.srt_router
        core.srt_task_id_to_idx = task_id_to_idx
        core.use_srt_routing = True
        core.encoder_frozen = self.encoder_frozen"""

new_wire = """        core.srt_router = self.srt_router
        core.srt_task_id_to_idx = task_id_to_idx
        core.use_srt_routing = True
        core.encoder_frozen = self.encoder_frozen

        # Debug: verify router and mapping are wired correctly
        print(f"  [SRT-WIRE] router tasks: {list(self.srt_router.signatures.keys())}")
        print(f"  [SRT-WIRE] task_id_to_idx mapping: {task_id_to_idx}")
        print(f"  [SRT-WIRE] encoder_frozen type: {type(self.encoder_frozen)}")
        print(f"  [SRT-WIRE] router._impl centroids: {len(self.srt_router._impl.centroids)}")"""

content = content.replace(old_wire, new_wire)

with open(trainer_path, "w") as f:
    f.write(content)
print("sgwi_srt_trainer.py: SRT wire debug added")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Add check in llama_gainlora.py: verify encoder is used at inference
# ──────────────────────────────────────────────────────────────────────────────
llama_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_gainlora/src/llama_gainlora.py"

with open(llama_path, "r") as f:
    content = f.read()

# Add encoder check debug right after the route_debug call
old_encoder_check = '''                    # Full route_debug: all Mahalanobis distances + confidence
                    route_info = self.srt_router.route_debug(route_inputs)'''

new_encoder_check = '''                    # Full route_debug: all Mahalanobis distances + confidence
                    route_info = self.srt_router.route_debug(route_inputs)

                    # Verify encoder was used (critical for routing correctness)
                    encoder_was_none = self.encoder_frozen is None
                    routing_via_srt = (not self.training) and self.use_srt_routing'''

content = content.replace(old_encoder_check, new_encoder_check)

# Now add a print in generate() to trace encoder availability
old_gen_override = '''        # Decoder forward pass
        result = super().forward('''

new_gen_override = '''        # Decoder forward pass
        # SRT debug: report router state
        if hasattr(self, 'use_srt_routing') and self.use_srt_routing:
            n_tasks = len(getattr(self.srt_router, 'signatures', {}))
            router_ok = self.srt_router is not None
            enc_ok = self.encoder_frozen is not None
            # Only print on first call per generation step to avoid spam
            if not hasattr(self, '_srt_gen_printed') or not self._srt_gen_printed:
                print(f"  [SRT-GEN] use_srt={self.use_srt_routing}  router={'OK' if router_ok else 'NONE'}  "
                      f"encoder={'OK' if enc_ok else 'NONE'}  n_tasks={n_tasks}")
                self._srt_gen_printed = True
        result = super().forward('''

content = content.replace(old_gen_override, new_gen_override)

with open(llama_path, "w") as f:
    f.write(content)
print("llama_gainlora.py: encoder verification debug added")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Add route() fallback debug in run_llama_gainlora_cl.py
#    Show what's in the router when prediction fails
# ──────────────────────────────────────────────────────────────────────────────
run_path = "/Users/nhatminhnguyen/Library/Mobile Documents/com~apple~CloudDocs/Thạc sĩ/Continual/new_llama_gainlora/run_llama_gainlora_cl.py"

with open(run_path, "r") as f:
    content = f.read()

# Add a summary print at the start of each batch
old_batch_loop = """    for start in tqdm(range(0, len(samples), batch_size), total=n_batches, desc=desc):
        batch_samples = samples[start : start + batch_size]
        batch = collator(batch_samples)
        batch = {k: v.to(device) for k, v in batch.items()}
        refs = [s["Instance"]["label"] for s in batch_samples]
        input_length = batch["input_ids_wo_label"].shape[1]"""

new_batch_loop = """    # SRT router status check (print once)
    if hasattr(model.model, 'srt_router') and model.model.srt_router is not None:
        router = model.model.srt_router
        print(f"  [SRT-INIT] router tasks={list(router.signatures.keys())}")
        print(f"  [SRT-INIT] task_id_to_idx={getattr(model.model, 'srt_task_id_to_idx', {})}")
        print(f"  [SRT-INIT] use_srt_routing={getattr(model.model, 'use_srt_routing', False)}")
        print(f"  [SRT-INIT] encoder_frozen={type(model.model.encoder_frozen).__name__ if model.model.encoder_frozen is not None else 'None'}")
    else:
        print(f"  [SRT-INIT] NO ROUTER FOUND!")

    for start in tqdm(range(0, len(samples), batch_size), total=n_batches, desc=desc):
        batch_samples = samples[start : start + batch_size]
        batch = collator(batch_samples)
        batch = {k: v.to(device) for k, v in batch.items()}
        refs = [s["Instance"]["label"] for s in batch_samples]
        input_length = batch["input_ids_wo_label"].shape[1]"""

content = content.replace(old_batch_loop, new_batch_loop)

with open(run_path, "w") as f:
    f.write(content)
print("run_llama_gainlora_cl.py: router status check + SRT-GEN debug added")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Add embedding extraction verification in sgwi_srt_trainer.py
#    Check if embeddings are reasonable
# ──────────────────────────────────────────────────────────────────────────────

with open(trainer_path, "r") as f:
    content = f.read()

# Find _extract_task_embeddings and add stats printing
old_extract = """        with torch.no_grad():
            h_all.append(h_batch.cpu())"""

new_extract = """        with torch.no_grad():
            h_all.append(h_batch.cpu())
            # Quick stats on embeddings
            if len(h_all) == 1:
                print(f"  [SRT-EMB] shape={h_batch.shape}, mean={h_batch.mean():.4f}, std={h_batch.std():.4f}, "
                      f"min={h_batch.min():.4f}, max={h_batch.max():.4f}")"""

content = content.replace(old_extract, new_extract)

with open(trainer_path, "w") as f:
    f.write(content)
print("sgwi_srt_trainer.py: embedding stats added")

print("\nAll additional debug traces added.")