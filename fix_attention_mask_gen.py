"""
Fix: attention_mask is incorrectly added to gen_kwargs before GenerationConfig,
but GenerationConfig does not accept attention_mask. It must be extracted and
passed directly to model.generate(), just like the synced_gpus fix.
"""
import os, re

REPOS = ["root_gainlora/src", "improve_gainlora/src"]

# Pattern to find and fix
OLD_BLOCK = (
    '        if "attention_mask" in inputs:\n'
    '            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)\n'
    '\n'
    '        generation_config = GenerationConfig(**gen_kwargs)\n'
)
NEW_BLOCK = (
    '        attention_mask = inputs.get("attention_mask", None)\n'
    '\n'
    '        generation_config = GenerationConfig(**gen_kwargs)\n'
)

ALREADY = 'attention_mask = inputs.get("attention_mask", None)'

# All three model.generate() patterns that need attention_mask added
# Pattern A: encoder-decoder branch (first if)
OLD_GEN_A = (
    '            generated_tokens = self.model.generate(\n'
    '                input_ids=generation_inputs, \n'
    '                generation_config=generation_config,\n'
    '                synced_gpus=synced_gpus,\n'
    '            )\n'
)
NEW_GEN_A = (
    '            generated_tokens = self.model.generate(\n'
    '                input_ids=generation_inputs, \n'
    '                generation_config=generation_config,\n'
    '                attention_mask=attention_mask,\n'
    '                synced_gpus=synced_gpus,\n'
    '            )\n'
)

# Pattern B: LLaMA branch with input_ids_wo_label
OLD_GEN_B = (
    '                generated_tokens = self.model.generate(\n'
    '                    input_ids=generation_inputs,\n'
    '                    input_ids_wo_label=inputs["input_ids_wo_label"],\n'
    '                    generation_config=generation_config,\n'
    '                    synced_gpus=synced_gpus,\n'
    '                )\n'
)
NEW_GEN_B = (
    '                generated_tokens = self.model.generate(\n'
    '                    input_ids=generation_inputs,\n'
    '                    input_ids_wo_label=inputs["input_ids_wo_label"],\n'
    '                    generation_config=generation_config,\n'
    '                    attention_mask=attention_mask,\n'
    '                    synced_gpus=synced_gpus,\n'
    '                )\n'
)

# Pattern C: T5 (else branch, no input_ids_wo_label)
OLD_GEN_C = (
    '                generated_tokens = self.model.generate(\n'
    '                    input_ids=generation_inputs,\n'
    '                    generation_config=generation_config,\n'
    '                    synced_gpus=synced_gpus,\n'
    '                )\n'
)
NEW_GEN_C = (
    '                generated_tokens = self.model.generate(\n'
    '                    input_ids=generation_inputs,\n'
    '                    generation_config=generation_config,\n'
    '                    attention_mask=attention_mask,\n'
    '                    synced_gpus=synced_gpus,\n'
    '                )\n'
)

for repo in REPOS:
    for fname in sorted(os.listdir(repo)):
        if not fname.startswith("cl_trainer_") or not fname.endswith(".py"):
            continue
        fpath = os.path.join(repo, fname)
        with open(fpath) as f:
            src = f.read()
        if ALREADY in src:
            print(f"SKIP (already fixed): {fpath}")
            continue
        if OLD_BLOCK not in src:
            print(f"SKIP (no old block): {fpath}")
            continue
        new_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
        new_src = new_src.replace(OLD_GEN_A, NEW_GEN_A)
        new_src = new_src.replace(OLD_GEN_B, NEW_GEN_B)
        new_src = new_src.replace(OLD_GEN_C, NEW_GEN_C)
        with open(fpath, "w") as f:
            f.write(new_src)
        print(f"FIXED: {fpath}")

print("Done.")
