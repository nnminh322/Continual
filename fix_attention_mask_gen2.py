"""
Fix v2: For all files where attention_mask is still in gen_kwargs before GenerationConfig.
Uses regex to handle all ordering variants.
"""
import os, re

REPOS = ["root_gainlora/src", "improve_gainlora/src"]
OLD_FLAG = 'gen_kwargs["attention_mask"]'
ALREADY = 'attention_mask = inputs.get("attention_mask", None)'

def fix_file(fpath):
    with open(fpath) as f:
        src = f.read()

    if ALREADY in src:
        print(f"SKIP (already): {fpath}")
        return
    if OLD_FLAG not in src:
        print(f"SKIP (no flag): {fpath}")
        return

    # Step 1: Replace the "gen_kwargs["attention_mask"] = ..." block with extraction
    step1 = re.sub(
        r'        if "attention_mask" in inputs:\n            gen_kwargs\["attention_mask"\] = inputs\.get\("attention_mask", None\)\n\n',
        '        attention_mask = inputs.get("attention_mask", None)\n\n',
        src,
    )
    if step1 == src:
        # Variant without blank line after
        step1 = re.sub(
            r'        if "attention_mask" in inputs:\n            gen_kwargs\["attention_mask"\] = inputs\.get\("attention_mask", None\)\n',
            '        attention_mask = inputs.get("attention_mask", None)\n',
            src,
        )

    # Step 2: Add attention_mask= to every model.generate() call that doesn't have it
    result = re.sub(
        r'(self\.model\.generate\(\n(?:(?!attention_mask)(?!synced_gpus)[^\n]*\n)*?)(\s*synced_gpus=synced_gpus,\n\s*\))',
        r'\1                attention_mask=attention_mask,\n\2',
        step1,
    )

    if result == src:
        print(f"WARNING: no change for {fpath}")
        return

    with open(fpath, "w") as f:
        f.write(result)
    print(f"FIXED: {fpath}")

for repo in REPOS:
    for fname in sorted(os.listdir(repo)):
        if not fname.startswith("cl_trainer_") or not fname.endswith(".py"):
            continue
        fix_file(os.path.join(repo, fname))

print("Done.")
