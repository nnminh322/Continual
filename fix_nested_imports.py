import os

REPOS = ["root_gainlora/src", "improve_gainlora/src"]

NEW_IMPORT = (
    "from transformers.trainer_pt_utils import (\n"
    "    nested_truncate, nested_concat, nested_numpify,\n"
    "    denumpify_detensorize, find_batch_size,\n"
    ")\n"
)

MARKER = "from transformers.trainer import *\n"
ALREADY = "nested_truncate, nested_concat"

for repo in REPOS:
    for fname in sorted(os.listdir(repo)):
        if not fname.startswith("cl_trainer_") or not fname.endswith(".py"):
            continue
        fpath = os.path.join(repo, fname)
        with open(fpath) as f:
            src = f.read()
        if MARKER not in src:
            print(f"SKIP (no marker): {fpath}")
            continue
        if ALREADY in src:
            print(f"SKIP (already fixed): {fpath}")
            continue
        new_src = src.replace(MARKER, MARKER + NEW_IMPORT, 1)
        with open(fpath, "w") as f:
            f.write(new_src)
        print(f"FIXED: {fpath}")

print("Done.")
