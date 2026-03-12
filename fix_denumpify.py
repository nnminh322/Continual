import os

REPOS = ["root_gainlora/src", "improve_gainlora/src"]

OLD = (
    "from transformers.trainer_pt_utils import (\n"
    "    nested_truncate, nested_concat, nested_numpify,\n"
    "    denumpify_detensorize, find_batch_size,\n"
    ")\n"
)

NEW = (
    "from transformers.trainer_pt_utils import (\n"
    "    nested_truncate, nested_concat, nested_numpify,\n"
    "    find_batch_size,\n"
    ")\n"
    "try:\n"
    "    from transformers.trainer_pt_utils import denumpify_detensorize\n"
    "except ImportError:\n"
    "    from transformers.trainer_utils import denumpify_detensorize\n"
)

for repo in REPOS:
    for fname in sorted(os.listdir(repo)):
        if not fname.startswith("cl_trainer_") or not fname.endswith(".py"):
            continue
        fpath = os.path.join(repo, fname)
        with open(fpath) as f:
            src = f.read()
        if OLD not in src:
            print(f"SKIP: {fpath}")
            continue
        with open(fpath, "w") as f:
            f.write(src.replace(OLD, NEW, 1))
        print(f"FIXED: {fpath}")

print("Done.")
