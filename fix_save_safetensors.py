import os

REPOS = ["root_gainlora/src", "improve_gainlora/src"]

MARKER = "    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:\n"

OVERRIDE = (
    "    def _save(self, output_dir=None, state_dict=None):\n"
    "        # T5 shared embeddings are incompatible with safetensors; force pytorch format\n"
    "        old = getattr(self.args, 'save_safetensors', True)\n"
    "        self.args.save_safetensors = False\n"
    "        try:\n"
    "            super()._save(output_dir=output_dir, state_dict=state_dict)\n"
    "        finally:\n"
    "            self.args.save_safetensors = old\n"
    "\n"
)

ALREADY = "def _save(self, output_dir=None, state_dict=None):"

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
        if MARKER not in src:
            print(f"SKIP (no marker): {fpath}")
            continue
        new_src = src.replace(MARKER, OVERRIDE + MARKER, 1)
        with open(fpath, "w") as f:
            f.write(new_src)
        print(f"FIXED: {fpath}")

print("Done.")
