"""
Comprehensive compat fix for all trainer files in root_gainlora and improve_gainlora.

Fixes applied:
1. Add compat shims for ShardedDDPOption, is_torch_tpu_available, IterableDatasetShard
   if not already present.
2. Replace unguarded self.do_grad_scaling / self.use_apex / self.sharded_ddp / self.fsdp
   with getattr() fallbacks.
3. Ensure _pad_across_processes -> accelerator.pad_across_processes (already done, verify).
"""

import re
import pathlib

# -----------------------------------------------------------------------
# Config: files to process and which fixes each needs
# -----------------------------------------------------------------------

# Files that need BOTH import shims AND attribute getattr fixes
FILES_FULL = [
    "root_gainlora/src/cl_trainer_inflora.py",
    "root_gainlora/src/cl_trainer_inflora_llama.py",
    "root_gainlora/src/cl_trainer_olora.py",
    "root_gainlora/src/cl_trainer_olora_llama.py",
    "root_gainlora/src/cl_trainer_gainlora_olora.py",
    "root_gainlora/src/cl_trainer_gainlora_olora_llama.py",
    "root_gainlora/src/cl_trainer_gainlora_inflora_llama.py",   # needs shims
    "improve_gainlora/src/cl_trainer_gainlora_inflora.py",
    "improve_gainlora/src/cl_trainer_inflora.py",
    "improve_gainlora/src/cl_trainer_inflora_llama.py",
    "improve_gainlora/src/cl_trainer_olora.py",
    "improve_gainlora/src/cl_trainer_olora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_olora.py",
    "improve_gainlora/src/cl_trainer_gainlora_olora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_inflora_llama.py",  # needs shims
]

# Files that already have shims but may still have unguarded attributes
FILES_ATTR_ONLY = [
    "root_gainlora/src/cl_trainer_gainlora_inflora.py",      # manually fixed before
    "improve_gainlora/src/cl_trainer_specroute.py",
]

SHIM_BLOCK = """\
# Compat: ShardedDDPOption removed in transformers >= 4.40
try:
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    from types import SimpleNamespace
    ShardedDDPOption = SimpleNamespace(SIMPLE='simple')

# Compat: is_torch_tpu_available removed in transformers >= 4.40
try:
    from transformers import is_torch_tpu_available
except ImportError:
    def is_torch_tpu_available():
        return False

# Compat: IterableDatasetShard moved/removed in transformers >= 4.40
try:
    from transformers.trainer_pt_utils import IterableDatasetShard
except ImportError:
    from torch.utils.data import IterableDataset as IterableDatasetShard

"""

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def add_shims(txt: str) -> str:
    """Insert the shim block right after the last top-level import block."""
    # Skip if already has our shim block (identified by SimpleNamespace shim or our own def)
    if 'SimpleNamespace' in txt and 'ShardedDDPOption' in txt:
        return txt  # already has shims
    if "def is_torch_tpu_available():\n        return False" in txt:
        return txt  # already has is_torch_tpu shim
    # Insert before first class or function definition
    insert_marker = re.search(r'^(class |def )', txt, re.MULTILINE)
    if insert_marker:
        pos = insert_marker.start()
        return txt[:pos] + SHIM_BLOCK + txt[pos:]
    return SHIM_BLOCK + txt


def fix_attributes(txt: str) -> str:
    """Replace direct attribute access with getattr() calls.
    
    Skips assignment targets (self.x = ...) but fixes reads and comparisons.
    Uses (?!\s*=[^=]) to distinguish  = (assign) from == (compare).
    Also skips if already wrapped in getattr().
    """
    def make_pattern(attr):
        # Negative lookbehind: not inside getattr( ... 'attr'
        # Negative lookahead: not followed by assignment  = (but allow ==)
        return rf"(?<!getattr\(self, ')(?<!\w)self\.{attr}(?!\s*=[^=])(?!\w)"

    txt = re.sub(make_pattern('do_grad_scaling'),
                 "getattr(self, 'do_grad_scaling', False)", txt)
    txt = re.sub(make_pattern('use_apex'),
                 "getattr(self, 'use_apex', False)", txt)
    txt = re.sub(make_pattern('sharded_ddp'),
                 "getattr(self, 'sharded_ddp', None)", txt)
    txt = re.sub(make_pattern('fsdp'),
                 "getattr(self, 'fsdp', None)", txt)
    return txt


def process(path_str: str, add_import_shims: bool) -> None:
    p = pathlib.Path(path_str)
    if not p.exists():
        print(f"  SKIP (not found): {path_str}")
        return
    txt = p.read_text()
    original = txt

    if add_import_shims:
        txt = add_shims(txt)
    txt = fix_attributes(txt)

    if txt == original:
        print(f"  OK (no changes): {path_str}")
    else:
        p.write_text(txt)
        # Count replacements
        attr_changes = sum([
            len(re.findall(r"getattr\(self, 'do_grad_scaling'", txt)) -
            len(re.findall(r"getattr\(self, 'do_grad_scaling'", original)),
            len(re.findall(r"getattr\(self, 'use_apex'", txt)) -
            len(re.findall(r"getattr\(self, 'use_apex'", original)),
            len(re.findall(r"getattr\(self, 'sharded_ddp'", txt)) -
            len(re.findall(r"getattr\(self, 'sharded_ddp'", original)),
            len(re.findall(r"getattr\(self, 'fsdp'", txt)) -
            len(re.findall(r"getattr\(self, 'fsdp'", original)),
        ])
        shim_added = 'shims' if add_import_shims and 'SimpleNamespace' not in original else ''
        print(f"  FIXED ({attr_changes} attrs{', shims' if shim_added else ''}): {path_str}")


print("=== Adding shims + fixing attributes ===")
for f in FILES_FULL:
    process(f, add_import_shims=True)

print("\n=== Fixing attributes only (shims already present) ===")
for f in FILES_ATTR_ONLY:
    process(f, add_import_shims=False)

print("\nDone")
