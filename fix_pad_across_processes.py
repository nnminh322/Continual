"""Replace self._pad_across_processes(x) with inline accelerator call in all trainer files."""
import pathlib

FILES = [
    "root_gainlora/src/cl_trainer_gainlora_inflora.py",
    "root_gainlora/src/cl_trainer_inflora.py",
    "root_gainlora/src/cl_trainer_inflora_llama.py",
    "root_gainlora/src/cl_trainer_olora.py",
    "root_gainlora/src/cl_trainer_olora_llama.py",
    "root_gainlora/src/cl_trainer_gainlora_olora.py",
    "root_gainlora/src/cl_trainer_gainlora_olora_llama.py",
    "root_gainlora/src/cl_trainer_gainlora_inflora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_inflora.py",
    "improve_gainlora/src/cl_trainer_inflora.py",
    "improve_gainlora/src/cl_trainer_inflora_llama.py",
    "improve_gainlora/src/cl_trainer_olora.py",
    "improve_gainlora/src/cl_trainer_olora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_olora.py",
    "improve_gainlora/src/cl_trainer_gainlora_olora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_inflora_llama.py",
]

OLD = "self._pad_across_processes("
# Replace with identical semantics: pad only when distributed, otherwise return as-is.
# self.accelerator.pad_across_processes pads dim=0 by default (sequence dim for gathered batches).
NEW = "self.accelerator.pad_across_processes("

count_total = 0
for rel in FILES:
    p = pathlib.Path(rel)
    if not p.exists():
        print(f"  SKIP (not found): {rel}")
        continue
    txt = p.read_text()
    n = txt.count(OLD)
    if n == 0:
        print(f"  OK (no calls): {rel}")
        continue
    txt = txt.replace(OLD, NEW)
    p.write_text(txt)
    count_total += n
    print(f"  FIXED ({n} calls): {rel}")

print(f"\nTotal replacements: {count_total}")
print("Done")
