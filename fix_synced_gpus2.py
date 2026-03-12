"""Fix the broken synced_gpus insertion: convert
    generation_config=generation_config,
            )    synced_gpus=synced_gpus,
            )
to the correct form:
    generation_config=generation_config,
                synced_gpus=synced_gpus,
            )

This must be idempotent.
"""
import re, pathlib

FILES = [
    "root_gainlora/src/cl_trainer_inflora.py",
    "root_gainlora/src/cl_trainer_olora.py",
    "root_gainlora/src/cl_trainer_gainlora_olora_llama.py",
    "root_gainlora/src/cl_trainer_gainlora_inflora_llama.py",
    "root_gainlora/src/cl_trainer_gainlora_inflora.py",   # already done manually but check
    "improve_gainlora/src/cl_trainer_gainlora_inflora.py",
    "improve_gainlora/src/cl_trainer_inflora.py",
    "improve_gainlora/src/cl_trainer_inflora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_olora.py",
    "improve_gainlora/src/cl_trainer_gainlora_olora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_inflora_llama.py",
]

# Pattern produced by the broken regex:
#   generation_config=generation_config,\n{ws})    synced_gpus=synced_gpus,\n{ws})
BAD_PAT = re.compile(
    r'(generation_config=generation_config,)\n'
    r'( +)\)    synced_gpus=synced_gpus,\n'
    r'( +\))'
)

# Correct form: synced_gpus goes INSIDE the call, before the closing paren
# We add 4 extra spaces relative to the closing-paren indent level.
def fix_bad(txt):
    def replacer(m):
        arg_line = m.group(1)            # generation_config=generation_config,
        inner_ws = m.group(2)            # whitespace before the doubled )
        close_paren = m.group(3)         # whitespace + )
        # synced_gpus should be indented 4 more than the closing paren indent
        synced_ws = inner_ws + "    "
        return f"{arg_line}\n{synced_ws}synced_gpus=synced_gpus,\n{close_paren}"
    return BAD_PAT.sub(replacer, txt)

for rel in FILES:
    p = pathlib.Path(rel)
    txt = p.read_text()

    if ")    synced_gpus=synced_gpus," in txt:
        fixed = fix_bad(txt)
        p.write_text(fixed)
        n = txt.count(")    synced_gpus=synced_gpus,")
        print(f"  FIXED ({n} occurrences): {rel}")
    else:
        print(f"  OK (no bad pattern): {rel}")

print("Done")
