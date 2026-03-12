import re, pathlib

FILES = [
    "root_gainlora/src/cl_trainer_inflora.py",
    "root_gainlora/src/cl_trainer_olora.py",
    "root_gainlora/src/cl_trainer_gainlora_olora_llama.py",
    "root_gainlora/src/cl_trainer_gainlora_inflora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_inflora.py",
    "improve_gainlora/src/cl_trainer_inflora.py",
    "improve_gainlora/src/cl_trainer_inflora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_olora.py",
    "improve_gainlora/src/cl_trainer_gainlora_olora_llama.py",
    "improve_gainlora/src/cl_trainer_gainlora_inflora_llama.py",
]

for rel in FILES:
    p = pathlib.Path(rel)
    txt = p.read_text()

    # Skip if already fixed
    if 'gen_kwargs.pop("synced_gpus"' in txt:
        pop_idx = txt.index('gen_kwargs.pop("synced_gpus"')
        gc_idx  = txt.index('generation_config = GenerationConfig(**gen_kwargs)')
        if pop_idx < gc_idx:
            print(f"  ALREADY FIXED: {rel}")
            continue

    old_gc = '        generation_config = GenerationConfig(**gen_kwargs)'
    new_gc = '        synced_gpus = gen_kwargs.pop("synced_gpus", False)\n        generation_config = GenerationConfig(**gen_kwargs)'
    if old_gc not in txt:
        print(f"  WARNING: GenerationConfig pattern not found in {rel}")
        continue
    txt = txt.replace(old_gc, new_gc, 1)

    # Add synced_gpus= to each generate() call after generation_config= line
    txt = re.sub(
        r'(generation_config=generation_config,)\n(\s+\))',
        r'\1\n\2    synced_gpus=synced_gpus,\n\2',
        txt,
    )
    txt = re.sub(
        r'(generation_config=generation_config)\n(\s+\))',
        r'\1,\n\2    synced_gpus=synced_gpus,\n\2',
        txt,
    )

    p.write_text(txt)
    print(f"  FIXED: {rel}")

print("Done")
