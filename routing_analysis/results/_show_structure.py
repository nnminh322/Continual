#!/usr/bin/env python3
import json, os

DIR = os.path.dirname(os.path.abspath(__file__))

def load(fn):
    fp = os.path.join(DIR, fn)
    with open(fp) as f:
        return json.load(f)

def show_structure(d, prefix="", depth=0, max_depth=3):
    if depth >= max_depth:
        return
    if isinstance(d, dict):
        for k in list(d.keys())[:15]:
            v = d[k]
            if isinstance(v, dict):
                print(f"{prefix}{k}: dict({len(v)} keys: {list(v.keys())[:8]})")
                if depth < max_depth - 1:
                    show_structure(v, prefix + "  ", depth + 1, max_depth)
            elif isinstance(v, list):
                sample = str(v[0])[:60] if v else "empty"
                print(f"{prefix}{k}: list[{len(v)}] first={sample}")
            else:
                print(f"{prefix}{k}: {str(v)[:100]}")
    elif isinstance(d, list):
        print(f"{prefix}list[{len(d)}]")
        if d and isinstance(d[0], dict):
            show_structure(d[0], prefix + "  ", depth + 1, max_depth)

# === GEOMETRY ===
print("="*60)
print("GEOMETRY JSON STRUCTURE")
print("="*60)
d = load("geometry_flan-t5-large_Long_Sequence.json")
show_structure(d)

# === ROUTING ===
print("\n" + "="*60)
print("ROUTING JSON STRUCTURE")
print("="*60)
d = load("routing_flan-t5-large_Long_Sequence.json")
show_structure(d)

# === ABLATION ===
print("\n" + "="*60)
print("ABLATION JSON STRUCTURE")
print("="*60)
d = load("ablation_flan-t5-large_Long_Sequence.json")
show_structure(d)

# === THEORY ===
print("\n" + "="*60)
print("THEORY JSON STRUCTURE")
print("="*60)
d = load("theory_flan-t5-large_Long_Sequence.json")
show_structure(d)

# === LEARNED ROUTING ===
print("\n" + "="*60)
print("LEARNED ROUTING JSON STRUCTURE")
print("="*60)
d = load("learned_routing_flan-t5-large_Long_Sequence.json")
show_structure(d)
