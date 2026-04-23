import marshal, types

f = open("improve_gainlora/src/__pycache__/t5_specroute.cpython-312.pyc", "rb")
f.read(16)
code = marshal.load(f)

def ff(co, depth=0):
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            has_kw = any(kw in c.co_name.lower() for kw in ["rls", "collect", "feature", "update_rls", "router"])
            if has_kw:
                print(f"FUNC: {c.co_name} (line {c.co_firstlineno})")
                print(f"  ALL vars: {c.co_varnames}")
                print(f"  ALL names: {c.co_names}")
                ss = [s for s in c.co_consts if isinstance(s, str)]
                print(f"  ALL strings: {ss}")
                print()
            ff(c, depth+1)
ff(code)
            if has_kw:
                print(f"FUNC: {c.co_name} (line {c.co_firstlineno})")
                print(f"  args: {c.co_varnames[:c.co_argcount]}")
                rv = [v for v in c.co_varnames if any(k in v.lower() for k in ["embed", "hidden", "encoder", "feature", "mean", "pool", "rls", "input", "h_", "token"])]
                print(f"  relevant_vars: {rv}")
                rn = [n for n in c.co_names if any(k in n.lower() for k in ["embed", "hidden", "encoder", "feature", "mean", "pool", "rls", "input", "forward", "token"])]
                print(f"  relevant_names: {rn}")
                ss = [s for s in c.co_consts if isinstance(s, str) and len(s) > 3 and any(k in s.lower() for k in ["rls", "feature", "embed", "encoder", "mean", "pool", "routing", "hidden", "input", "collect"])]
                if ss:
                    print(f"  strings: {ss}")
                print()
            ff(c, depth+1)

ff(code)
