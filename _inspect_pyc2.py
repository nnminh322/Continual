import marshal, types

f = open("improve_gainlora/src/__pycache__/t5_specroute.cpython-312.pyc", "rb")
f.read(16)
code = marshal.load(f)

def find_func(co, target_name, target_line=None, depth=0):
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            if c.co_name == target_name and (target_line is None or c.co_firstlineno == target_line):
                print(f"FUNC: {c.co_name} (line {c.co_firstlineno})")
                print(f"  ALL vars: {c.co_varnames}")
                print(f"  ALL names: {c.co_names}")
                ss = [s for s in c.co_consts if isinstance(s, str)]
                print(f"  ALL strings: {ss}")
                print()
            find_func(c, target_name, target_line, depth+1)

# T5Stack.__init__
find_func(code, "__init__", 120)
# T5Stack.forward (the one with all the routing logic)
find_func(code, "forward", 250)
