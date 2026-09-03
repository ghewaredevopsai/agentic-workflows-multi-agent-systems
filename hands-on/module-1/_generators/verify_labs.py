#!/usr/bin/env python3
"""A participant hitting Run All on an untouched lab must not get a hard crash.

Unfilled blanks should surface as [TODO]/notes, never as an uncaught traceback.
"""
import io, json, os, sys, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
LABDIR = os.path.abspath(os.path.join(HERE, ".."))
for v in ("LAB_LLM_BASE_URL", "OPENAI_BASE_URL", "LAB_LLM_MODEL", "OPENAI_MODEL"):
    os.environ.pop(v, None)

bad = 0
for fn in sorted(f for f in os.listdir(LABDIR) if f.endswith(".ipynb")):
    nb = json.load(open(os.path.join(LABDIR, fn)))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns, crashed = {"__name__": "__nb__"}, []
    buf = io.StringIO()
    for i, c in enumerate(cells):
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile("".join(c["source"]), f"{fn}#cell{i}", "exec"), ns)
        except Exception as exc:
            crashed.append(f"cell {i}: {type(exc).__name__}: {exc}")
    out = buf.getvalue()
    print(f"[{'OK    ' if not crashed else 'CRASH ' }] {fn:44} "
          f"{out.count('[TODO]')} todo, {out.count('[PASS]')} pass, {len(crashed)} crash")
    for c in crashed:
        print("           " + c)
        bad += 1
sys.exit(1 if bad else 0)
