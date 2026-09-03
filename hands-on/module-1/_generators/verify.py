#!/usr/bin/env python3
"""Execute every solution notebook's code cells and assert a clean score.

Live-model cells self-skip because LAB_LLM_BASE_URL is unset, so this runs offline.
Also checks that each lab has blanks and that no solution does.
"""
import io, json, os, sys, contextlib, re

HERE = os.path.dirname(os.path.abspath(__file__))
LABDIR = os.path.abspath(os.path.join(HERE, ".."))
SOLDIR = os.path.join(LABDIR, "solutions")

for v in ("LAB_LLM_BASE_URL", "OPENAI_BASE_URL", "LAB_LLM_MODEL", "OPENAI_MODEL"):
    os.environ.pop(v, None)

fails = 0
for fn in sorted(f for f in os.listdir(SOLDIR) if f.endswith(".ipynb")):
    nb = json.load(open(os.path.join(SOLDIR, fn)))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    buf, ns = io.StringIO(), {"__name__": "__nb__"}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, fn, "exec"), ns)
    except Exception as exc:
        print(f"[ERROR ] {fn}: {type(exc).__name__}: {exc}")
        fails += 1
        continue
    out = buf.getvalue()
    n_fail = out.count("[FAIL]")
    n_todo = out.count("[TODO]")
    n_pass = out.count("[PASS]")
    m = re.search(r"Score: (\d+)/(\d+)", out)
    ok = n_fail == 0 and n_todo == 0 and m and m.group(1) == m.group(2)
    print(f"[{'OK    ' if ok else 'BROKEN'}] {fn:44} {n_pass} pass, {n_fail} fail, {n_todo} todo, "
          f"score {m.group(0) if m else 'MISSING'}")
    if not ok:
        fails += 1
        for line in out.splitlines():
            if "[FAIL]" in line or "[TODO]" in line:
                print("           " + line)

print("\n--- blanks ---")
for fn in sorted(f for f in os.listdir(LABDIR) if f.endswith(".ipynb")):
    lab = json.load(open(os.path.join(LABDIR, fn)))
    sol = json.load(open(os.path.join(SOLDIR, fn)))
    lb = sum("".join(c["source"]).count("BLANK") for c in lab["cells"] if c["cell_type"] == "code")
    sb = sum("".join(c["source"]).count("BLANK") for c in sol["cells"] if c["cell_type"] == "code")
    good = lb > 0 and sb == 0
    print(f"[{'OK    ' if good else 'BROKEN'}] {fn:44} {lb} blanks in lab, {sb} in solution")
    if not good:
        fails += 1

sys.exit(1 if fails else 0)
