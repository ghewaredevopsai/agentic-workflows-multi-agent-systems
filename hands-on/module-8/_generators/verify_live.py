#!/usr/bin/env python3
"""Execute every SOLUTION notebook against the live sandbox model, in a real Jupyter kernel.

This is the verifier that matters since Day 1 became framework-forward: the offline
verify.py proves the object-level self-checks hold, but only this proves that the
create_agent runs, the tool bindings, the structured outputs and the checkpointer
actually work against the served model.

    python3 verify_live.py                 # every solution
    python3 verify_live.py lab-8-03        # just the ones whose name matches

Run it ON THE CLUSTER (or anywhere the LAB_LLM_* variables point at a live gateway).
It makes real model calls and costs real tokens -- a full Module 8 pass is about a cent.
"""
import os, sys, json, time

CELL_TIMEOUT = int(os.environ.get("LAB_CELL_TIMEOUT", "1200"))
HERE = os.path.dirname(os.path.abspath(__file__))
SOLDIR = os.path.join(os.path.abspath(os.path.join(HERE, "..")), "solutions")

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError, DeadKernelError

if not (os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")):
    sys.exit("no gateway configured -- this verifier is meant to run where the model is live")

want = sys.argv[1:]
fails = 0
for fn in sorted(f for f in os.listdir(SOLDIR) if f.endswith(".ipynb")):
    if want and not any(w in fn for w in want):
        continue
    nb = nbformat.read(os.path.join(SOLDIR, fn), as_version=4)
    t0 = time.time()
    note = []
    try:
        NotebookClient(nb, timeout=CELL_TIMEOUT, kernel_name="python3",
                       allow_errors=True).execute()
    except CellTimeoutError:
        note.append(f"TIMEOUT after {CELL_TIMEOUT}s")
    except DeadKernelError:
        note.append("KERNEL DIED (OOM?)")

    out, errors = "", []
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                out += o.get("text", "")
            elif o.get("output_type") == "error":
                errors.append(f"cell {i}: {o.get('ename')}: {(o.get('evalue') or '')[:120]}")
            elif o.get("output_type") == "execute_result":
                out += o.get("data", {}).get("text/plain", "")

    n_fail, n_todo = out.count("[FAIL]"), out.count("[TODO]")
    # A live cell that silently degraded is the failure this verifier exists to catch.
    degraded = out.count("<model unavailable") + out.count("not importable here") \
             + out.count("a blank above is still unfilled")
    ok = not errors and not note and n_fail == 0 and n_todo == 0 and degraded == 0
    fails += not ok
    print(f"[{'OK    ' if ok else 'BROKEN'}] {fn:44} {time.time()-t0:6.1f}s  "
          f"{n_fail} fail, {n_todo} todo, {degraded} degraded, {len(errors)} error")
    for line in errors[:6] + note:
        print("           " + line)
    if degraded:
        for line in out.splitlines():
            if any(k in line for k in ("<model unavailable", "not importable here",
                                       "a blank above is still unfilled")):
                print("           " + line.strip()[:150])
    if n_fail:
        for line in out.splitlines():
            if "[FAIL]" in line:
                print("           " + line.strip()[:150])

sys.exit(1 if fails else 0)
