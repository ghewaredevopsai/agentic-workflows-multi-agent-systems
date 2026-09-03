#!/usr/bin/env python3
"""A participant hitting Run All on an untouched lab must not get a hard crash.

Unfilled blanks should surface as [TODO]/notes, never as an uncaught traceback.

Each lab is executed TWICE and the two runs must agree:

  plain exec()          -- cheap reference semantics
  a real Jupyter kernel -- what a participant actually gets

Agreement is the real check, and it is what makes this file worth its runtime.
A notebook must not depend on IPython-specific semantics, and the difference is
not academic:

  * IPython's displayhook predefines _, __ and ___ as the output history,
    initialised to "". A blank spelled with underscores is therefore a *defined
    empty string* in a notebook and raises no NameError. [TODO] silently becomes
    [FAIL] or even [PASS], and a blank used as a loop guard never stops its loop
    -- lab 1.1 once spun in `while True` until the pod was OOM-killed.
  * A plain-exec-only verifier reported all-green on exactly that notebook, and
    so did one built on a bare InteractiveShell (which does not seed those names
    either). Only a kernel is faithful, and only the COMPARISON localises the
    problem instead of just tripping over it.

Hence the sentinel `BLANK`, which is undefined under both. Still offline: no
cluster, no model, no network -- just a local kernel.
"""
import io, json, os, sys, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
LABDIR = os.path.abspath(os.path.join(HERE, ".."))
# A hang is among the worst failures here, so cells run under a timeout. A
# runaway that ALLOCATES can still kill the kernel first; that is caught too.
CELL_TIMEOUT = int(os.environ.get("LAB_CELL_TIMEOUT", "120"))

for v in ("LAB_LLM_BASE_URL", "OPENAI_BASE_URL", "LAB_LLM_MODEL", "OPENAI_MODEL"):
    os.environ.pop(v, None)

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError, DeadKernelError


def tally(text):
    return (text.count("[TODO]"), text.count("[PASS]"), text.count("[FAIL]"))


def run_plain(path):
    """Old reference semantics: exec each cell in one namespace."""
    nb = json.load(open(path))
    ns, buf, crashed = {"__name__": "__nb__"}, io.StringIO(), []
    for i, c in enumerate(x for x in nb["cells"] if x["cell_type"] == "code"):
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile("".join(c["source"]), f"{os.path.basename(path)}#{i}", "exec"), ns)
        except Exception as exc:
            crashed.append(f"cell {i}: {type(exc).__name__}: {exc}")
    return tally(buf.getvalue()), crashed


def run_kernel(path):
    """What a participant gets: a real Jupyter kernel."""
    nb = nbformat.read(path, as_version=4)
    crashed = []
    try:
        NotebookClient(nb, timeout=CELL_TIMEOUT, kernel_name="python3",
                       allow_errors=True).execute()
    except CellTimeoutError:
        crashed.append(f"TIMEOUT after {CELL_TIMEOUT}s -- runaway loop?")
    except DeadKernelError:
        # A runaway loop that allocates can kill the kernel before the timeout
        # fires: memory loses the race against wall clock.
        crashed.append("KERNEL DIED -- runaway allocation? (OOM before timeout)")
    out = ""
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                crashed.append(f"cell {i}: {o.get('ename')}: {(o.get('evalue') or '')[:80]}")
            t = o.get("text") or (o.get("data", {}) or {}).get("text/plain") or ""
            out += "".join(t) if isinstance(t, list) else str(t)
    return tally(out), crashed


bad = 0
for fn in sorted(f for f in os.listdir(LABDIR) if f.endswith(".ipynb")):
    path = os.path.join(LABDIR, fn)
    (p_todo, p_pass, p_fail), p_crash = run_plain(path)
    (k_todo, k_pass, k_fail), k_crash = run_kernel(path)

    problems = [f"kernel  {c}" for c in k_crash] + [f"plain   {c}" for c in p_crash]
    if (p_todo, p_pass, p_fail) != (k_todo, k_pass, k_fail):
        problems.append(
            "MISMATCH plain vs kernel -- the notebook depends on IPython-only "
            f"semantics: plain={p_todo}/{p_pass}/{p_fail} kernel={k_todo}/{k_pass}/{k_fail} "
            "(todo/pass/fail)")
    if k_todo == 0:
        problems.append("no [TODO] in an untouched lab -- blanks are not raising NameError")

    status = "OK    " if not problems else "BROKEN"
    print(f"[{status}] {fn:44} {k_todo} todo, {k_pass} pass, {len(k_crash)} crash")
    for p in problems:
        print("           " + p)
        bad += 1

sys.exit(1 if bad else 0)
