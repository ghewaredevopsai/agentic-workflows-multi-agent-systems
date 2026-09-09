#!/usr/bin/env python3
"""Execute every solution notebook's code cells and assert a clean score.

Live-model cells self-skip because LAB_LLM_BASE_URL is unset, so this runs offline.
Also checks that each lab has blanks and that no solution does.
"""
import io, json, os, sys, contextlib, re, tokenize, token

HERE = os.path.dirname(os.path.abspath(__file__))
LABDIR = os.path.abspath(os.path.join(HERE, ".."))
SOLDIR = os.path.join(LABDIR, "solutions")

# Every name the notebooks resolve a model from, including the LITELLM_* fallback and
# OPENAI_API_BASE. Miss one and this "offline" verifier makes real model calls on the
# cluster -- where all of them are set -- so the run stops being free and deterministic.
for v in ("LAB_LLM_BASE_URL", "LAB_LLM_MODEL",
          "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_MODEL",
          "LITELLM_BASE_URL", "LITELLM_MODEL",
          "JIRA_MCP_URL", "JIRA_MCP_AUTH", "JIRA_MCP_PROJECT"):
    os.environ.pop(v, None)

# A WALKTHROUGH lab has nothing to fill in and nothing to score: the participant notebook
# and the solution are the same file on purpose (Lab 4.1 exists so a participant SEES a real
# MCP server work before Module 4 asks them to build one). For those the rules invert --
# zero blanks on both sides, the two files byte-identical, and no score line expected. Every
# cell is guarded on JIRA_MCP_URL/JIRA_MCP_AUTH, which this verifier unsets, so the whole
# notebook must run clean and print its skip message instead.
WALKTHROUGH = {"lab-4-01-opencode-jira-over-mcp.ipynb"}

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
    m = re.search(r"Self-check: (\d+)/(\d+)", out)
    # A solution that trips guard() has an unfilled name in it -- the cell is ungraded, so
    # nothing else here would notice. This is how a broken "Run it for real" cell hides.
    n_guard = out.count("a blank above is still unfilled")
    if fn in WALKTHROUGH:
        # no self-checks by design; it passes if it executed cleanly and skipped its
        # live steps rather than reaching the network
        skipped = out.count("skipped - see the preflight cell")
        ok = n_fail == 0 and n_todo == 0 and n_guard == 0
        print(f"[{'OK    ' if ok else 'BROKEN'}] {fn:44} walkthrough: ran clean, "
              f"{skipped} step(s) self-skipped with no credentials")
    else:
        ok = n_fail == 0 and n_todo == 0 and n_guard == 0 and m and m.group(1) == m.group(2)
        print(f"[{'OK    ' if ok else 'BROKEN'}] {fn:44} {n_pass} pass, {n_fail} fail, {n_todo} todo, "
              f"score {m.group(0) if m else 'MISSING'}")
    if not ok:
        fails += 1
        if n_guard:
            print(f"           {n_guard} guard() message(s) in a SOLUTION -- an unfilled name in "
                  f"an ungraded cell")
        for line in out.splitlines():
            if "[FAIL]" in line or "[TODO]" in line:
                print("           " + line)

def count_blanks(nb):
    """Unfilled blanks = BLANK used as a bare NAME. Not the word in a comment or a string.

    This was a plain substring count, which cannot tell an unfilled blank from a guard that
    legitimately mentions the sentinel -- a lab that compares a Field description against the
    literal "BLANK" would read as several unfilled blanks in the SOLUTION and flip to BROKEN.
    Tokenising counts only real identifier uses, so it still catches every genuine leftover.
    """
    total = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            total += sum(1 for t in tokenize.generate_tokens(io.StringIO(src).readline)
                         if t.type == token.NAME and t.string == "BLANK")
        except (tokenize.TokenError, IndentationError, SyntaxError):
            total += src.count("BLANK")      # unparseable cell: fall back to the blunt count
    return total


print("\n--- blanks ---")
for fn in sorted(f for f in os.listdir(LABDIR) if f.endswith(".ipynb")):
    lab = json.load(open(os.path.join(LABDIR, fn)))
    sol = json.load(open(os.path.join(SOLDIR, fn)))
    lb = count_blanks(lab)
    sb = count_blanks(sol)
    if fn in WALKTHROUGH:
        same = json.dumps(lab, sort_keys=True) == json.dumps(sol, sort_keys=True)
        good = lb == 0 and sb == 0 and same
        note = "walkthrough: no blanks, lab == solution"
        if not same:
            note = "walkthrough: lab and solution DIFFER"
        elif lb or sb:
            note = f"walkthrough: expected 0 blanks, found {lb}/{sb}"
        print(f"[{'OK    ' if good else 'BROKEN'}] {fn:44} {note}")
    else:
        good = lb > 0 and sb == 0
        print(f"[{'OK    ' if good else 'BROKEN'}] {fn:44} {lb} blanks in lab, {sb} in solution")
    if not good:
        fails += 1

sys.exit(1 if fails else 0)
