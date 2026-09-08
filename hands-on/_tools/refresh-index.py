#!/usr/bin/env python3
"""Rebuild each module index page's lab cards from the generated notebooks.

The cards carried hand-written titles, bullets and counts, so they drifted the moment
the labs were rewritten. Now they are derived.
"""
import json, os, re, io, tokenize, token, sys

BLURBS = {
 "lab-1-01": "Write the agent loop by hand in fifteen lines, then hand the whole thing to <code>create_agent</code>.",
 "lab-1-02": "Each of the four blocks as a real LangChain object &mdash; including a tool that refuses.",
 "lab-1-03": "Same model, same functions, two sets of words. Measure what the words are worth.",
 "lab-1-04": "One agent or three? Build both, measure both, and let the team lose if it loses.",
 "lab-1-05": "The rubric that answers &ldquo;do we need multiple agents?&rdquo; before anyone starts building.",
 "lab-2-01": "Compose your first LCEL chain, then measure what &ldquo;think step by step&rdquo; is worth here.",
 "lab-2-02": "Write the ReAct parser, break it, then meet the version where the argument is schema-checked.",
 "lab-2-03": "The failure that looks like bad luck, and the one that looks like a reasoning bug.",
 "lab-2-04": "Branch concurrently, score with a structured judge, and stop reflecting when it stops paying.",
 "lab-2-05": "Four architectures, one eval set, and a bar written before you look at any result.",
 "lab-3-01": "State, one node, two edges, <code>compile()</code>. No model in it &mdash; and that is the point.",
 "lab-3-02": "Three nodes in a row, and the discovery that your edges are a dependency order.",
 "lab-3-03": "A routing function is a pure function returning a name. Module 5's supervisor, early.",
 "lab-3-04": "Declare the merge rule once. Then run two nodes in one step and watch it matter.",
 "lab-3-05": "A cycle is an edge pointing backwards. One comparison is what stops it being a hang.",
 "lab-3-06": "One keyword argument, and resume, inspect, rewind and audit all become possible.",
 "lab-3-07": "Pause before the irreversible step, let a person edit the state, then meet the rewind surprise.",
 "lab-3-08": "Assemble all of it &mdash; and gate one branch rather than the whole graph."
}

SCORE_OLD = ("<code>[FAIL]</code> or <code>[TODO]</code>, and the last cell prints your\n"
             "      <code>Score</code>.</p>")
SCORE_NEW = ("<code>[FAIL]</code> or <code>[TODO]</code>, and the last cell prints a\n"
             "      <code>Self-check</code> tally.</p>")

HOW_OLD = re.compile(
    r"<p><strong>Your score never depends on a live model\.</strong>.*?</p>", re.S)
HOW_NEW = ("<p><strong>You write real LangChain and LangGraph code in every lab.</strong> The\n"
           "      <strong>Self-check</strong> cells assert on the objects you build &mdash; a bound tool,\n"
           "      a compiled graph, a validated schema &mdash; so they are deterministic and do not need\n"
           "      the model. Cells marked <strong>Run it for real</strong> put your code in front of the\n"
           "      sandbox model; that is the part worth watching. <em>Run All</em> is always safe on an\n"
           "      untouched notebook, and the score line is feedback, not a grade.</p>")


def count_blanks(nb):
    total = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            total += sum(1 for t in tokenize.generate_tokens(io.StringIO(src).readline)
                         if t.type == token.NAME and t.string == "BLANK")
        except Exception:
            total += src.count("BLANK")
    return total


def header_of(nb):
    md = "".join(nb["cells"][0]["source"])
    title = re.search(r"^# (Lab [\d.]+) &mdash; (.+)$", md, re.M)
    level = re.search(r"\*\*Level:\*\* (.+?) &nbsp;", md)
    mins = re.search(r"\*\*Est\. time:\*\* (\d+) min", md)
    bullets = re.findall(r"^- (.+)$", md, re.M)
    return title.group(1), title.group(2), level.group(1), mins.group(1), bullets


def card(slug, nb_lab, nb_sol, fn):
    num, title, level, mins, bullets = header_of(nb_lab)
    blanks = count_blanks(nb_lab)
    checks = sum("".join(c["source"]).count("check(")
                 for c in nb_sol["cells"] if c["cell_type"] == "code")
    lis = "\n".join(f"      <li>{b}</li>" for b in bullets)
    return (f'  <div class="lab">\n'
            f'    <div class="tags">{mins} min<span class="lvl">{level}</span></div>\n'
            f'    <h3><a href="{fn}">{num} &mdash; {title}</a></h3>\n'
            f'    <p>{BLURBS[slug]}</p>\n'
            f'    <ul>\n{lis}\n    </ul>\n'
            f'    <div class="foot">{blanks} blank{"" if blanks == 1 else "s"} &middot; {checks} checks &middot; solution:\n'
            f'      <a href="solutions/{fn}">solutions/{fn}</a></div>\n'
            f'  </div>')


for m in (1, 2, 3):
    d = f"module-{m}"
    page = open(f"{d}/index.html", encoding="utf-8").read()
    if HOW_NEW not in page:                      # idempotent: safe to re-run
        page, n = HOW_OLD.subn(HOW_NEW, page)
        assert n == 1, f"{d}: how-block not found"
    if SCORE_OLD in page:
        page = page.replace(SCORE_OLD, SCORE_NEW)
    assert SCORE_NEW in page, f"{d}: score sentence not updated"

    files = sorted(f for f in os.listdir(d) if re.fullmatch(rf"lab-{m}-\d\d-.+\.ipynb", f))
    assert len(files) == (8 if m == 3 else 5), files
    cards = []
    for fn in files:
        slug = fn[:8]
        cards.append(card(slug, json.load(open(f"{d}/{fn}")),
                          json.load(open(f"{d}/solutions/{fn}")), fn))

    start = page.index('  <div class="lab">')
    tail_start = page.index('  <div class="note">', start)   # the block that follows the cards
    page = page[:start] + "\n\n".join(cards) + "\n\n" + page[tail_start:]
    open(f"{d}/index.html", "w", encoding="utf-8").write(page)
    print(f"{d}/index.html: {len(cards)} cards rebuilt")
