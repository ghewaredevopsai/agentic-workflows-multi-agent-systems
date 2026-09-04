#!/usr/bin/env python3
"""
Generate Module 6 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-6-0N-*.ipynb and ../solutions/

Design rules (from Training/courses/CLAUDE.md and this course's stack):
  * Graded cells are pure Python -- they never call an LLM, so a self-check is
    deterministic and a flaky endpoint can never fail a participant.
  * Live-model cells are clearly marked, guarded, and never crash Run All.
  * "BLANK" marks a blank; an unfilled blank raises NameError and prints [TODO].
    NOT three underscores: IPython PREDEFINES _, __ and ___ as its output history
    (they start as ""), so under a real Jupyter kernel that token is a defined empty
    string, not an undefined name. The NameError never fires, [TODO] silently becomes
    [FAIL], and a blank used as a loop guard is falsy forever -- lab 1.1 spun in
    `while True` until the pod was OOM-killed. Plain-exec verifiers cannot see any
    of this, which is why verify_labs.py now runs cells through IPython.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABDIR = os.path.abspath(os.path.join(HERE, ".."))
SOLDIR = os.path.join(LABDIR, "solutions")


# --------------------------------------------------------------------------- #
# cell helpers
# --------------------------------------------------------------------------- #
class Cell:
    def __init__(self, kind, lab, sol=None):
        self.kind, self.lab, self.sol = kind, lab, sol if sol is not None else lab

def md(text):
    return Cell("markdown", text)

def code(lab, sol=None):
    return Cell("code", lab, sol)


def to_source(text):
    """nbformat wants a list of lines, each keeping its trailing newline."""
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build_notebook(cells, solution):
    out = []
    for i, c in enumerate(cells):
        src = c.sol if solution else c.lab
        # nbformat 4.5 requires a stable per-cell id
        cell = {"id": f"cell-{i:02d}", "cell_type": c.kind, "metadata": {}, "source": to_source(src)}
        if c.kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# shared cells
# --------------------------------------------------------------------------- #
def header(num, title, level, minutes, bullets, note):
    items = "\n".join("- " + b for b in bullets)
    return md(f"""
# Lab 6.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 2 &middot; Module 6 &mdash; Agentic RAG**

### What you'll do
{items}

> **How this lab works.** Fill every `BLANK`, then run the **Self-check** cell under each section.
> Graded cells are plain Python and never call a model, so your score never depends on a
> live endpoint. Cells marked **Run it for real** do call the sandbox model; if it is not
> reachable they print how to fix it instead of crashing.

{note}
""")


SETUP_COMMON = '''
# ---------------------------------------------------------------- Setup: run me first
import os, json, time, textwrap
from typing import Any, Callable

WORK = os.path.join("/tmp", "awmas-lab-6-{num:02d}")
os.makedirs(WORK, exist_ok=True)

# ---- self-check plumbing -------------------------------------------------
_results = []

def check(name: str, fn: Callable[[], Any], hint: str = "") -> None:
    """[PASS] / [FAIL] / [TODO] for one assertion. An unfilled blank prints [TODO]."""
    try:
        ok = bool(fn())
    except NameError:
        print(f"[TODO] {{name}}")
        _results.append(None)
        return
    except Exception as exc:
        print(f"[FAIL] {{name}} -- {{type(exc).__name__}}: {{exc}}")
        _results.append(False)
        return
    print(("[PASS] " if ok else "[FAIL] ") + name + ("" if ok else (" -- " + hint if hint else "")))
    _results.append(ok)

def guard(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run fn(). If a blank above is still unfilled, say so and carry on -- never crash Run All."""
    try:
        return fn()
    except NameError:
        print("(a blank above is still unfilled -- fill it in, then re-run this cell)")
        return default

def score() -> None:
    done = [r for r in _results if r is not None]
    passed = sum(1 for r in done if r)
    todo = sum(1 for r in _results if r is None)
    print(f"\\nScore: {{passed}}/{{len(_results)}}" + (f"   ({{todo}} still TODO)" if todo else ""))

# ---- the sandbox model ---------------------------------------------------
# Your sandbox already has an LLM configured -- nothing to install, no key to register.
# These two values are read from the environment so this notebook never hardcodes an endpoint.
LLM_BASE_URL = (os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LITELLM_BASE_URL"))
LLM_MODEL    = (os.environ.get("LAB_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
                or os.environ.get("LITELLM_MODEL"))
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "sandbox")

def llm_ready() -> bool:
    if not LLM_BASE_URL or not LLM_MODEL:
        print("Model not configured. In a sandbox terminal run `env | grep -i llm` and set:")
        print("  export LAB_LLM_BASE_URL=...    # the gateway URL from your welcome sheet")
        print("  export LAB_LLM_MODEL=...       # the model name from your welcome sheet")
        return False
    return True

_llm = None
def get_llm(temperature: float = 0.0):
    """A LangChain chat model pointed at the sandbox gateway (OpenAI-compatible)."""
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL,
                          api_key=LLM_API_KEY, temperature=temperature)
    return _llm

def ask(prompt: str, system: str | None = None) -> str:
    """One stateless call. Returns text, or an error string -- never raises."""
    try:
        msgs = ([("system", system)] if system else []) + [("human", prompt)]
        return get_llm().invoke(msgs).content
    except Exception as exc:
        return f"<model unavailable: {{type(exc).__name__}}: {{exc}}>"

print("work dir:", WORK)
print("model   :", LLM_MODEL or "(not configured -- graded cells still work)")
'''


def setup(num, extra=""):
    return code(SETUP_COMMON.format(num=num) + extra)


# --------------------------------------------------------------------------- #
# the shared synthetic domain -- one use case runs through all five labs
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# One domain runs through all five Module 6 labs -- the same payment exceptions, now answered
# from a corpus the agent has to go and read.
# Nothing here is real data and nothing leaves this notebook.

LEDGER = {
    "PMT-1001": {"amount": 250000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "settled",  "value_date": "2026-09-01", "reason_code": None},
    "PMT-1002": {"amount":  48250.75, "ccy": "EUR", "counterparty": "ACME-EU",
                 "status": "failed",   "value_date": "2026-09-02", "reason_code": "INSUFFICIENT_FUNDS"},
    "PMT-1003": {"amount": 990000.00, "ccy": "USD", "counterparty": "ZENITH",
                 "status": "held",     "value_date": "2026-09-02", "reason_code": "LIMIT_BREACH"},
    "PMT-1004": {"amount":   1200.00, "ccy": "GBP", "counterparty": "ACME-UK",
                 "status": "failed",   "value_date": "2026-09-03", "reason_code": "INVALID_IBAN"},
    "PMT-1005": {"amount": 750000.00, "ccy": "USD", "counterparty": "NORTHWIND",
                 "status": "held",     "value_date": "2026-09-03", "reason_code": "SANCTIONS_REVIEW"},
}

POLICY = {
    "INSUFFICIENT_FUNDS": "Retry once after 24h. If it fails again, notify the client desk. No manual funding.",
    "LIMIT_BREACH":       "Payments above USD 500,000 need Treasury approval before release.",
    "INVALID_IBAN":       "Return to originator with code R04. Never repair beneficiary details in-house.",
    "SANCTIONS_REVIEW":   "Hold. Compliance decides. Operations must not release or cancel.",
}

# Which reason codes may an agent resolve on its own, and which need a human?
NEEDS_HUMAN = {"LIMIT_BREACH", "SANCTIONS_REVIEW"}

print(f"{len(LEDGER)} payments, {len(POLICY)} policy rules loaded")
'''




# the two tools from Lab 1.2 of Module 1, carried forward so each notebook stands alone
CARRIED_TOOLS = '''
# ------------------------------------------------- carried forward from Lab 1.2 of Module 1
# The tools you wrote in Lab 1.2 of Module 1. Nothing to fill in -- they are here so this
# notebook runs on its own. Note the docstrings: they name the case AND the boundary.

def lookup_payment(ref: str) -> str:
    """Return the ledger record for one payment reference such as 'PMT-1002'.

    Use when you need the status, amount, counterparty or reason code of a specific payment.
    Not for searching across payments.
    """
    record = LEDGER.get(ref)
    if record is None:
        return f"no payment found with reference {ref!r}"
    return json.dumps({"ref": ref, **record})


def policy_for(reason_code: str) -> str:
    """Return the operating policy for one failure reason code, e.g. 'LIMIT_BREACH'.

    Use after you know why a payment failed and need to know what to do about it.
    """
    return POLICY.get(reason_code, f"no policy on file for reason code {reason_code!r}")


TOOLS = {"lookup_payment": lookup_payment, "policy_for": policy_for}
print("carried forward:", ", ".join(TOOLS))
'''





# the corpus these labs retrieve from -- shared by all five
CORPUS = '''
# ------------------------------------------------- the corpus (synthetic, self-contained)
# Two short operating documents. Read 3.2: the rule and the exception that qualifies it are
# adjacent sentences, which is the whole of Lab 6.1's first lesson. Note also what is NOT
# here -- there is nothing about FX or hedging anywhere, and Lab 6.4 needs that gap.

DOCS = {
    "ops-runbook-v4.md": """
## 3.1 Insufficient funds
A payment returned INSUFFICIENT_FUNDS is retried once after 24 hours. If the retry also fails,
notify the client desk. Operations must not fund the account manually.

## 3.2 Limit breaches
Payments above USD 500,000 require Treasury approval before release. This does not apply to
intra-group transfers, which settle same-day without any approval.

## 3.3 Invalid beneficiary details
A payment returned INVALID_IBAN is returned to the originator with code R04. Beneficiary
details are never repaired in-house.

## 3.4 Sanctions review
A payment held for SANCTIONS_REVIEW is decided by Compliance. Operations must not release or
cancel it under any circumstances.
""",
    "escalation-policy-v2.md": """
## 1 Approval authority
A duty manager may approve a release up to USD 250,000. Above that figure Treasury approval is
required, and must be recorded against the payment reference.

## 2 Escalation timers
If an approver has not responded within 15 minutes, escalate to the Treasury lead, and after a
further 15 minutes to the head of operations.
""",
}

print(f"{len(DOCS)} documents, {sum(len(d) for d in DOCS.values())} characters")
'''


# =========================================================================== #
# Lab 6.1 -- a retriever you can inspect
# =========================================================================== #
LAB1 = [
    header(1, "A Retriever You Can Inspect", "Intermediate", 30,
           ["Chunk a document two ways, and watch one of them make an answer unreachable",
            "Score, rank, and see that top-k always returns k &mdash; whatever is in the corpus",
            "Build the score floor that turns &lsquo;the least bad thing&rsquo; into an empty result",
            "Scope with metadata, which is what production retrieval actually looks like"],
           "> **No embeddings here, deliberately.** The scoring function is a stand-in so every\n"
           "> check is exact and offline. Chunking, ranking, floors and filters are the same\n"
           "> whatever computes the similarity &mdash; and they decide more than the model does."),
    setup(1),
    code(CORPUS),

    md("""
## Concept

A retriever is four decisions, and only one of them is the embedding model:

1. **How the corpus is cut up.** Decides what can ever be returned together.
2. **How a chunk is scored** against a query. This is the part people think is the whole thing.
3. **How many come back, and how bad they are allowed to be.**
4. **What is in scope** before ranking starts.

This lab builds 1, 3 and 4 exactly, and 2 approximately &mdash; because the approximate version is
enough to see everything that matters, and it makes every check deterministic.
"""),

    md("""
## Section 1 &mdash; Chunking decides what can be found

Section 3.2 states a rule and then exempts intra-group transfers from it. Cut that in half and no
retriever can ever return the two together, because they are no longer one thing.
"""),
    code('''
import re

SECTION_RE = re.compile(r"^##\\s+(.*)$", re.M)

def chunk_by_chars(source, text, size=120):
    """The naive chunker: cut every `size` characters, meaning be damned."""
    flat = " ".join(text.split())
    return [{"source": source, "section": f"chars {i}-{i + size}", "text": flat[i:i + size]}
            for i in range(0, len(flat), size)]


def chunk_by_section(source, text):
    """One chunk per section, so a rule and the exception that qualifies it stay together."""
    out, parts = [], SECTION_RE.split(text)
    # parts is [preamble, heading, body, heading, body, ...]
    for i in range(1, len(parts) - 1, 2):
        heading, body = parts[i].strip(), " ".join(parts[i + 1].split())
        # TODO: the retrievable text for this chunk. The heading carries words a query will
        # use ("limit breaches"), so it belongs in what gets scored -- not just in the label.
        out.append({"source": source, "section": heading, "text": BLANK})
    return out


def build_index(chunker):
    """Run one chunker over every document."""
    return [c for source, text in DOCS.items() for c in chunker(source, text)]
''', '''
import re

SECTION_RE = re.compile(r"^##\\s+(.*)$", re.M)

def chunk_by_chars(source, text, size=120):
    """The naive chunker: cut every `size` characters, meaning be damned."""
    flat = " ".join(text.split())
    return [{"source": source, "section": f"chars {i}-{i + size}", "text": flat[i:i + size]}
            for i in range(0, len(flat), size)]


def chunk_by_section(source, text):
    """One chunk per section, so a rule and the exception that qualifies it stay together."""
    out, parts = [], SECTION_RE.split(text)
    # parts is [preamble, heading, body, heading, body, ...]
    for i in range(1, len(parts) - 1, 2):
        heading, body = parts[i].strip(), " ".join(parts[i + 1].split())
        out.append({"source": source, "section": heading, "text": heading + " -- " + body})
    return out


def build_index(chunker):
    """Run one chunker over every document."""
    return [c for source, text in DOCS.items() for c in chunker(source, text)]
'''),
    code('''
# --- Self-check: Section 1
def by_section():
    return build_index(chunk_by_section)

def by_chars():
    return build_index(chunk_by_chars)

def limit_chunk(index):
    """The chunk that states the USD 500,000 rule."""
    return next(c for c in index if "500,000" in c["text"])

check("section chunking finds all six sections across the two documents",
      lambda: len(by_section()) == 6)
check("every chunk knows where it came from",
      lambda: all(c["source"] in DOCS and c["section"] for c in by_section()))
check("the heading is part of what gets scored, not just a label",
      lambda: "Limit breaches" in limit_chunk(by_section())["text"],
      "a query says 'limit breach'; if that phrase is only in the label it cannot be matched")
check("the rule and the exception that qualifies it are in ONE chunk",
      lambda: "intra-group" in limit_chunk(by_section())["text"])
check("character chunking splits them apart",
      lambda: "intra-group" not in limit_chunk(by_chars())["text"],
      "after this cut, no retriever on earth can return them together")
check("and that is not a small-chunk problem -- it is a boundary problem",
      lambda: any("intra-group" in c["text"] for c in by_chars()),
      "the exception is still in the index; it is just no longer attached to the rule it qualifies")

def _show():
    print("  by section:", limit_chunk(by_section())["text"][:96], "...")
    print("  by chars  :", limit_chunk(by_chars())["text"][:96], "...")
guard(_show)
'''),

    md("""
## Section 2 &mdash; Rank, and then refuse to

The scoring function below is term overlap. An embedding would score differently and better; it
would not change anything else in this lab, which is the point.

The important part is the last argument: **top-k always returns k**, so the floor is the only thing
standing between you and four confident irrelevant chunks.
"""),
    code('''
STOP = set("""a an the of for is are was were do does did what which who this that these those it
its to in on at by with from about and or not no be been have has had can could should would will
you your we our i me my how why when where there here as if then than so such only just also very
more most some any other""".split())

def terms(text):
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if w not in STOP and len(w) > 1}


def similarity(query, chunk):
    """How well one chunk answers one query, 0.0 to 1.0. A stand-in for a cosine similarity.

    Named `similarity`, not `score` -- this notebook already has a score() that prints your
    marks, and shadowing it would break the last cell of the lab.
    """
    q = terms(query)
    if not q:
        return 0.0
    return len(q & terms(chunk["text"])) / len(q)


def search(query, index, k=4, floor=0.0):
    """Top-k by score, then drop anything that did not clear the floor."""
    scored = sorted(((similarity(query, c), c) for c in index), key=lambda sc: -sc[0])
    top = scored[:k]
    # TODO: the floor is what creates an empty result. Without it every query returns k rows.
    return [{"score": round(s, 3), **c} for s, c in top if BLANK]
''', '''
STOP = set("""a an the of for is are was were do does did what which who this that these those it
its to in on at by with from about and or not no be been have has had can could should would will
you your we our i me my how why when where there here as if then than so such only just also very
more most some any other""".split())

def terms(text):
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if w not in STOP and len(w) > 1}


def similarity(query, chunk):
    """How well one chunk answers one query, 0.0 to 1.0. A stand-in for a cosine similarity.

    Named `similarity`, not `score` -- this notebook already has a score() that prints your
    marks, and shadowing it would break the last cell of the lab.
    """
    q = terms(query)
    if not q:
        return 0.0
    return len(q & terms(chunk["text"])) / len(q)


def search(query, index, k=4, floor=0.0):
    """Top-k by score, then drop anything that did not clear the floor."""
    scored = sorted(((similarity(query, c), c) for c in index), key=lambda sc: -sc[0])
    top = scored[:k]
    return [{"score": round(s, 3), **c} for s, c in top if s >= floor]
'''),
    code('''
# --- Self-check: Section 2
LIMIT_Q = "what approval does a limit breach above 500,000 need"
JPY_Q   = "what is the FX hedging policy for JPY exposure"

check("a real question finds the right section first",
      lambda: search(LIMIT_Q, by_section())[0]["section"].startswith("3.2"))
check("results come back ranked, best first",
      lambda: [r["score"] for r in search(LIMIT_Q, by_section())]
              == sorted((r["score"] for r in search(LIMIT_Q, by_section())), reverse=True))
check("k is respected",
      lambda: len(search(LIMIT_Q, by_section(), k=2)) <= 2)
check("A QUESTION THE CORPUS CANNOT ANSWER STILL RETURNS FOUR RESULTS",
      lambda: len(search(JPY_Q, by_section(), k=4)) == 4,
      "nothing in either document mentions FX or JPY, and four chunks come back anyway")
check("scoring zero is not the same as being excluded",
      lambda: all(r["score"] == 0.0 for r in search(JPY_Q, by_section())),
      "a real embedding never returns exactly zero either -- it returns a small number, and k rows")
check("and none of them is about FX",
      lambda: not any("hedg" in r["text"].lower() for r in search(JPY_Q, by_section())))
check("a floor turns that into an empty result",
      lambda: search(JPY_Q, by_section(), floor=0.25) == [],
      "this is the only thing that lets the agent say 'I could not find it'")
check("and the same floor does not break the good query",
      lambda: len(search(LIMIT_Q, by_section(), floor=0.25)) > 0)
check("the floor has to be chosen against the corpus, not guessed",
      lambda: search(LIMIT_Q, by_section(), floor=0.95) == [],
      "set it too high and every question refuses -- Lab 6.5 measures where it should sit")

def _ranked():
    for q, label in ((LIMIT_Q, "answerable"), (JPY_Q, "not in the corpus")):
        print(f"  [{label}] {q}")
        for r in search(q, by_section()):
            print(f"      {r['score']:.2f}  {r['source']:24} {r['section']}")
        print()
guard(_ranked)
'''),

    md("""
## Section 3 &mdash; Scope before you rank

Most production retrieval is a metadata filter with a similarity search inside it: this version,
this jurisdiction, documents this user is allowed to see. An unfiltered index is a disclosure
waiting to be reported.
"""),
    code('''
def search_scoped(query, index, k=4, floor=0.0, where=None):
    """Narrow by metadata first, then rank inside the scope."""
    where = where or {}
    # TODO: keep a chunk only if it matches EVERY key/value pair in `where`.
    scope = [c for c in index if BLANK]
    return search(query, scope, k=k, floor=floor)
''', '''
def search_scoped(query, index, k=4, floor=0.0, where=None):
    """Narrow by metadata first, then rank inside the scope."""
    where = where or {}
    scope = [c for c in index if all(c.get(key) == val for key, val in where.items())]
    return search(query, scope, k=k, floor=floor)
'''),
    code('''
# --- Self-check: Section 3
check("no filter searches everything",
      lambda: len(search_scoped(LIMIT_Q, by_section()))
              == len(search(LIMIT_Q, by_section())))
check("a source filter restricts the results to that document",
      lambda: all(r["source"] == "escalation-policy-v2.md"
                  for r in search_scoped("who may approve a release", by_section(),
                                         where={"source": "escalation-policy-v2.md"})))
check("and it changes the answer, which is the whole point",
      lambda: search_scoped("who may approve a release", by_section(),
                            where={"source": "escalation-policy-v2.md"})[0]["section"]
              .startswith("1"))
check("filtering to something that does not exist returns nothing, not everything",
      lambda: search_scoped(LIMIT_Q, by_section(), where={"source": "no-such-doc.md"}) == [],
      "a filter that silently falls back to the whole index is how a disclosure happens")
check("every key in the filter has to match, not just one",
      lambda: search_scoped(LIMIT_Q, by_section(),
                            where={"source": "ops-runbook-v4.md",
                                   "section": "no such section"}) == [])
'''),

    md("""
## Run it for real &mdash; with actual embeddings

Everything above used term overlap. This cell puts the same corpus into ChromaDB with a real
embedding model and asks the same two questions, so you can see where semantics beat words.

**First run downloads about 80 MB** of embedding model, so give it a minute. If it is unavailable
the cell says so and the lab still scores.
"""),
    code('''
def with_real_embeddings():
    import chromadb
    index = build_index(chunk_by_section)
    client = chromadb.Client()
    col = client.get_or_create_collection("m6-lab1")
    col.add(ids=[f"c{i}" for i in range(len(index))],
            documents=[c["text"] for c in index],
            metadatas=[{"source": c["source"], "section": c["section"]} for c in index])
    for q in (LIMIT_Q, JPY_Q, "can we push a large payment between our own entities"):
        res = col.query(query_texts=[q], n_results=2)
        print(f"  {q}")
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            print(f"      dist {dist:.3f}  {meta['section']:26} {doc[:52]}")
        print()

try:
    guard(with_real_embeddings)
except Exception as exc:
    print(f"(embeddings unavailable here: {type(exc).__name__}: {exc})")
    print("The graded cells above do not need them.")
'''),
    md("""
### Read it

Look at the third question &mdash; *&ldquo;can we push a large payment between our own entities&rdquo;* &mdash; which
shares almost no words with section 3.2 but means exactly it. Measured on this corpus:

| | top result | is it right? |
|---|---|---|
| term overlap (this lab) | 3.1, 3.3 and 3.4, tied at 0.167 | no &mdash; 3.2 is not even in the top three |
| embeddings (the cell above) | 3.2 Limit breaches, distance 1.073 | yes |

The lexical retriever does not merely score it weakly. It returns three wrong sections, confidently
tied, and the right one never appears. **That** is what the embedding model buys you, and it is
worth having &mdash; nothing in this lab argues otherwise.

Now look at the second question, the one about FX, where the corpus genuinely has nothing.
Embeddings return 3.2 at distance 1.446: further away, still first, still four rows, still no
empty result. The scoring function changed. What did not change is that top-k returns k, that a
badly cut chunk cannot be reassembled, or that an unfiltered index returns things the reader
should not see. Those are the parts you build, and they are the rest of this module.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `chunk_by_chars` splits mid-sentence. Add a 40-character overlap and re-run the Section 1
   checks. Does overlap actually reattach the exception to its rule, or does it just make the
   failure less frequent and harder to find?
2. Sections are uneven &mdash; 3.4 is two sentences, 3.2 is four. Find the section that is too big to
   be a single retrievable idea and split it on meaning. What did you have to decide?
3. Set `floor` to each of 0.1, 0.25 and 0.4 and record, for both questions, whether you got an
   answer and whether it was right. That table is the beginning of Lab 6.5.
"""),
]


# =========================================================================== #
# Lab 6.2 -- retrieval as a tool the agent chooses
# =========================================================================== #
RETRIEVER = '''
# ------------------------------------------------- carried forward from Lab 6.1 (nothing to fill in)
import re

SECTION_RE = re.compile(r"^##\\s+(.*)$", re.M)
STOP = set("""a an the of for is are was were do does did what which who this that these those it
its to in on at by with from about and or not no be been have has had can could should would will
you your we our i me my how why when where there here as if then than so such only just also very
more most some any other""".split())

def terms(text):
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if w not in STOP and len(w) > 1}

def chunk_by_section(source, text):
    out, parts = [], SECTION_RE.split(text)
    for i in range(1, len(parts) - 1, 2):
        heading, body = parts[i].strip(), " ".join(parts[i + 1].split())
        out.append({"source": source, "section": heading, "text": heading + " -- " + body})
    return out

INDEX = [c for source, text in DOCS.items() for c in chunk_by_section(source, text)]

def similarity(query, chunk):
    q = terms(query)
    return len(q & terms(chunk["text"])) / len(q) if q else 0.0

def search(query, index=None, k=4, floor=0.0):
    index = INDEX if index is None else index
    scored = sorted(((similarity(query, c), c) for c in index), key=lambda sc: -sc[0])
    return [{"score": round(s, 3), **c} for s, c in scored[:k] if s >= floor]

print(f"index: {len(INDEX)} chunks")
'''


LAB2 = [
    header(2, "Retrieval as a Tool the Agent Chooses", "Intermediate &rarr; Advanced", 35,
           ["Decide <em>whether</em> to retrieve &mdash; and find the questions where retrieving hurts",
            "Price always-retrieve: the tokens, and the noise it puts next to the real context",
            "Write the query in the corpus's vocabulary instead of passing the user's words through",
            "Compare the pipeline and the agent on the same questions"],
           "> **Builds directly on Lab 6.1's retriever.** Same index, same scoring. What changes is\n"
           "> who decides when it runs and what it is asked."),
    setup(2),
    code(CORPUS),
    code(RETRIEVER),

    md("""
## Concept

A pipeline retrieves once, always, with the user's exact words. That is one decision, made at
build time, applied to every question.

An agent makes three decisions per question, and this lab builds the first two:

- **whether** &mdash; some questions are answered by the conversation, or by arithmetic
- **what to ask for** &mdash; users write in their words, corpora in the organisation's

The third, *whether to ask again*, is Lab 6.3.
"""),

    md("""
## Section 1 &mdash; Whether to retrieve at all

Three kinds of question, and only one of them wants the corpus.
"""),
    code('''
QUESTIONS = [
    # (question, does it need the corpus?)
    ("What approval does a payment above USD 500,000 need?",   True),
    ("What happens when a payment comes back INVALID_IBAN?",   True),
    ("Who decides on a payment held for sanctions review?",    True),
    ("How long before an unanswered approval escalates?",      True),
    ("What is 990,000 minus 500,000?",                         False),
    ("Calculate the difference between the amount and the limit.", False),
    ("Summarise what we just agreed.",                         False),
    ("What did I ask you a moment ago?",                       False),
]

CONVERSATION_HINTS = ("we just", "you said", "a moment ago", "earlier you", "we agreed",
                      "summarise what we", "recap")
ARITHMETIC_HINTS   = ("plus", "minus", "times", "calculate", "subtract", "difference between",
                      "how much is")

def needs_corpus(question: str) -> bool:
    """Should the agent go and read something for this question?

    Two kinds of question do not need the corpus: the ones the conversation has already
    answered, and the ones that are pure arithmetic.
    """
    low = (question or "").lower()
    # TODO: False for those two kinds, True for everything else.
    return BLANK
''', '''
QUESTIONS = [
    # (question, does it need the corpus?)
    ("What approval does a payment above USD 500,000 need?",   True),
    ("What happens when a payment comes back INVALID_IBAN?",   True),
    ("Who decides on a payment held for sanctions review?",    True),
    ("How long before an unanswered approval escalates?",      True),
    ("What is 990,000 minus 500,000?",                         False),
    ("Calculate the difference between the amount and the limit.", False),
    ("Summarise what we just agreed.",                         False),
    ("What did I ask you a moment ago?",                       False),
]

CONVERSATION_HINTS = ("we just", "you said", "a moment ago", "earlier you", "we agreed",
                      "summarise what we", "recap")
ARITHMETIC_HINTS   = ("plus", "minus", "times", "calculate", "subtract", "difference between",
                      "how much is")

def needs_corpus(question: str) -> bool:
    """Should the agent go and read something for this question?

    Two kinds of question do not need the corpus: the ones the conversation has already
    answered, and the ones that are pure arithmetic.
    """
    low = (question or "").lower()
    return not any(h in low for h in CONVERSATION_HINTS + ARITHMETIC_HINTS)
'''),
    code('''
# --- Self-check: Section 1
check("it gets every question in the set right",
      lambda: all(needs_corpus(q) is expected for q, expected in QUESTIONS))
check("a policy question retrieves",
      lambda: needs_corpus("Who decides on a payment held for sanctions review?") is True)
check("arithmetic does not",
      lambda: needs_corpus("What is 990,000 minus 500,000?") is False)
check("nor does a question about the conversation",
      lambda: needs_corpus("Summarise what we just agreed.") is False)
check("half the set needs no corpus at all",
      lambda: sum(1 for _, e in QUESTIONS if not e) == 4,
      "that fraction is the whole argument -- a pipeline retrieves for all eight")
check("an empty question does not crash the decision",
      lambda: isinstance(needs_corpus(""), bool))
'''),

    md("""
## Section 2 &mdash; What always-retrieve costs

Two things, and the second is the one that does not show up on an invoice: tokens spent, and
irrelevant policy prose sitting next to the real context while the model tries to answer.
"""),
    code('''
def tokens_of(results) -> int:
    """A rough token count for retrieved text."""
    return sum(len(r["text"]) // 4 for r in results)


def run_pipeline(question: str) -> dict:
    """Always retrieve, with the user's words, exactly once."""
    hits = search(question, k=4)
    return {"retrieved": hits, "tokens": tokens_of(hits), "asked_for_it": True}


def run_agentic(question: str) -> dict:
    """Retrieve only when the question needs the corpus."""
    if not needs_corpus(question):
        return {"retrieved": [], "tokens": 0, "asked_for_it": False}
    hits = search(question, k=4)
    return {"retrieved": hits, "tokens": tokens_of(hits), "asked_for_it": True}


def wasted_tokens(runner) -> int:
    """Tokens spent retrieving for questions that did not need the corpus."""
    total = 0
    for question, expected in QUESTIONS:
        if not expected:
            # TODO: what this runner spent on a question that needed nothing
            total += BLANK
    return total
''', '''
def tokens_of(results) -> int:
    """A rough token count for retrieved text."""
    return sum(len(r["text"]) // 4 for r in results)


def run_pipeline(question: str) -> dict:
    """Always retrieve, with the user's words, exactly once."""
    hits = search(question, k=4)
    return {"retrieved": hits, "tokens": tokens_of(hits), "asked_for_it": True}


def run_agentic(question: str) -> dict:
    """Retrieve only when the question needs the corpus."""
    if not needs_corpus(question):
        return {"retrieved": [], "tokens": 0, "asked_for_it": False}
    hits = search(question, k=4)
    return {"retrieved": hits, "tokens": tokens_of(hits), "asked_for_it": True}


def wasted_tokens(runner) -> int:
    """Tokens spent retrieving for questions that did not need the corpus."""
    total = 0
    for question, expected in QUESTIONS:
        if not expected:
            total += runner(question)["tokens"]
    return total
'''),
    code('''
# --- Self-check: Section 2
check("the pipeline retrieves for every question",
      lambda: all(run_pipeline(q)["asked_for_it"] for q, _ in QUESTIONS))
check("the agent retrieves for exactly the four that need it",
      lambda: sum(1 for q, _ in QUESTIONS if run_agentic(q)["asked_for_it"]) == 4)
check("the pipeline wastes real tokens on the other four",
      lambda: wasted_tokens(run_pipeline) > 0)
check("the agent wastes none",
      lambda: wasted_tokens(run_agentic) == 0)
check("both answer the corpus questions identically",
      lambda: all(run_pipeline(q)["retrieved"] == run_agentic(q)["retrieved"]
                  for q, e in QUESTIONS if e),
      "the agent is not retrieving less well -- it is retrieving less often")
check("and the noise is the part that does not show on the invoice",
      lambda: len(run_pipeline("Summarise what we just agreed.")["retrieved"]) == 4,
      "four chunks of policy prose, competing with the conversation the answer is actually in")

def _cost():
    p, a = sum(run_pipeline(q)["tokens"] for q, _ in QUESTIONS), \\
           sum(run_agentic(q)["tokens"] for q, _ in QUESTIONS)
    print(f"  pipeline  {p:>5} retrieved tokens   ({wasted_tokens(run_pipeline)} of them wasted)")
    print(f"  agent     {a:>5} retrieved tokens   ({wasted_tokens(run_agentic)} of them wasted)")
guard(_cost)
'''),

    md("""
## Section 3 &mdash; Ask in the corpus's words

Users describe their situation. Documents describe the organisation's rules. The agent's job is
translation &mdash; and it is the cheapest retrieval improvement there is, because it changes nothing
about the index.
"""),
    code('''
# What people say -> what the documents call it
VOCAB = {
    "bounce":          "INSUFFICIENT_FUNDS retry",
    "bounced":         "INSUFFICIENT_FUNDS retry",
    "push it through": "release Treasury approval",
    "push through":    "release Treasury approval",
    "wrong account":   "INVALID_IBAN beneficiary originator",
    "bad iban":        "INVALID_IBAN beneficiary originator",
    "on hold":         "SANCTIONS_REVIEW Compliance",
    "stuck":           "SANCTIONS_REVIEW Compliance",
    "chase":           "escalate approver Treasury lead",
    "our own entities": "intra-group transfers",
}

def rewrite(question: str) -> str:
    """The query the agent sends, which is the question plus the vocabulary it implies."""
    low = (question or "").lower()
    extra = [v for k, v in VOCAB.items() if k in low]
    # TODO: keep the user's words AND add the corpus vocabulary they imply.
    # Dropping the original loses everything the table does not cover.
    return BLANK
''', '''
# What people say -> what the documents call it
VOCAB = {
    "bounce":          "INSUFFICIENT_FUNDS retry",
    "bounced":         "INSUFFICIENT_FUNDS retry",
    "push it through": "release Treasury approval",
    "push through":    "release Treasury approval",
    "wrong account":   "INVALID_IBAN beneficiary originator",
    "bad iban":        "INVALID_IBAN beneficiary originator",
    "on hold":         "SANCTIONS_REVIEW Compliance",
    "stuck":           "SANCTIONS_REVIEW Compliance",
    "chase":           "escalate approver Treasury lead",
    "our own entities": "intra-group transfers",
}

def rewrite(question: str) -> str:
    """The query the agent sends, which is the question plus the vocabulary it implies."""
    low = (question or "").lower()
    extra = [v for k, v in VOCAB.items() if k in low]
    return question + (" " + " ".join(extra) if extra else "")
'''),
    code('''
# --- Self-check: Section 3
VAGUE = [
    ("Why did this one bounce, and do we try again?",            "3.1"),
    ("The client gave us the wrong account number. Now what?",   "3.3"),
    ("It is stuck. Who decides?",                                "3.4"),
    ("Can we push a big one through between our own entities?",  "3.2"),
]

def best_section(query):
    hits = search(query, k=1)
    return hits[0]["section"] if hits else None

check("the rewrite keeps the user's own words",
      lambda: rewrite("Why did this one bounce?").startswith("Why did this one bounce?"),
      "the table cannot cover everything; dropping the original loses whatever it missed")
check("and adds the corpus vocabulary",
      lambda: "INSUFFICIENT_FUNDS" in rewrite("Why did this one bounce?"))
check("a question with no match is passed through unchanged",
      lambda: rewrite("Who approves a release?") == "Who approves a release?")
check("raw vague questions mostly miss",
      lambda: sum(1 for q, want in VAGUE if (best_section(q) or "").startswith(want)) <= 2)
check("rewritten, they all land on the right section",
      lambda: all((best_section(rewrite(q)) or "").startswith(want) for q, want in VAGUE),
      "same index, same scoring, same k -- the only change is who wrote the query")

def _rewrites():
    for q, want in VAGUE:
        print(f"  {q}")
        print(f"      raw       -> {best_section(q)}")
        print(f"      rewritten -> {best_section(rewrite(q))}   (want {want})")
guard(_rewrites)
'''),

    md("""
## Run it for real

Let the model do the rewriting instead of a lookup table &mdash; which is what you would actually
ship, because no table survives contact with real users.
"""),
    code('''
if llm_ready():
    def _model_rewrite():
        vocab_hint = ("The documents use terms like: INSUFFICIENT_FUNDS, INVALID_IBAN, "
                      "SANCTIONS_REVIEW, Treasury approval, intra-group transfer, escalation.")
        for q, want in VAGUE:
            query = ask(f"{vocab_hint}\\n\\nRewrite this into a search query using those terms. "
                        f"Reply with the query alone.\\n\\n{q}",
                        system="Reply with a search query and nothing else.")
            got = best_section((query or "").strip())
            flag = "ok " if (got or "").startswith(want) else "MISS"
            print(f"  [{flag}] {q[:44]:46} -> {got}")
    guard(_model_rewrite)
'''),
    md("""
### Read it

If the model's rewrites land as well as the lookup table's, you have something that generalises to
questions you never enumerated &mdash; at the cost of one model call before every retrieval. That is a
real trade, and Lab 6.5 is where you price it.

Watch for the failure mode too: a rewrite that invents a term the corpus does not contain retrieves
*worse* than the raw question. Same lesson as Module 4 &mdash; a description, or a query, can attract
the wrong thing as easily as the right one.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `needs_corpus` is a keyword list, so it fails on any phrasing you did not think of. Write three
   questions that should not retrieve and that it gets wrong. What does that tell you about
   shipping this as a rule rather than as a model call?
2. There is a third answer besides yes and no: *retrieve, but only if the first attempt at
   answering is thin*. Sketch it, and say what it costs in latency.
3. Combine the two halves: rewrite first, then decide whether to retrieve based on how well the
   rewritten query scores. Does that ordering help, or have you just moved the guess?
"""),
]


# =========================================================================== #
# Lab 6.3 -- adequacy, re-querying and the hop budget
# =========================================================================== #
LAB3 = [
    header(3, "Adequacy, Re-querying and Multi-Hop", "Advanced", 35,
           ["Judge your own retrieval &mdash; does it actually contain what was asked for?",
            "Re-query with a term the first hop taught you",
            "Follow a chain across three hops without inventing a fourth",
            "Stop: a hop budget, a repeat detector, and &lsquo;I could not find it&rsquo; as a real outcome"],
           "> **This is what makes it agentic.** A pipeline retrieves once. Everything in this lab\n"
           "> is the loop that a pipeline cannot have, and the stops that keep it from running away."),
    setup(3),
    code(CORPUS),
    code(RETRIEVER),

    md("""
## Concept

The first retrieval usually returns something. The question is whether it returns *enough*, and
that is answerable without a model: **did what came back contain the thing the question asked
about?**

When it did not, the results still tell you something &mdash; they hand you the corpus's own
vocabulary, which is exactly what the second query needed.
"""),

    md("""
## Section 1 &mdash; Was that enough?

An adequacy test that is honest has to be able to say no. Test it on a retrieval you know is
inadequate before you trust it on one you do not.
"""),
    code('''
def adequate(question: str, results: list, need_terms=None) -> bool:
    """True if the retrieved text covers what the question is about.

    `need_terms` names the things that must appear; when it is None, fall back to the
    question's own content words.
    """
    if not results:
        return False
    need = set(t.lower() for t in need_terms) if need_terms else terms(question)
    covered = terms(" ".join(r["text"] for r in results))
    # TODO: every term the question needs has to appear somewhere in what came back.
    return BLANK


def missing_terms(question: str, results: list, need_terms=None) -> set:
    """What the question asked about that the results never mention."""
    need = set(t.lower() for t in need_terms) if need_terms else terms(question)
    return need - terms(" ".join(r["text"] for r in results))
''', '''
def adequate(question: str, results: list, need_terms=None) -> bool:
    """True if the retrieved text covers what the question is about.

    `need_terms` names the things that must appear; when it is None, fall back to the
    question's own content words.
    """
    if not results:
        return False
    need = set(t.lower() for t in need_terms) if need_terms else terms(question)
    covered = terms(" ".join(r["text"] for r in results))
    return need <= covered


def missing_terms(question: str, results: list, need_terms=None) -> set:
    """What the question asked about that the results never mention."""
    need = set(t.lower() for t in need_terms) if need_terms else terms(question)
    return need - terms(" ".join(r["text"] for r in results))
'''),
    code('''
# --- Self-check: Section 1
check("an empty retrieval is never adequate",
      lambda: adequate("anything at all", []) is False)
check("a retrieval that covers the asked-for term is adequate",
      lambda: adequate("sanctions", search("sanctions review", k=2), need_terms=["sanctions"])
              is True)
check("one that does not is NOT adequate, even though it returned rows",
      lambda: adequate("hedging", search("FX hedging policy", k=4), need_terms=["hedging"])
              is False,
      "four chunks came back and none of them is about hedging -- a length check would pass this")
check("and it names what was missing",
      lambda: "hedging" in missing_terms("hedging", search("FX hedging policy", k=4),
                                         need_terms=["hedging"]))
check("nothing is missing from an adequate retrieval",
      lambda: missing_terms("sanctions", search("sanctions review", k=2),
                            need_terms=["sanctions"]) == set())
check("the missing term is what the next query should be about",
      lambda: len(missing_terms("r04", search("invalid iban", k=2), need_terms=["r04"])) <= 1)
'''),

    md("""
## Section 2 &mdash; The chain

Three hops, and the point is that hop two's query contains a word you could not have known before
hop one ran. That is what &ldquo;multi-hop&rdquo; means &mdash; not three searches, but three searches where
each one is written from the last one's answer.
"""),
    code('''
CODE_RE = re.compile(r"\\b(R\\d{2}|[A-Z]{2,}_[A-Z_]+)\\b")

def follow_up(results: list, asked: str):
    """A new query built from a term the results just taught you, or None if they taught nothing."""
    found = []
    for r in results:
        found += CODE_RE.findall(r["text"])
    fresh = [f for f in found if f.lower() not in (asked or "").lower()]
    # TODO: the next query is the first genuinely new code the results named.
    # Return None when they named nothing you did not already have.
    return BLANK


def multi_hop(question: str, max_hops: int = 3) -> dict:
    """Retrieve, read what came back, and ask again with what it taught you."""
    asked, hops, seen = question, [], set()
    for _ in range(max_hops):
        results = search(asked, k=2)
        hops.append({"query": asked, "sections": [r["section"] for r in results]})
        nxt = follow_up(results, asked)
        if nxt is None:
            return {"hops": hops, "outcome": "exhausted"}
        if nxt in seen:
            return {"hops": hops, "outcome": "repeat"}
        seen.add(nxt)
        asked = nxt
    return {"hops": hops, "outcome": "budget"}
''', '''
CODE_RE = re.compile(r"\\b(R\\d{2}|[A-Z]{2,}_[A-Z_]+)\\b")

def follow_up(results: list, asked: str):
    """A new query built from a term the results just taught you, or None if they taught nothing."""
    found = []
    for r in results:
        found += CODE_RE.findall(r["text"])
    fresh = [f for f in found if f.lower() not in (asked or "").lower()]
    return fresh[0] if fresh else None


def multi_hop(question: str, max_hops: int = 3) -> dict:
    """Retrieve, read what came back, and ask again with what it taught you."""
    asked, hops, seen = question, [], set()
    for _ in range(max_hops):
        results = search(asked, k=2)
        hops.append({"query": asked, "sections": [r["section"] for r in results]})
        nxt = follow_up(results, asked)
        if nxt is None:
            return {"hops": hops, "outcome": "exhausted"}
        if nxt in seen:
            return {"hops": hops, "outcome": "repeat"}
        seen.add(nxt)
        asked = nxt
    return {"hops": hops, "outcome": "budget"}
'''),
    code('''
# --- Self-check: Section 2
check("the first hop on a beneficiary question finds section 3.3",
      lambda: multi_hop("what happens with a wrong beneficiary iban")["hops"][0]["sections"][0]
              .startswith("3.3"))
check("hop two asks about a code hop one taught it",
      lambda: multi_hop("what happens with a wrong beneficiary iban")["hops"][1]["query"]
              in ("R04", "INVALID_IBAN"),
      "that term was not in the question and could not have been -- the corpus supplied it")
check("a question whose answer names no codes stops instead of inventing one",
      lambda: multi_hop("who may approve a release up to 250,000")["outcome"] == "exhausted")
check("the run always reports why it stopped",
      lambda: multi_hop("what happens with a wrong beneficiary iban")["outcome"]
              in ("exhausted", "repeat", "budget"))
check("the hop budget is respected",
      lambda: len(multi_hop("what happens with a wrong beneficiary iban", max_hops=2)["hops"]) <= 2)
check("a term already in the query is not chased again",
      lambda: follow_up(search("INVALID_IBAN", k=2), "INVALID_IBAN") != "INVALID_IBAN")
check("nothing new to chase returns None rather than an empty string",
      lambda: follow_up([{"text": "no codes here at all"}], "x") is None)

def _chain():
    out = multi_hop("what happens with a wrong beneficiary iban")
    for i, h in enumerate(out["hops"], 1):
        print(f"  hop {i}: {h['query'][:48]:50} -> {h['sections']}")
    print("  stopped:", out["outcome"])
guard(_chain)
'''),

    md("""
## Section 3 &mdash; Retrieve, judge, retry

Now put Section 1 and Section 2 together: retrieve, ask whether it was enough, and if it was not,
try again with the corpus's own words &mdash; under a budget, and reporting failure honestly.
"""),
    code('''
def answer_with_retry(question: str, need_terms=None, max_tries: int = 3) -> dict:
    """Retrieve until adequate, or until the budget runs out. Never pretends."""
    tried, query = [], question
    for attempt in range(max_tries):
        results = search(query, k=3)
        tried.append(query)
        if adequate(question, results, need_terms):
            return {"outcome": "answered", "query": query, "results": results,
                    "attempts": attempt + 1}
        nxt = follow_up(results, query)
        if nxt is None or nxt in tried:
            break
        query = nxt
    return {"outcome": "not found", "query": query, "results": [],
            "attempts": len(tried),
            "missing": sorted(missing_terms(question, search(question, k=3), need_terms))}
'''),
    code('''
# --- Self-check: Section 3
check("an answerable question is answered",
      lambda: answer_with_retry("what does a sanctions review need",
                                need_terms=["sanctions", "compliance"])["outcome"] == "answered")
check("and it did not need many attempts",
      lambda: answer_with_retry("what does a sanctions review need",
                                need_terms=["sanctions", "compliance"])["attempts"] <= 2)
check("an unanswerable question ends as 'not found', not as a wrong answer",
      lambda: answer_with_retry("what is the JPY hedging policy",
                                need_terms=["hedging"])["outcome"] == "not found")
check("and it says what was missing",
      lambda: "hedging" in answer_with_retry("what is the JPY hedging policy",
                                             need_terms=["hedging"])["missing"],
      "'the corpus has nothing on hedging' is a useful answer; 'I don't know' is not")
check("it returns no results when it did not find them",
      lambda: answer_with_retry("what is the JPY hedging policy",
                                need_terms=["hedging"])["results"] == [],
      "handing back the four irrelevant chunks anyway is how a refusal becomes a hallucination")
check("the attempt count never exceeds the budget",
      lambda: answer_with_retry("what is the JPY hedging policy", need_terms=["hedging"],
                                max_tries=2)["attempts"] <= 2)

def _both():
    for q, need in (("what does a sanctions review need", ["sanctions", "compliance"]),
                    ("what is the JPY hedging policy", ["hedging"])):
        out = answer_with_retry(q, need_terms=need)
        extra = f"  missing: {out.get('missing')}" if out["outcome"] == "not found" else ""
        print(f"  {out['outcome']:10} in {out['attempts']} attempt(s)  {q[:38]}{extra}")
guard(_both)
'''),

    md("""
## Run it for real

Let the model judge adequacy instead of the term check, on the same two questions. The one to
watch is the second: a model asked &ldquo;is this enough?&rdquo; about four irrelevant chunks has every
incentive to say yes.
"""),
    code('''
if llm_ready():
    def _judge():
        for q in ("what does a sanctions review need", "what is the JPY hedging policy for us"):
            results = search(q, k=3)
            context = "\\n".join(f"- [{r['section']}] {r['text'][:150]}" for r in results)
            verdict = ask(f"Question: {q}\\n\\nRetrieved:\\n{context}\\n\\n"
                          "Can this question be answered from the retrieved text alone? "
                          "Reply YES or NO, then one short sentence.",
                          system="Begin your reply with YES or NO.")
            print(f"  {q}")
            print(f"      model: {verdict.strip()[:150]}")
            print(f"      term check: {'adequate' if adequate(q, results) else 'not adequate'}")
            print()
    guard(_judge)
'''),
    md("""
### Read it

If the model says YES to the FX question, you have watched the failure this lab exists to prevent:
the retrieval was inadequate, the judge was the same kind of thing that will write the answer, and
nothing stopped it.

The term check is crude and cannot be talked round. In production you want both &mdash; the cheap
mechanical check as a floor, and the model for the judgements the check is too blunt to make.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `adequate` requires *every* term. Make it a fraction &mdash; three-quarters covered is enough &mdash;
   and find the question where that change gives you a confident wrong answer.
2. `follow_up` chases reason codes because that is what this corpus is made of. What is the
   equivalent handle in your corpus &mdash; a ticket id, a product code, a section number? Write the
   regex and see how far a chain gets.
3. Give `multi_hop` a wall-clock deadline as well as a hop budget, then make one search slow.
   Which stop fires first, and which one would you actually have wanted?
"""),
]


# =========================================================================== #
# Lab 6.4 -- citations bound to spans, and refusing
# =========================================================================== #
LAB4 = [
    header(4, "Citations Bound to Spans, and Refusing", "Advanced", 40,
           ["Bind every claim to the exact characters that support it",
            "Drop a claim that cannot name its source &mdash; before it ships, not after",
            "Refuse when the corpus cannot answer, and say what is missing",
            "Prove that neither behaviour depends on the model choosing to co-operate"],
           "> **Extractive grounding.** Every claim here is a quotation, so the binding is exact\n"
           "> and a citation is checkable by string comparison. Looser generation needs the\n"
           "> faithfulness score from Lab 6.5 &mdash; but this is the version you can prove."),
    setup(4),
    code(CORPUS),
    code(RETRIEVER),

    md("""
## Concept

Two behaviours a regulated client will ask about, and both have to be **mechanisms** rather than
requests, because a request is something the model can decline to honour on any given run.

- **Citation** &mdash; not &ldquo;here are the documents that were in context&rdquo;, but *this claim came from
  these characters of that section*.
- **Refusal** &mdash; not &ldquo;the model decided it did not know&rdquo;, but *nothing cleared the floor, so
  there is nothing to answer from*.
"""),

    md("""
## Section 1 &mdash; Bind the claim to the characters

A citation that names a document proves nothing: the document was in the context whatever the
model wrote. A span is checkable.
"""),
    code('''
def normalise(text: str) -> str:
    return " ".join((text or "").split()).lower()


def find_span(claim: str, chunk: dict):
    """The (start, end) character range in the chunk that supports this claim, or None."""
    hay, needle = normalise(chunk["text"]), normalise(claim)
    i = hay.find(needle)
    # TODO: the range the claim occupies, or None when the chunk does not contain it.
    return BLANK


def bind(claim: str, results: list):
    """Attach the first retrieved chunk that actually contains this claim."""
    for r in results:
        span = find_span(claim, r)
        if span:
            return {"claim": claim, "source": r["source"], "section": r["section"], "span": span}
    return None
''', '''
def normalise(text: str) -> str:
    return " ".join((text or "").split()).lower()


def find_span(claim: str, chunk: dict):
    """The (start, end) character range in the chunk that supports this claim, or None."""
    hay, needle = normalise(chunk["text"]), normalise(claim)
    i = hay.find(needle)
    return (i, i + len(needle)) if i >= 0 else None


def bind(claim: str, results: list):
    """Attach the first retrieved chunk that actually contains this claim."""
    for r in results:
        span = find_span(claim, r)
        if span:
            return {"claim": claim, "source": r["source"], "section": r["section"], "span": span}
    return None
'''),
    code('''
# --- Self-check: Section 1
LIMIT_HITS = search("limit breach approval above 500,000", k=3)
SUPPORTED  = "Payments above USD 500,000 require Treasury approval before release"
INVENTED   = "Payments above USD 500,000 may be released by the duty manager"

check("a supported claim finds its span",
      lambda: bind(SUPPORTED, LIMIT_HITS) is not None)
check("and the span points into the right section",
      lambda: bind(SUPPORTED, LIMIT_HITS)["section"].startswith("3.2"))
check("the span is a real character range",
      lambda: bind(SUPPORTED, LIMIT_HITS)["span"][1] > bind(SUPPORTED, LIMIT_HITS)["span"][0])
check("and it can be checked by slicing the source text",
      lambda: normalise(SUPPORTED) in
              normalise(next(r["text"] for r in LIMIT_HITS
                             if r["section"] == bind(SUPPORTED, LIMIT_HITS)["section"])),
      "an auditor follows one link; the check is a string comparison, not a judgement")
check("AN INVENTED CLAIM BINDS TO NOTHING",
      lambda: bind(INVENTED, LIMIT_HITS) is None,
      "it is plausible, it is about the retrieved topic, and it is not in the text")
check("whitespace differences do not break a real citation",
      lambda: bind("Payments above USD 500,000\\n   require Treasury approval", LIMIT_HITS)
              is not None)
'''),

    md("""
## Section 2 &mdash; No span, no claim

The binding is only worth something if an unbound claim is *dropped*. Otherwise you have added a
field, not a control.
"""),
    code('''
def compose(claims: list, results: list) -> dict:
    """Keep only the claims that can name their source. Report what was dropped."""
    bound = [(c, bind(c, results)) for c in claims]
    # TODO: keep the ones that bound, and record the ones that did not.
    kept = [b for c, b in bound if BLANK]
    dropped = [c for c, b in bound if b is None]
    return {"claims": kept, "dropped": dropped,
            "citations": [f"{b['source']}#{b['section']} [{b['span'][0]}:{b['span'][1]}]"
                          for b in kept]}
''', '''
def compose(claims: list, results: list) -> dict:
    """Keep only the claims that can name their source. Report what was dropped."""
    bound = [(c, bind(c, results)) for c in claims]
    kept = [b for c, b in bound if b is not None]
    dropped = [c for c, b in bound if b is None]
    return {"claims": kept, "dropped": dropped,
            "citations": [f"{b['source']}#{b['section']} [{b['span'][0]}:{b['span'][1]}]"
                          for b in kept]}
'''),
    code('''
# --- Self-check: Section 2
DRAFT = [SUPPORTED,
         "This does not apply to intra-group transfers",
         INVENTED]

check("the two supported claims survive",
      lambda: len(compose(DRAFT, LIMIT_HITS)["claims"]) == 2)
check("the invented one is dropped",
      lambda: compose(DRAFT, LIMIT_HITS)["dropped"] == [INVENTED])
check("every surviving claim has a citation",
      lambda: len(compose(DRAFT, LIMIT_HITS)["citations"])
              == len(compose(DRAFT, LIMIT_HITS)["claims"]))
check("the citation names a section and a character range",
      lambda: "#3.2" in compose(DRAFT, LIMIT_HITS)["citations"][0]
              and "[" in compose(DRAFT, LIMIT_HITS)["citations"][0])
check("the exception survives alongside the rule, because Lab 6.1 chunked them together",
      lambda: any("intra-group" in b["claim"] for b in compose(DRAFT, LIMIT_HITS)["claims"]),
      "chunk them apart and this claim becomes unciteable, so this control would delete it")
check("dropping is silent to the reader but visible to you",
      lambda: compose(DRAFT, LIMIT_HITS)["dropped"] != [],
      "what got dropped is the most interesting log line in the system")
'''),

    md("""
## Section 3 &mdash; Refuse, usefully

&ldquo;I don't know&rdquo; is a refusal. &ldquo;The runbook covers USD limits and says nothing about FX&rdquo; is
a refusal *and* a work item for whoever owns the corpus.
"""),
    code('''
FLOOR = 0.25

def respond(question: str, claims=None, floor: float = FLOOR) -> dict:
    """Answer from the corpus, or refuse and say what was missing."""
    results = search(question, k=3, floor=floor)
    if not results:
        nearest = search(question, k=1)          # what we would have used, had we allowed it
        topic = ", ".join(sorted(terms(question))[:4])
        # TODO: refuse. Say what was searched for and what the corpus does have nearby,
        # so the refusal is a work item rather than a shrug.
        return {"answered": False, "citations": [], "why": BLANK}
    out = compose(claims or [], results)
    if not out["claims"]:
        return {"answered": False, "citations": [],
                "why": f"retrieved {len(results)} section(s) but nothing supported a claim"}
    return {"answered": True, "citations": out["citations"],
            "claims": [b["claim"] for b in out["claims"]], "dropped": out["dropped"]}
''', '''
FLOOR = 0.25

def respond(question: str, claims=None, floor: float = FLOOR) -> dict:
    """Answer from the corpus, or refuse and say what was missing."""
    results = search(question, k=3, floor=floor)
    if not results:
        nearest = search(question, k=1)          # what we would have used, had we allowed it
        topic = ", ".join(sorted(terms(question))[:4])
        near = nearest[0]["section"] if nearest else "nothing"
        return {"answered": False, "citations": [],
                "why": f"nothing in the corpus clears the bar for [{topic}]; "
                       f"the closest section is {near}"}
    out = compose(claims or [], results)
    if not out["claims"]:
        return {"answered": False, "citations": [],
                "why": f"retrieved {len(results)} section(s) but nothing supported a claim"}
    return {"answered": True, "citations": out["citations"],
            "claims": [b["claim"] for b in out["claims"]], "dropped": out["dropped"]}
'''),
    code('''
# --- Self-check: Section 3
FX_Q = "what is the FX hedging policy for JPY exposure"

check("an answerable question is answered",
      lambda: respond("limit breach approval above 500,000", claims=[SUPPORTED])["answered"]
              is True)
check("and it comes back with a citation",
      lambda: len(respond("limit breach approval above 500,000",
                          claims=[SUPPORTED])["citations"]) == 1)
check("a question the corpus cannot answer is REFUSED",
      lambda: respond(FX_Q, claims=[SUPPORTED])["answered"] is False)
check("the refusal names what was searched for",
      lambda: "hedging" in respond(FX_Q)["why"])
check("and points at the nearest thing the corpus does have",
      lambda: "closest section" in respond(FX_Q)["why"],
      "that sentence is a work item for whoever owns the corpus")
check("retrieving something but supporting nothing also refuses",
      lambda: respond("limit breach approval above 500,000",
                      claims=[INVENTED])["answered"] is False,
      "the second gate: results cleared the floor, and still no claim could name a span")
check("neither refusal asked the model to be careful",
      lambda: respond(FX_Q)["answered"] is False
              and respond("limit breach approval", claims=[INVENTED])["answered"] is False)

def _responses():
    for q, cl in (("limit breach approval above 500,000", [SUPPORTED]),
                  ("limit breach approval above 500,000", [INVENTED]),
                  (FX_Q, [SUPPORTED])):
        r = respond(q, claims=cl)
        print(f"  {'ANSWERED' if r['answered'] else 'REFUSED '}  {q[:40]:42} "
              f"{r.get('citations') or r['why'][:60]}")
guard(_responses)
'''),

    md("""
## Run it for real

Ask the model to answer the FX question from the retrieved context, with and without the floor
applied. This is the experiment that decides whether your grounding is a mechanism or a hope.
"""),
    code('''
if llm_ready():
    def _grounding():
        for label, floor in (("no floor (all 4 chunks)", 0.0), ("floor 0.25 (nothing)", FLOOR)):
            results = search(FX_Q, k=4, floor=floor)
            context = "\\n".join(f"- [{r['section']}] {r['text'][:160]}" for r in results) \\
                      or "(no documents were retrieved)"
            reply = ask(f"Context:\\n{context}\\n\\nQuestion: {FX_Q}\\n\\n"
                        "Answer using ONLY the context. If it does not contain the answer, say so.",
                        system="Be brief.")
            print(f"  [{label}]")
            print(f"      {reply.strip()[:240]}")
            print()
    guard(_grounding)
'''),
    md("""
### Read it

With the floor applied there is no context, so there is nothing to be wrong from &mdash; the refusal
is structural. Without it, four chunks about payment limits are sitting in front of a question
about FX, and the instruction &ldquo;use ONLY the context&rdquo; is the only thing standing between you and
an answer stitched out of the nearest available prose.

Sometimes the model handles it perfectly. That is worth noticing and not worth relying on: you
cannot put &ldquo;the model was sensible&rdquo; in a control document, and it is not the same sentence in
the next model version.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Extractive grounding is the strictest kind and the least fluent. Let the model paraphrase, then
   decide how you would still bind a claim to a span &mdash; and what you lose when the match stops
   being exact.
2. `respond` drops unsupported claims silently. Log them instead, and after a day of traffic read
   the log: the claims a model keeps trying to make and cannot support are a map of what your
   corpus is missing.
3. `FLOOR` is 0.25 because it worked here. Find the value where the FX question refuses and the
   four real questions still answer &mdash; then argue for it with a number rather than a feeling.
   That argument is Lab 6.5.
"""),
]


# =========================================================================== #
# Lab 6.5 -- challenge: score retrieval and generation apart
# =========================================================================== #
LAB5 = [
    header(5, "Challenge: Score the Two Halves Apart", "Advanced &middot; challenge", 40,
           ["Label an eval set: which section <em>should</em> have come back, for each question",
            "Measure retrieval with recall@k and precision@k",
            "Measure generation with faithfulness and answer relevance &mdash; no labels needed",
            "Put a run in the 2&times;2 and read off which half to fix"],
           "> **The bridge into Day 3.** One end-to-end score cannot tell a lucky answer from a\n"
           "> grounded one. Two scores can, and the fixes are unrelated."),
    setup(5),
    code(CORPUS),
    code(RETRIEVER),

    md("""
## Concept

A RAG system is two systems. They fail differently, they are fixed differently, and a single
end-to-end score cannot tell you which one broke.

The worst box in the 2&times;2 is **right answer, wrong evidence** &mdash; the model knew it already, or
guessed well. It passes every demo, and it stops working the day the model changes.
"""),

    md("""
## Section 1 &mdash; Label the eval set

Anything measured `@k` needs a ground truth: for this question, which section *should* come back?
Producing those labels is the actual work, and there is no shortcut.
"""),
    code('''
# (question, the section that answers it, one true claim from that section)
LABELLED = [
    ("What approval is needed above USD 500,000?", "3.2 Limit breaches",
     "Payments above USD 500,000 require Treasury approval before release"),
    ("What happens to a payment returned INVALID_IBAN?", "3.3 Invalid beneficiary details",
     "A payment returned INVALID_IBAN is returned to the originator with code R04"),
    ("Who decides on a payment held for sanctions review?", "3.4 Sanctions review",
     "A payment held for SANCTIONS_REVIEW is decided by Compliance"),
    ("How much may a duty manager approve?", "1 Approval authority",
     "A duty manager may approve a release up to USD 250,000"),
    ("What is retried after a funding failure?", "3.1 Insufficient funds",
     "A payment returned INSUFFICIENT_FUNDS is retried once after 24 hours"),
]

def retrieved_sections(question: str, k: int = 3) -> list:
    return [r["section"] for r in search(question, k=k)]


def recall_at_k(k: int = 3) -> float:
    """Fraction of questions whose answering section came back at all."""
    hits = 0
    for question, want, _ in LABELLED:
        # TODO: did the section that actually answers this question appear in the top k?
        if BLANK:
            hits += 1
    return hits / len(LABELLED)


def precision_at_k(k: int = 3) -> float:
    """Of everything returned across the eval set, what fraction was the right section?"""
    returned = sum(len(retrieved_sections(q, k)) for q, _, _ in LABELLED)
    correct = sum(1 for q, want, _ in LABELLED if want in retrieved_sections(q, k))
    return correct / returned if returned else 0.0
''', '''
# (question, the section that answers it, one true claim from that section)
LABELLED = [
    ("What approval is needed above USD 500,000?", "3.2 Limit breaches",
     "Payments above USD 500,000 require Treasury approval before release"),
    ("What happens to a payment returned INVALID_IBAN?", "3.3 Invalid beneficiary details",
     "A payment returned INVALID_IBAN is returned to the originator with code R04"),
    ("Who decides on a payment held for sanctions review?", "3.4 Sanctions review",
     "A payment held for SANCTIONS_REVIEW is decided by Compliance"),
    ("How much may a duty manager approve?", "1 Approval authority",
     "A duty manager may approve a release up to USD 250,000"),
    ("What is retried after a funding failure?", "3.1 Insufficient funds",
     "A payment returned INSUFFICIENT_FUNDS is retried once after 24 hours"),
]

def retrieved_sections(question: str, k: int = 3) -> list:
    return [r["section"] for r in search(question, k=k)]


def recall_at_k(k: int = 3) -> float:
    """Fraction of questions whose answering section came back at all."""
    hits = 0
    for question, want, _ in LABELLED:
        if want in retrieved_sections(question, k):
            hits += 1
    return hits / len(LABELLED)


def precision_at_k(k: int = 3) -> float:
    """Of everything returned across the eval set, what fraction was the right section?"""
    returned = sum(len(retrieved_sections(q, k)) for q, _, _ in LABELLED)
    correct = sum(1 for q, want, _ in LABELLED if want in retrieved_sections(q, k))
    return correct / returned if returned else 0.0
'''),
    code('''
# --- Self-check: Section 1
check("every labelled section really exists in the index",
      lambda: all(want in [c["section"] for c in INDEX] for _, want, _ in LABELLED))
check("every labelled claim really appears in its section",
      lambda: all(" ".join(claim.split()).lower() in
                  " ".join(next(c["text"] for c in INDEX if c["section"] == want).split()).lower()
                  for _, want, claim in LABELLED),
      "a mislabelled ground truth measures your labelling, not your retriever")
check("recall at k=3 is high on this corpus",
      lambda: recall_at_k(3) >= 0.8)
check("recall never decreases as k grows",
      lambda: recall_at_k(5) >= recall_at_k(1))
check("precision does the opposite -- more results, more noise",
      lambda: precision_at_k(1) > precision_at_k(5),
      "raising k always helps recall and always hurts precision; that trade is the whole tuning job")
check("at k=1 precision and recall are the same number",
      lambda: abs(precision_at_k(1) - recall_at_k(1)) < 1e-9)
'''),

    md("""
## Section 2 &mdash; The half you can measure without labels

Faithfulness and answer relevance need only what you already have in a trace: the question, the
retrieved text, and the answer. Start here on Monday.
"""),
    code('''
def normalise(t):
    return " ".join((t or "").split()).lower()


def faithfulness(claims: list, results: list) -> float:
    """Fraction of the answer's claims that are actually supported by the retrieved text."""
    if not claims:
        return 0.0
    context = normalise(" ".join(r["text"] for r in results))
    # TODO: a claim is faithful when the retrieved text contains it.
    supported = sum(1 for c in claims if BLANK)
    return supported / len(claims)


def answer_relevance(question: str, claims: list) -> float:
    """How much of what the question asked about the answer actually addresses."""
    asked = terms(question)
    if not asked:
        return 0.0
    answered = terms(" ".join(claims))
    return len(asked & answered) / len(asked)
''', '''
def normalise(t):
    return " ".join((t or "").split()).lower()


def faithfulness(claims: list, results: list) -> float:
    """Fraction of the answer's claims that are actually supported by the retrieved text."""
    if not claims:
        return 0.0
    context = normalise(" ".join(r["text"] for r in results))
    supported = sum(1 for c in claims if normalise(c) in context)
    return supported / len(claims)


def answer_relevance(question: str, claims: list) -> float:
    """How much of what the question asked about the answer actually addresses."""
    asked = terms(question)
    if not asked:
        return 0.0
    answered = terms(" ".join(claims))
    return len(asked & answered) / len(asked)
'''),
    code('''
# --- Self-check: Section 2
Q0, WANT0, TRUE0 = LABELLED[0]
HITS0 = search(Q0, k=3)
LIE0 = "Payments above USD 500,000 may be released by the duty manager"

check("a fully supported answer is perfectly faithful",
      lambda: faithfulness([TRUE0], HITS0) == 1.0)
check("an invented claim is not",
      lambda: faithfulness([LIE0], HITS0) == 0.0)
check("a half-invented answer scores half",
      lambda: abs(faithfulness([TRUE0, LIE0], HITS0) - 0.5) < 1e-9,
      "faithfulness is per claim, which is what makes it actionable")
check("an empty answer is not faithful by default",
      lambda: faithfulness([], HITS0) == 0.0,
      "saying nothing is not the same as saying only supported things")
check("an on-topic answer is relevant",
      lambda: answer_relevance(Q0, [TRUE0]) > 0.5)
check("a perfectly grounded answer to a DIFFERENT question is faithful and irrelevant",
      lambda: faithfulness([LABELLED[2][2]], search(LABELLED[2][0], k=3)) == 1.0
              and answer_relevance(Q0, [LABELLED[2][2]]) < 0.4,
      "the two metrics are independent, which is exactly why you need both")
check("neither metric needed a label",
      lambda: faithfulness([TRUE0], HITS0) == 1.0)
'''),

    md("""
## Section 3 &mdash; Read the 2&times;2

Now put a run in a box and read off which half to fix.
"""),
    code('''
def diagnose(question: str, want_section: str, claims: list, k: int = 3) -> dict:
    """Which of the four boxes is this run in, and what should be fixed?"""
    results = search(question, k=k)
    got_evidence = want_section in [r["section"] for r in results]
    faithful = faithfulness(claims, results) == 1.0
    # TODO: name the box and the half to fix.
    #   evidence yes + faithful yes -> "grounded",     fix "nothing"
    #   evidence yes + faithful no  -> "ignored the evidence", fix "generation"
    #   evidence no  + faithful yes -> "faithful to the wrong text", fix "retrieval"
    #   evidence no  + faithful no  -> "unsupported",   fix "both"
    box, fix = BLANK
    return {"box": box, "fix": fix, "evidence": got_evidence, "faithful": faithful}
''', '''
def diagnose(question: str, want_section: str, claims: list, k: int = 3) -> dict:
    """Which of the four boxes is this run in, and what should be fixed?"""
    results = search(question, k=k)
    got_evidence = want_section in [r["section"] for r in results]
    faithful = faithfulness(claims, results) == 1.0
    box, fix = {
        (True,  True):  ("grounded", "nothing"),
        (True,  False): ("ignored the evidence", "generation"),
        (False, True):  ("faithful to the wrong text", "retrieval"),
        (False, False): ("unsupported", "both"),
    }[(got_evidence, faithful)]
    return {"box": box, "fix": fix, "evidence": got_evidence, "faithful": faithful}
'''),
    code('''
# --- Self-check: Section 3
def missed_section(question, k=3):
    """A real section that this question does NOT retrieve -- computed, not assumed.

    Hard-coding one here is how you write a check that passes for the wrong reason: the
    section you picked as 'wrong' may well be in the top k.
    """
    got = retrieved_sections(question, k)
    return next(c["section"] for c in INDEX if c["section"] not in got)

def quoted_from_top(question, k=3):
    """A claim lifted verbatim from whatever DID come back, so it is faithful by construction."""
    return next(r["text"][:60] for r in search(question, k=k))

check("right evidence, supported claim -> grounded, nothing to fix",
      lambda: diagnose(Q0, WANT0, [TRUE0])["fix"] == "nothing")
check("right evidence, invented claim -> fix generation",
      lambda: diagnose(Q0, WANT0, [LIE0])["fix"] == "generation",
      "the evidence was sitting right there and the answer went past it")
check("the counterfactual section really is one that did not come back",
      lambda: missed_section(Q0) not in retrieved_sections(Q0, 3))
check("wrong evidence, and a claim supported by whatever DID come back -> fix retrieval",
      lambda: diagnose(Q0, missed_section(Q0), [quoted_from_top(Q0)])["fix"] == "retrieval",
      "faithful to the text in front of it, and the text in front of it was the wrong text")
check("wrong evidence and an invented claim -> fix both",
      lambda: diagnose(Q0, "no such section", [LIE0])["fix"] == "both")
check("the diagnosis reports the two inputs, not just the verdict",
      lambda: set(diagnose(Q0, WANT0, [TRUE0])) == {"box", "fix", "evidence", "faithful"})
check("an end-to-end score cannot separate these",
      lambda: diagnose(Q0, WANT0, [TRUE0])["box"]
              != diagnose(Q0, missed_section(Q0), [quoted_from_top(Q0)])["box"],
      "both runs return a fluent, supported-looking answer; only two scores tell them apart")

def _scorecard():
    print(f"  recall@1 {recall_at_k(1):.0%}   recall@3 {recall_at_k(3):.0%}   "
          f"recall@5 {recall_at_k(5):.0%}")
    print(f"  prec@1   {precision_at_k(1):.0%}   prec@3   {precision_at_k(3):.0%}   "
          f"prec@5   {precision_at_k(5):.0%}")
    print()
    for q, want, claim in LABELLED:
        d = diagnose(q, want, [claim])
        print(f"  {d['box']:28} fix: {d['fix']:12} {q[:40]}")
guard(_scorecard)
'''),

    md("""
## Run it for real

Everything above scored a *supplied* answer. Now let the model write one and score that &mdash; which
is the version you would run against production traces.
"""),
    code('''
if llm_ready():
    def _score_the_model():
        print(f"  {'box':30}{'fit':>6}{'rel':>6}  question")
        print("  " + "-" * 74)
        for q, want, _ in LABELLED:
            results = search(q, k=3)
            context = "\\n".join(f"- {r['text'][:170]}" for r in results)
            reply = ask(f"Context:\\n{context}\\n\\nQuestion: {q}\\n\\n"
                        "Answer in one sentence, quoting the context as closely as you can.",
                        system="Be brief and stay inside the context.")
            claims = [reply.strip()]
            d = diagnose(q, want, claims)
            print(f"  {d['box']:30}{faithfulness(claims, results):>6.0%}"
                  f"{answer_relevance(q, claims):>6.0%}  {q[:34]}")
    guard(_score_the_model)
'''),
    md("""
### Read it

Expect faithfulness to look poor, and read why before you believe it. This `faithfulness` is an
exact-substring check, so a model that rephrases &mdash; which is what a model does &mdash; scores zero on
a claim that is perfectly well supported. **The metric is measuring quotation, not support.**

That is the honest limitation of a stdlib scorer, and it is the right thing to hit here rather
than in production. Two ways out, and Day 3 uses both:

- ask a model to judge support, and accept that your evaluator is now a model too
- score overlap rather than containment, and pick the threshold with a labelled set

What survives either way is the shape: two numbers, not one, and a 2&times;2 that names which half
to fix.

**What you take from Module 6:** retrieval is a decision, not a step; chunking and metadata decide
more than the embedding does; top-k always returns k, so a floor is what makes refusal possible;
a citation is a span, not a filename; and the two halves are measured apart, or a lucky answer
looks exactly like a good one.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Replace `faithfulness` with a token-overlap score and pick a threshold that accepts a fair
   paraphrase and rejects `LIE0`. How confident are you in that threshold on five examples?
2. Add a sixth labelled question whose answer spans **two** sections. What does recall@k mean now,
   and what did you have to decide about the label?
3. Sweep `FLOOR` from Lab 6.4 across 0.1 to 0.5 and plot refusals against correct answers. The
   value you pick is a policy decision about how often you would rather say nothing than be wrong.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-6-01-a-retriever-you-can-inspect",   LAB1),
    ("lab-6-02-retrieval-as-a-tool",           LAB2),
    ("lab-6-03-adequacy-and-multi-hop",        LAB3),
    ("lab-6-04-citations-and-refusing",        LAB4),
    ("lab-6-05-challenge-score-the-halves",    LAB5),
]


def main():
    os.makedirs(SOLDIR, exist_ok=True)
    for name, cells in LABS:
        for solution, folder in ((False, LABDIR), (True, SOLDIR)):
            path = os.path.join(folder, name + ".ipynb")
            with open(path, "w") as fh:
                json.dump(build_notebook(cells, solution), fh, indent=1)
                fh.write("\n")
            print(("solution " if solution else "lab      ") + os.path.relpath(path, LABDIR))
    print(f"\n{len(LABS)} labs, {len(LABS) * 2} notebooks written")


if __name__ == "__main__":
    main()
