#!/usr/bin/env python3
"""
Generate Module 6 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-6-0N-*.ipynb and ../solutions/

Design rules (rebuilt 2026-09-09 to the framework-forward Day 1 rules):
  * The participant writes REAL LangChain code in every lab -- a text splitter, an
    Embeddings implementation, a Chroma collection, a @tool, a compiled StateGraph, a
    Pydantic output parser. The framework is the learning, not an optional appendix.
  * Self-checks assert on framework OBJECTS -- a Document, a collection, a bound tool,
    a compiled graph, a parser -- which is deterministic and needs no endpoint. Only
    model INVOCATION needs the gateway, and that lives in "Run it for real" cells.
  * Blanks ask a DESIGN DECISION -- which id, which floor, which parser, which verdict.
    Where the answer is a Python idiom (a comprehension, a set operation, an f-string)
    the code is given and the question moves to what only understanding answers.
  * Embeddings run OFFLINE. There is no egress in the sandbox, and chromadb's default
    embedding function downloads ~80 MB of ONNX model on first use, so it can never be
    reached from a graded cell. LabEmbeddings is a hand-written hashed bag-of-words in
    1024 dimensions: deterministic, readable, and cosine behaves the way you expect.
  * "BLANK" marks a blank; an unfilled blank raises NameError and prints [TODO].
    NOT three underscores: IPython PREDEFINES _, __ and ___ as its output history
    (they start as ""), so under a real Jupyter kernel that token is a defined empty
    string, not an undefined name. The NameError never fires, [TODO] silently becomes
    [FAIL], and a blank used as a loop guard never stops its loop.
  * Blanks live INSIDE function bodies, and every framework object built at module level
    is built lazily, so an untouched lab survives Run All.
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

> **How this lab works.** You write real LangChain code. Fill every `BLANK`, then run the
> **Self-check** cell under each section &mdash; those check the *objects you built* (a chunked
> `Document`, a Chroma collection, a bound tool, a compiled graph, a parser), so they are
> deterministic and do not depend on the model. Cells marked **Run it for real** put your code in
> front of the sandbox model; that is the part worth watching. The score line is feedback, not a
> grade.

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
    except NameError as exc:
        print(f"(a blank above is still unfilled: {{exc}} -- fill it in, then re-run this cell)")
        return default

def score() -> None:
    done = [r for r in _results if r is not None]
    passed = sum(1 for r in done if r)
    todo = sum(1 for r in _results if r is None)
    print(f"\\nScore: {{passed}}/{{len(_results)}}" + (f"   ({{todo}} still TODO)" if todo else ""))

# ---- the sandbox model ---------------------------------------------------
# Your sandbox already has an LLM configured -- nothing to install, no key to register.
# These values are read from the environment so this notebook never hardcodes an endpoint.
LLM_BASE_URL = (os.environ.get("LAB_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("LITELLM_BASE_URL"))
LLM_MODEL    = (os.environ.get("LAB_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
                or os.environ.get("LITELLM_MODEL"))
LLM_API_KEY  = os.environ.get("OPENAI_API_KEY", "sandbox")

# The served model reasons before it answers, and the reasoning is billed as completion
# tokens. Thinking is off by default here because you will make a lot of calls today;
# pass think=True to any call below to see the difference for yourself.
NO_THINK = {{"chat_template_kwargs": {{"enable_thinking": False}}}}

def llm_ready() -> bool:
    if not LLM_BASE_URL or not LLM_MODEL:
        print("Model not configured. In a sandbox terminal run `env | grep -i llm` and set:")
        print("  export LAB_LLM_BASE_URL=...    # the gateway URL from your welcome sheet")
        print("  export LAB_LLM_MODEL=...       # the model name from your welcome sheet")
        return False
    return True

_llm_cache = {{}}
def get_llm(temperature: float = 0.0, think: bool = False):
    """A LangChain chat model pointed at the sandbox gateway (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI
    key = (temperature, think)
    if key not in _llm_cache:
        kwargs = {{}} if think else {{"extra_body": NO_THINK}}
        _llm_cache[key] = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL,
                                     api_key=LLM_API_KEY, temperature=temperature, **kwargs)
    return _llm_cache[key]

def ask(prompt: str, system: str | None = None, think: bool = False) -> str:
    """One stateless call. Returns text, or an error string -- never raises."""
    try:
        msgs = ([("system", system)] if system else []) + [("human", prompt)]
        return get_llm(think=think).invoke(msgs).content
    except Exception as exc:
        return f"<model unavailable: {{type(exc).__name__}}: {{exc}}>"

print("work dir:", WORK)
print("model   :", LLM_MODEL or "(not configured -- the object-level self-checks still work)")
'''


def setup(num, extra=""):
    return code(SETUP_COMMON.format(num=num) + extra)


# --------------------------------------------------------------------------- #
# the shared synthetic domain -- the same payments as every other module
# --------------------------------------------------------------------------- #
DOMAIN = '''
# ------------------------------------------------- the case file (synthetic, self-contained)
# The same payment exceptions the other modules work on. Nothing here is real data.

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

print(f"{len(LEDGER)} payments loaded")
'''


# the corpus these labs retrieve from -- shared by all five
CORPUS = '''
# ------------------------------------------------- the corpus (synthetic, self-contained)
# Two short operating documents about the same payments. Read 3.2: the rule and the exception
# that qualifies it are adjacent sentences, which is the whole of Lab 6.1's first lesson. Note
# also what is NOT here -- nothing mentions FX or hedging anywhere, and Lab 6.4 needs that gap.

DOCS = {
    "ops-runbook-v4.md": """## 3.1 Insufficient funds
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
    "escalation-policy-v2.md": """## 1 Approval authority
A duty manager may approve a release up to USD 250,000. Above that figure Treasury approval is
required, and must be recorded against the payment reference.

## 2 Escalation timers
If an approver has not responded within 15 minutes, escalate to the Treasury lead, and after a
further 15 minutes to the head of operations.
""",
}

print(f"{len(DOCS)} documents, {sum(len(d) for d in DOCS.values())} characters")
'''


# the embedding function -- given whole in every lab, including 6.1
EMBEDDINGS = '''
# ------------------------------------------------- the embedding model (nothing to fill in)
# The sandbox has no egress, and chromadb's DEFAULT embedding function downloads about 80 MB
# of ONNX model the first time it is called. So this module brings its own: one hashed bucket
# per meaningful word, normalised to unit length. It is arithmetic rather than learning, which
# is the point -- it runs offline, it is deterministic, and you can read every line of it.
#
# What it CAN do: score two texts by the words they share. What it CANNOT do: match meaning
# with no words in common. Lab 6.2 is about living with exactly that.
import re, math, hashlib
from langchain_core.embeddings import Embeddings

STOP = set("""a an the of for is are was were do does did what which who this that these those it
its to in on at by with from about and or not no be been have has had can could should would will
you your we our i me my how why when where there here as if then than so such only just also very
more most some any other""".split())

def content_words(text: str) -> list:
    """The words worth indexing: lower-cased, no punctuation, no stop words."""
    return [w for w in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if w not in STOP and len(w) > 1]


class LabEmbeddings(Embeddings):
    """A tiny embedding model you can read. Same interface as any other LangChain embedding."""

    dim = 1024                      # enough buckets that two different words rarely collide

    def _vector(self, text: str) -> list:
        vec = [0.0] * self.dim
        for word in content_words(text):
            bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.dim
            vec[bucket] += 1.0
        length = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / length for x in vec]      # unit length, so cosine is just a dot product

    def embed_documents(self, texts: list) -> list:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list:
        return self._vector(text)


print("embeddings:", LabEmbeddings.dim, "dimensions, offline, deterministic")
'''


# the whole retriever, carried forward into labs 6.2 - 6.5
RETRIEVAL_STACK = '''
# ------------------------------------------------- carried forward from Lab 6.1 (nothing to fill in)
# Exactly what you built in Lab 6.1: split on headings, index in Chroma, search with a floor.
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma

FLOOR = 0.20            # the similarity a chunk must clear to be used at all (Lab 6.1)

def section_chunks() -> list:
    """One Document per '##' section, with the heading kept in the text and in the metadata."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")],
                                          strip_headers=False)
    out = []
    for name, text in DOCS.items():
        for chunk in splitter.split_text(text):
            chunk.metadata["source"] = name
            out.append(chunk)
    return out


_store = None
def store():
    """The Chroma collection, built once, on first use."""
    global _store
    if _store is None:
        chunks = section_chunks()
        _store = Chroma(collection_name="module6-corpus",
                        embedding_function=LabEmbeddings(),
                        persist_directory=os.path.join(WORK, "chroma"),
                        collection_configuration={"hnsw": {"space": "cosine"}})
        # ids derived from the chunk, so re-running this notebook updates instead of duplicating
        _store.add_documents(chunks, ids=[f"{c.metadata['source']}#{c.metadata['section']}"
                                          for c in chunks])
    return _store


def search(query: str, k: int = 4, floor: float = 0.0, where: dict | None = None) -> list:
    """Top-k from the store as plain dicts, with anything below `floor` dropped."""
    hits = store().similarity_search_with_score(query, k=k, filter=where)
    out = []
    for doc, distance in hits:
        similarity = 1.0 - distance         # cosine space: 1.0 identical, 0.0 nothing in common
        if similarity >= floor:
            out.append({"score": round(similarity, 3), "text": doc.page_content,
                        "source": doc.metadata["source"], "section": doc.metadata["section"]})
    return out


print(f"index ready: {len(store().get()['ids'])} chunks")
'''


# =========================================================================== #
# Lab 6.1 -- a retriever you can inspect
# =========================================================================== #
LAB1 = [
    header(1, "A Retriever You Can Inspect", "Intermediate", 35,
           ["Split a document two ways with real splitters, and watch one of them make an "
            "answer unreachable",
            "Write an <code>Embeddings</code> class and index the corpus in <strong>Chroma</strong>",
            "See top-k return k whatever is in the corpus &mdash; then choose the floor that "
            "makes an empty result possible",
            "Scope with a metadata filter, which is what production retrieval actually looks like"],
           "> **Everything here runs offline.** The embedding model is thirty lines you can read,\n"
           "> not an 80&nbsp;MB download &mdash; the sandbox has no egress. Chunking, ranking, floors\n"
           "> and filters are the same whatever computes the similarity, and they decide more than\n"
           "> the model does."),
    setup(1),
    code(CORPUS),

    md("""
## Concept

A retriever is four decisions, and only one of them is the embedding model:

| Decision | The LangChain piece |
|---|---|
| how the corpus is cut up | a **text splitter** &mdash; decides what can ever be returned together |
| how a chunk is scored | an **`Embeddings`** implementation |
| how many come back, and how bad they may be | `k`, and a **floor** you impose yourself |
| what is in scope before ranking starts | a **metadata filter** |

You build all four in this lab, on a real `Chroma` collection. Three of them are yours to
decide; only the second is bought off a shelf, and it is the one people think is the whole thing.
"""),

    md("""
## Section 1 &mdash; Chunking decides what can be found

Section 3.2 states a rule and then exempts intra-group transfers from it. Cut that in half and no
retriever can ever return the two together, because they are no longer one thing.

`MarkdownHeaderTextSplitter` splits on headings and writes the heading into each chunk's
metadata. `RecursiveCharacterTextSplitter` splits on size, and does not care what it cuts.
"""),
    code('''
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def split_by_section(doc_name: str, text: str) -> list:
    """One Document per '##' section, so a rule and its exception stay together."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section")],
        # TODO: the heading "3.2 Limit breaches" carries words a query will use. Should it stay
        # in the text that gets embedded, or be stripped out and left only in the metadata?
        strip_headers=BLANK)
    chunks = splitter.split_text(text)
    for chunk in chunks:
        chunk.metadata["source"] = doc_name      # the splitter fills in "section"; we add the file
    return chunks


def split_by_size(doc_name: str, text: str, size: int = 120) -> list:
    """The naive alternative: cut every `size` characters, meaning be damned."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=0)
    return splitter.create_documents([text], metadatas=[{"source": doc_name}])


def build_chunks(splitter_fn) -> list:
    """Run one splitter over every document in the corpus."""
    return [c for name, text in DOCS.items() for c in splitter_fn(name, text)]
''', '''
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def split_by_section(doc_name: str, text: str) -> list:
    """One Document per '##' section, so a rule and its exception stay together."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section")],
        strip_headers=False)          # a query says "limit breach" -- keep those words in the text
    chunks = splitter.split_text(text)
    for chunk in chunks:
        chunk.metadata["source"] = doc_name      # the splitter fills in "section"; we add the file
    return chunks


def split_by_size(doc_name: str, text: str, size: int = 120) -> list:
    """The naive alternative: cut every `size` characters, meaning be damned."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=0)
    return splitter.create_documents([text], metadatas=[{"source": doc_name}])


def build_chunks(splitter_fn) -> list:
    """Run one splitter over every document in the corpus."""
    return [c for name, text in DOCS.items() for c in splitter_fn(name, text)]
'''),
    code('''
# --- Self-check: Section 1   (Document objects only -- no store yet, no model)
def by_section():
    return build_chunks(split_by_section)

def by_size():
    return build_chunks(split_by_size)

def limit_chunk(chunks):
    """The chunk that states the USD 500,000 rule."""
    return next(c for c in chunks if "500,000" in c.page_content)

check("the section splitter finds all six sections across the two documents",
      lambda: len(by_section()) == 6)
check("each chunk is a Document that knows its file and its section",
      lambda: all(isinstance(c, Document) and c.metadata["source"] in DOCS
                  and c.metadata["section"] for c in by_section()))
check("the heading is part of the text that will be embedded, not just a label",
      lambda: "Limit breaches" in limit_chunk(by_section()).page_content,
      "a query says 'limit breach'; if that phrase is only in the metadata it cannot be matched")
check("the rule and the exception that qualifies it are in ONE chunk",
      lambda: "intra-group" in limit_chunk(by_section()).page_content)
check("cutting by size splits them apart",
      lambda: "intra-group" not in limit_chunk(by_size()).page_content,
      "after this cut, no retriever on earth can return them together")
check("and that is a boundary problem, not a small-chunk problem",
      lambda: any("intra-group" in c.page_content for c in by_size()),
      "the exception is still indexed; it is just no longer attached to the rule it qualifies")

def _compare():
    print("  by section:", limit_chunk(by_section()).page_content[:96].replace("\\n", " "), "...")
    print("  by size   :", limit_chunk(by_size()).page_content[:96].replace("\\n", " "), "...")
guard(_compare)
'''),

    code(EMBEDDINGS),

    md("""
## Section 2 &mdash; Index it in Chroma

`Chroma` is a vector store: you hand it `Document`s and an `Embeddings`, and it keeps the vectors
so you can search them. Two arguments below are worth understanding rather than copying:

- **`collection_configuration={"hnsw": {"space": "cosine"}}`** &mdash; the distance metric. In cosine
  space a distance of `0.0` means identical and `1.0` means nothing in common, so
  `similarity = 1 - distance` reads the way you expect.
- **`ids=`** &mdash; `add_documents` *upserts* on the id. Stable ids mean re-running this notebook
  leaves six chunks; ids that change mean six more every time, and every score after that is
  measured on a duplicated corpus.
"""),
    code('''
from langchain_chroma import Chroma

def chunk_id(chunk) -> str:
    """A stable id for one chunk -- the same string on every run of this notebook."""
    # TODO: build the id out of the chunk's own metadata, so re-running updates rather than
    # duplicates. A counter or a uuid would be different on the next run.
    return BLANK


def open_store(chunks):
    """A persistent Chroma collection over the corpus, embedded by LabEmbeddings."""
    store = Chroma(collection_name="module6-corpus",
                   embedding_function=LabEmbeddings(),
                   persist_directory=os.path.join(WORK, "chroma"),
                   collection_configuration={"hnsw": {"space": "cosine"}})
    store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return store


_store = None
def store():
    """The collection, built once, on first use."""
    global _store
    if _store is None:
        _store = open_store(build_chunks(split_by_section))
    return _store
''', '''
from langchain_chroma import Chroma

def chunk_id(chunk) -> str:
    """A stable id for one chunk -- the same string on every run of this notebook."""
    return f"{chunk.metadata['source']}#{chunk.metadata['section']}"


def open_store(chunks):
    """A persistent Chroma collection over the corpus, embedded by LabEmbeddings."""
    store = Chroma(collection_name="module6-corpus",
                   embedding_function=LabEmbeddings(),
                   persist_directory=os.path.join(WORK, "chroma"),
                   collection_configuration={"hnsw": {"space": "cosine"}})
    store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return store


_store = None
def store():
    """The collection, built once, on first use."""
    global _store
    if _store is None:
        _store = open_store(build_chunks(split_by_section))
    return _store
'''),
    code('''
# --- Self-check: Section 2   (a real Chroma collection -- built locally, no network)
check("the collection holds one chunk per section",
      lambda: len(store().get()["ids"]) == 6)
check("indexing the same corpus again leaves it at six, not twelve",
      lambda: len(open_store(build_chunks(split_by_section)).get()["ids"]) == 6,
      "add_documents upserts on the id -- unstable ids duplicate the corpus on every re-run")
check("the ids are unique",
      lambda: len(set(store().get()["ids"])) == 6)
check("an id is derived from the chunk, so it is the same on a fresh split",
      lambda: chunk_id(build_chunks(split_by_section)[3])
              == chunk_id(build_chunks(split_by_section)[3]),
      "a uuid or a counter fails this, and that failure is what duplicates the corpus")
check("the metadata went into the store with the text",
      lambda: all(set(m) == {"source", "section"} for m in store().get()["metadatas"]))
check("the embedding is the one you can read, not a downloaded one",
      lambda: store().embeddings.__class__ is LabEmbeddings,
      "chromadb's default embedding function needs an 80 MB download and this sandbox has no egress")
'''),

    md("""
## Section 3 &mdash; Rank, and then refuse to

`similarity_search_with_score` returns `(Document, distance)` pairs, best first. `search` below
turns those into plain dicts and drops anything under a floor.

The floor is the interesting part. **Top-k always returns k** &mdash; ask a corpus about something
it has never heard of and you still get four confident rows back. The floor is the only thing
standing between you and answering from them.
"""),
    code('''
def search(query: str, k: int = 4, floor: float = 0.0, where: dict | None = None) -> list:
    """Top-k from the store as plain dicts, with anything below `floor` dropped."""
    hits = store().similarity_search_with_score(query, k=k, filter=where)
    out = []
    for doc, distance in hits:
        similarity = 1.0 - distance         # cosine space: 1.0 identical, 0.0 nothing in common
        if similarity >= floor:
            out.append({"score": round(similarity, 3), "text": doc.page_content,
                        "source": doc.metadata["source"], "section": doc.metadata["section"]})
    return out


REAL_QUESTIONS = [
    "what approval does a limit breach above USD 500,000 need",
    "what happens to a payment returned INVALID_IBAN",
    "who decides on a payment held for sanctions review",
    "how much may a duty manager approve",
]
FX_Q = "what is the FX hedging policy for JPY exposure"
'''),
    code('''
# Look at the numbers before you pick anything. Every score, both kinds of question.
def _scores():
    for label, question in [("answerable", REAL_QUESTIONS[0]), ("not in the corpus", FX_Q)]:
        print(f"  [{label}] {question}")
        for r in search(question, k=4):
            print(f"      {r['score']:.3f}  {r['source']:26} {r['section']}")
        print()
guard(_scores)
'''),
    code('''
def chosen_floor() -> float:
    """The similarity a chunk must clear before it is used at all.

    The cell above printed every score. The answerable question's best chunk and the
    unanswerable question's best chunk are the two numbers your floor has to separate.
    """
    return BLANK      # TODO: pick it from those printed scores, not from a round number you like
''', '''
def chosen_floor() -> float:
    """The similarity a chunk must clear before it is used at all.

    The cell above printed every score. The answerable question's best chunk and the
    unanswerable question's best chunk are the two numbers your floor has to separate.
    """
    return 0.20       # every real question's best chunk clears it; nothing FX-related does
'''),
    code('''
# --- Self-check: Section 3   (ranking and the floor -- still no model)
check("results come back ranked, best first",
      lambda: [r["score"] for r in search(REAL_QUESTIONS[0])]
              == sorted((r["score"] for r in search(REAL_QUESTIONS[0])), reverse=True))
check("k is respected",
      lambda: len(search(REAL_QUESTIONS[0], k=2)) == 2)
check("A QUESTION THE CORPUS CANNOT ANSWER STILL RETURNS FOUR ROWS",
      lambda: len(search(FX_Q, k=4)) == 4,
      "nothing in either document mentions FX or JPY, and four chunks come back anyway")
check("and none of them is about FX",
      lambda: not any("hedg" in r["text"].lower() for r in search(FX_Q, k=4)))
check("your floor is a similarity, somewhere between 0 and 1",
      lambda: 0.0 < chosen_floor() < 1.0)
check("your floor turns the unanswerable question into an EMPTY result",
      lambda: search(FX_Q, floor=chosen_floor()) == [],
      "this is the only thing that lets the agent say 'I could not find it'")
check("and it still answers all four real questions",
      lambda: all(search(q, floor=chosen_floor()) for q in REAL_QUESTIONS),
      "a floor that refuses everything is not a safe floor, it is a broken one")
check("a floor of 0.95 refuses even the good question -- too high is its own failure",
      lambda: search(REAL_QUESTIONS[0], floor=0.95) == [],
      "Lab 6.5 measures where the floor should sit instead of arguing about it")
'''),

    md("""
## Section 4 &mdash; Scope before you rank

Most production retrieval is a metadata filter with a similarity search inside it: this version,
this jurisdiction, the documents this user is allowed to see. An unfiltered index is a disclosure
waiting to be reported.

Chroma takes the filter as `where`. One condition is a plain `{"key": value}`; two conditions
have to be spelled out with `$and` &mdash; `{"a": 1, "b": 2}` is an error, not an AND.
"""),
    code('''
def scope_to(doc_name: str) -> dict:
    """A filter that keeps retrieval inside ONE document."""
    # TODO: which piece of metadata did every chunk get in Section 1?
    return {BLANK: doc_name}


def scope_to_section(doc_name: str, section: str) -> dict:
    """Two conditions at once, the way Chroma wants them."""
    return {"$and": [{"source": doc_name}, {"section": section}]}
''', '''
def scope_to(doc_name: str) -> dict:
    """A filter that keeps retrieval inside ONE document."""
    return {"source": doc_name}


def scope_to_section(doc_name: str, section: str) -> dict:
    """Two conditions at once, the way Chroma wants them."""
    return {"$and": [{"source": doc_name}, {"section": section}]}
'''),
    code('''
# --- Self-check: Section 4   (metadata filtering -- exact, and offline)
APPROVAL_Q = "who may approve a release"

check("an unfiltered search sees both documents",
      lambda: {r["source"] for r in search(APPROVAL_Q, k=6)} == set(DOCS))
check("scoping to the escalation policy returns only its sections",
      lambda: all(r["source"] == "escalation-policy-v2.md"
                  for r in search(APPROVAL_Q, where=scope_to("escalation-policy-v2.md"))))
check("and it changes the answer, which is the whole point",
      lambda: search(APPROVAL_Q, where=scope_to("escalation-policy-v2.md"))[0]["section"]
              .startswith("1"))
check("scoping to a document that does not exist returns NOTHING, not everything",
      lambda: search(APPROVAL_Q, where=scope_to("no-such-doc.md")) == [],
      "a filter that silently falls back to the whole index is how a disclosure happens")
check("both conditions of an $and have to match",
      lambda: search(APPROVAL_Q,
                     where=scope_to_section("ops-runbook-v4.md", "no such section")) == [])
check("and a real pair matches exactly one chunk",
      lambda: len(search("limit breach", k=6,
                         where=scope_to_section("ops-runbook-v4.md", "3.2 Limit breaches"))) == 1)
'''),

    md("""
## Run it for real &mdash; the chunking decides the answer

One question, one model, two chunkings. The context is the chunk that states the USD 500,000
rule &mdash; taken once from the section split and once from the size split.
"""),
    code('''
if llm_ready():
    def _chunking_changes_the_answer():
        question = "Can we release a large intra-group transfer without Treasury approval?"
        for label, chunks in (("cut on section headings", build_chunks(split_by_section)),
                              ("cut every 120 characters", build_chunks(split_by_size))):
            context = limit_chunk(chunks).page_content
            reply = ask(f"Context:\\n{context}\\n\\nQuestion: {question}\\n\\n"
                        "Answer from the context alone, in one sentence.",
                        system="Be brief. If the context does not settle it, say so.")
            print(f"  [{label}]")
            print(f"      context: {context[:88]}...".replace("\\n", " "))
            print(f"      answer : {reply.strip()[:200]}")
            print()
    guard(_chunking_changes_the_answer)
'''),
    md("""
### Read it

The model is the same in both halves. The retrieval floor, the metadata, the prompt and the
question are the same. The only difference is where a splitter put a boundary &mdash; and the
size-split context stops one clause short of *&ldquo;this does not apply to intra-group transfers&rdquo;*,
so a correct-sounding answer is now the wrong one. No amount of prompting fixes that: the words
are not in front of the model.

Two things to carry out of this lab.

**The floor is a policy, not a constant.** You chose a number that separates a question the corpus
answers from one it does not. The gap on this corpus is wide, because the FX question shares no
words at all with either document. On a real corpus it is narrower, which is why Lab 6.5 measures
the floor instead of arguing about it.

**The embedding is the part you did not have to build.** `LabEmbeddings` matches on shared words,
so it misses *&ldquo;can we push a large payment between our own entities&rdquo;* &mdash; a question that
means section 3.2 exactly and shares almost no words with it. A trained embedding gets that one
right. What it would *not* change is anything else you did here: top-k still returns k, a badly cut
chunk still cannot be reassembled, and an unfiltered index still returns things the reader should
not see. Those are the parts you build, and they are the rest of this module.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Give `split_by_size` a `chunk_overlap` of 40 and re-run the Section 1 checks. Does overlap
   actually reattach the exception to its rule, or does it just make the failure rarer and
   harder to find?
2. Ask `search` the semantic question &mdash; *&ldquo;can we push a large payment between our own
   entities&rdquo;* &mdash; and look at where 3.2 ranks. Now add the word &ldquo;intra-group&rdquo; to the query.
   That gap is the whole of Lab 6.2's third section.
3. Set `floor` to 0.1, 0.2 and 0.4 in turn and record, for both kinds of question, whether you got
   an answer and whether it was right. That table is the beginning of Lab 6.5.
"""),
]


# =========================================================================== #
# Lab 6.2 -- retrieval as a tool the agent chooses
# =========================================================================== #
LAB2 = [
    header(2, "Retrieval as a Tool the Agent Chooses", "Intermediate &rarr; Advanced", 40,
           ["Wrap the retriever in a <code>@tool</code> &mdash; and write the description the "
            "model actually reads",
            "Offer it alongside the ledger tool with <code>bind_tools</code>, and let the model "
            "decide whether to retrieve",
            "Price always-retrieve: the tokens, and the noise it puts next to the real context",
            "Rewrite the user's words into the corpus's vocabulary"],
           "> **Builds directly on Lab 6.1's retriever.** Same splitter, same embeddings, same\n"
           "> Chroma collection. What changes is who decides when it runs, and what it is asked."),
    setup(2),
    code(DOMAIN),
    code(CORPUS),
    code(EMBEDDINGS),
    code(RETRIEVAL_STACK),

    md("""
## Concept

A pipeline retrieves once, always, with the user's exact words. That is one decision, made at
build time, applied to every question.

An agent makes three decisions per question, and this lab builds the first two:

- **whether** to retrieve &mdash; some questions are answered by the ledger, by the conversation,
  or by arithmetic
- **what to ask for** &mdash; users write in their words, corpora in the organisation's

The third, *whether to ask again*, is Lab 6.3.

The mechanism for the first one is not a rule you write. It is a **tool description**: the model
chooses between the tools you bind, and the description is all it has to choose on.
"""),

    md("""
## Section 1 &mdash; Wrap retrieval in a tool

`@tool` turns a function into something a model can call. It takes the **name** from the function,
the **argument schema** from the type hints, and the **description from the docstring**.

That docstring is the whole interface. Measured on this sandbox: writing the descriptions
properly moved first-tool accuracy from **2/5 to 5/5** &mdash; same model, same functions.
"""),
    code('''
from langchain_core.tools import tool

def retrieve_text(query: str) -> str:
    """The retrieval itself: search the corpus and format what came back. Nothing to fill in."""
    hits = search(query, k=3, floor=FLOOR)
    if not hits:
        return "nothing in the operating documents cleared the relevance floor for that query"
    return "\\n\\n".join(f"[{h['source']} #{h['section']}] {h['text']}" for h in hits)


@tool
def search_operating_docs(query: str) -> str:
    """BLANK"""
    # TODO: replace that docstring. It is the ONLY thing the model reads when it decides
    # whether to call this tool. Say what the corpus contains (the payments operating runbook
    # and the escalation policy), what a good query looks like, and -- the part people skip --
    # what this tool is NOT for. Use the word "not" when you say it.
    return retrieve_text(query)


@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record (amount, currency, counterparty, status, reason code) for ONE
    payment reference such as 'PMT-1003'. Use when the question names a specific payment. This
    reads the ledger only -- it is not a search over documents, procedures or policy.
    """
    record = LEDGER.get(ref)
    return json.dumps({"ref": ref, **record}) if record else f"no payment found for {ref!r}"
''', '''
from langchain_core.tools import tool

def retrieve_text(query: str) -> str:
    """The retrieval itself: search the corpus and format what came back. Nothing to fill in."""
    hits = search(query, k=3, floor=FLOOR)
    if not hits:
        return "nothing in the operating documents cleared the relevance floor for that query"
    return "\\n\\n".join(f"[{h['source']} #{h['section']}] {h['text']}" for h in hits)


@tool
def search_operating_docs(query: str) -> str:
    """Search the payments operating documents -- the operations runbook and the escalation
    policy -- and return the sections that match, with their headings. Use it for questions
    about procedure: what to do about a failure reason code, who may approve a release, how
    long before something escalates. Query it with the words the documents would use, such as
    'INVALID_IBAN return to originator' rather than 'the client typed the wrong number'. It is
    not a ledger lookup and knows nothing about individual payments.
    """
    return retrieve_text(query)


@tool
def lookup_payment(ref: str) -> str:
    """Return the ledger record (amount, currency, counterparty, status, reason code) for ONE
    payment reference such as 'PMT-1003'. Use when the question names a specific payment. This
    reads the ledger only -- it is not a search over documents, procedures or policy.
    """
    record = LEDGER.get(ref)
    return json.dumps({"ref": ref, **record}) if record else f"no payment found for {ref!r}"
'''),
    code('''
# --- Self-check: Section 1   (the tool OBJECTS, and one real retrieval -- no model)
def _desc(t) -> str:
    d = (t.description or "").strip()
    if d == "BLANK" or not d:
        raise NameError(f"{t.name} still has the placeholder docstring")
    return d

check("@tool took the name from the function",
      lambda: search_operating_docs.name == "search_operating_docs")
check("the argument schema was inferred from the type hint",
      lambda: "query" in search_operating_docs.args)
check("the description is a description, not a label",
      lambda: len(_desc(search_operating_docs)) > 80,
      "'Searches documents.' tells the model nothing it could not guess from the name")
check("it names what is in the corpus",
      lambda: any(w in _desc(search_operating_docs).lower()
                  for w in ("runbook", "escalation", "operating")),
      "the model cannot guess which documents you indexed")
check("it says what the tool is NOT for",
      lambda: "not" in _desc(search_operating_docs).lower(),
      "two tools that both sound like 'looks things up' is how the wrong one gets called")
check("calling the tool really retrieves",
      lambda: "3.2" in search_operating_docs.invoke({"query": "approval above USD 500,000"}))
check("and it says so plainly when nothing clears the floor",
      lambda: "nothing" in search_operating_docs.invoke(
          {"query": "FX hedging policy for JPY"}).lower(),
      "the floor from Lab 6.1, doing its job inside a tool")
check("the ledger tool is a different tool with a different boundary",
      lambda: "PMT-1003" in lookup_payment.invoke({"ref": "PMT-1003"}))
'''),

    md("""
## Section 2 &mdash; Offer both, and let the model choose

`bind_tools` returns a model that may answer with a **tool call** instead of prose. Give it both
tools and *whether to retrieve* stops being a rule you maintain and becomes a choice it makes
per question &mdash; on the strength of the descriptions you just wrote.
"""),
    code('''
def agent_tools() -> list:
    """The tool OBJECTS this agent may choose between. bind_tools wants objects, not names."""
    return BLANK          # TODO: which tools may it call?


def bound_model():
    """A model allowed to answer with a tool call. Used by the live cells below."""
    return get_llm().bind_tools(agent_tools())
''', '''
def agent_tools() -> list:
    """The tool OBJECTS this agent may choose between. bind_tools wants objects, not names."""
    return [lookup_payment, search_operating_docs]


def bound_model():
    """A model allowed to answer with a tool call. Used by the live cells below."""
    return get_llm().bind_tools(agent_tools())
'''),
    code('''
# --- Self-check: Section 2   (the tool list -- building it needs no endpoint)
check("both tools are on offer",
      lambda: {t.name for t in agent_tools()} == {"lookup_payment", "search_operating_docs"})
check("bind_tools was given the objects, not their names",
      lambda: all(hasattr(t, "invoke") and hasattr(t, "name") for t in agent_tools()),
      "a list of strings binds nothing -- the model would be offered no tools at all")
check("every tool it may choose carries a real description",
      lambda: all(len(_desc(t)) > 80 for t in agent_tools()),
      "the choice the model makes is made entirely out of these strings")
'''),

    md("""
## Section 3 &mdash; What always-retrieve costs

Before the model gets a say, price the alternative. A pipeline retrieves for every question. Two
things go wrong, and the second does not show up on an invoice: tokens spent, and irrelevant
policy prose sitting next to the context the answer was actually in.

The keyword rule below is the version you could write by hand. It is here as a *baseline* &mdash;
it works on the eight questions someone thought of, and the live cell asks whether the model
does better without one.
"""),
    code('''
QUESTIONS = [
    # (question, does it need the corpus?)
    ("What approval does a payment above USD 500,000 need?",       True),
    ("What happens when a payment comes back INVALID_IBAN?",       True),
    ("Who decides on a payment held for sanctions review?",        True),
    ("How long before an unanswered approval escalates?",          True),
    ("What is 990,000 minus 500,000?",                             False),
    ("Calculate the difference between the amount and the limit.", False),
    ("Summarise what we just agreed.",                             False),
    ("What did I ask you a moment ago?",                           False),
]

CONVERSATION_HINTS = ("we just", "you said", "a moment ago", "earlier you", "we agreed",
                      "summarise what we", "recap")
ARITHMETIC_HINTS   = ("plus", "minus", "times", "calculate", "subtract", "difference between",
                      "how much is")

def needs_corpus(question: str) -> bool:
    """The hand-written baseline: two kinds of question do not need the corpus."""
    low = (question or "").lower()
    return not any(h in low for h in CONVERSATION_HINTS + ARITHMETIC_HINTS)


def retrieved_tokens(question: str, always: bool) -> int:
    """Roughly what retrieval put into the context for this question."""
    if not always and not needs_corpus(question):
        return 0
    return sum(len(r["text"]) // 4 for r in search(question, k=4))
'''),
    code('''
# --- Self-check: Section 3   (counting, over the real index -- no model)
check("the baseline gets all eight questions right",
      lambda: all(needs_corpus(q) is expected for q, expected in QUESTIONS))
check("half the set needs no corpus at all",
      lambda: sum(1 for _, e in QUESTIONS if not e) == 4,
      "that fraction is the whole argument -- a pipeline retrieves for all eight")
check("always-retrieve spends tokens on questions that needed nothing",
      lambda: sum(retrieved_tokens(q, always=True) for q, e in QUESTIONS if not e) > 0)
check("deciding first spends none",
      lambda: sum(retrieved_tokens(q, always=False) for q, e in QUESTIONS if not e) == 0)
check("and the corpus questions are retrieved identically either way",
      lambda: all(retrieved_tokens(q, True) == retrieved_tokens(q, False)
                  for q, e in QUESTIONS if e),
      "deciding is not retrieving less well -- it is retrieving less often")
check("the noise is the part that never shows on the invoice",
      lambda: len(search("Summarise what we just agreed.", k=4)) == 4,
      "four chunks of policy prose, competing with the conversation the answer is actually in")

def _cost():
    always = sum(retrieved_tokens(q, True) for q, _ in QUESTIONS)
    decide = sum(retrieved_tokens(q, False) for q, _ in QUESTIONS)
    print(f"  always retrieve : {always:>5} retrieved tokens")
    print(f"  decide first    : {decide:>5} retrieved tokens")
guard(_cost)
'''),

    md("""
## Section 4 &mdash; Ask in the corpus's words

Users describe their situation. Documents describe the organisation's rules. Translating between
them is the cheapest retrieval improvement there is, because it changes nothing about the index.
"""),
    code('''
# What people say -> what the documents call it
VOCAB = {
    "bounce":           "INSUFFICIENT_FUNDS retry",
    "bounced":          "INSUFFICIENT_FUNDS retry",
    "push it through":  "release Treasury approval",
    "push through":     "release Treasury approval",
    "wrong account":    "INVALID_IBAN beneficiary originator",
    "bad iban":         "INVALID_IBAN beneficiary originator",
    "on hold":          "SANCTIONS_REVIEW Compliance",
    "stuck":            "SANCTIONS_REVIEW Compliance",
    "chase":            "escalate approver Treasury lead",
    "our own entities": "intra-group transfers",
}

def rewrite(question: str) -> str:
    """The query the agent actually sends."""
    extra = [v for k, v in VOCAB.items() if k in (question or "").lower()]
    if not extra:
        return question
    # TODO: build the query out of `question` and `extra`. One of them alone is wrong:
    # the table cannot cover every phrasing, and the user's words do not match the documents.
    return BLANK
''', '''
# What people say -> what the documents call it
VOCAB = {
    "bounce":           "INSUFFICIENT_FUNDS retry",
    "bounced":          "INSUFFICIENT_FUNDS retry",
    "push it through":  "release Treasury approval",
    "push through":     "release Treasury approval",
    "wrong account":    "INVALID_IBAN beneficiary originator",
    "bad iban":         "INVALID_IBAN beneficiary originator",
    "on hold":          "SANCTIONS_REVIEW Compliance",
    "stuck":            "SANCTIONS_REVIEW Compliance",
    "chase":            "escalate approver Treasury lead",
    "our own entities": "intra-group transfers",
}

def rewrite(question: str) -> str:
    """The query the agent actually sends."""
    extra = [v for k, v in VOCAB.items() if k in (question or "").lower()]
    if not extra:
        return question
    return question + " " + " ".join(extra)
'''),
    code('''
# --- Self-check: Section 4   (retrieval quality, measured -- no model)
VAGUE = [
    ("Why did this one bounce, and do we try again?",           "3.1"),
    ("The client gave us the wrong account number. Now what?",  "3.3"),
    ("It is stuck. Who decides?",                               "3.4"),
    ("Can we push a big one through between our own entities?", "3.2"),
]

def best_section(query: str):
    hits = search(query, k=1)
    return hits[0]["section"] if hits else None

check("the rewrite keeps the user's own words",
      lambda: rewrite("Why did this one bounce?").startswith("Why did this one bounce?"),
      "the table cannot cover everything; dropping the original loses whatever it missed")
check("and adds the corpus vocabulary they imply",
      lambda: "INSUFFICIENT_FUNDS" in rewrite("Why did this one bounce?"))
check("a question with no match is passed through unchanged",
      lambda: rewrite("Who approves a release?") == "Who approves a release?")
check("RAW, these four vague questions land on the wrong section",
      lambda: sum(1 for q, want in VAGUE if (best_section(q) or "").startswith(want)) <= 1)
check("rewritten, they all land on the right one",
      lambda: all((best_section(rewrite(q)) or "").startswith(want) for q, want in VAGUE),
      "same index, same embeddings, same k -- the only change is who wrote the query")

def _rewrites():
    for q, want in VAGUE:
        print(f"  {q}")
        print(f"      raw       -> {best_section(q)}")
        print(f"      rewritten -> {best_section(rewrite(q))}   (want {want})")
guard(_rewrites)
'''),

    md("""
## Run it for real &mdash; part 1: does the description do the work?

Two bindings of the same two functions. One arm has your descriptions; the other has what a
rushed codebase actually looks like. Same model, same questions. Watch which tool it reaches for.
"""),
    code('''
if llm_ready():
    def _description_ab():
        from langchain_core.tools import StructuredTool

        def _ledger(ref: str) -> str:
            return lookup_payment.invoke({"ref": ref})

        def _docs(query: str) -> str:
            return retrieve_text(query)

        # the same two functions, behind the descriptions a rushed codebase actually ships
        vague = [
            StructuredTool.from_function(_ledger, name="tool_a", description="Gets data."),
            StructuredTool.from_function(_docs,   name="tool_b", description="Looks things up."),
        ]
        probes = [
            ("What approval does a payment above USD 500,000 need?", "docs"),
            ("What is the status of PMT-1003?",                      "ledger"),
            ("Who decides on a payment held for sanctions review?",  "docs"),
            ("How much is PMT-1005 for?",                            "ledger"),
            ("How long before an unanswered approval escalates?",    "docs"),
        ]
        for label, tools in (("vague descriptions", vague), ("your descriptions", agent_tools())):
            right = 0
            for question, want in probes:
                calls = get_llm().bind_tools(tools).invoke(question).tool_calls
                picked = calls[0]["name"] if calls else "(no tool)"
                got = "ledger" if picked in ("tool_a", "lookup_payment") else \\
                      "docs" if picked in ("tool_b", "search_operating_docs") else "?"
                right += got == want
                print(f"  [{label:18}] {question[:44]:46} -> {picked}")
            print(f"  [{label:18}] first-tool accuracy {right}/{len(probes)}\\n")
    guard(_description_ab)
'''),
    md("""
## Run it for real &mdash; part 2: let the model write the query

The lookup table is a stand-in. This is what you would actually ship, because no table survives
contact with real users.
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
            flag = "ok  " if (got or "").startswith(want) else "MISS"
            print(f"  [{flag}] {q[:42]:44} -> {got}")
    guard(_model_rewrite)
'''),
    md("""
### Read it

**The A/B.** Two identical functions behind two sets of strings. The equivalent experiment on this
sandbox in Module 1 &mdash; five tools, same model &mdash; moved first-tool accuracy from **2/5 to 5/5** on
the descriptions alone. Two tools is an easier problem than five, so expect a smaller gap here;
what you are watching for is *which* questions the vague arm gets wrong. It will be the ones where
both names sound equally plausible. If your own arm scores badly, read your description the way the
model does: does it say which questions belong to this tool, and which do not?

**The rewrites.** If the model's rewrites land as well as the lookup table's, you have something
that generalises to questions you never enumerated, at the cost of one model call before every
retrieval. That is a real trade, and Lab 6.5 is where you price it. Watch for the failure mode
too: a rewrite that invents a term the corpus does not contain retrieves *worse* than the raw
question. A query, like a tool description, can attract the wrong thing as easily as the right one.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. `needs_corpus` is a keyword list, so it fails on any phrasing you did not think of. Write three
   questions that should not retrieve and that it gets wrong. What does that tell you about
   shipping the rule rather than the tool description?
2. Add a third tool that overlaps with `search_operating_docs` &mdash; say `search_escalation_policy`,
   scoped with the metadata filter from Lab 6.1. Now write both descriptions so the model can tell
   them apart, and re-run the A/B.
3. There is a third answer besides yes and no: *retrieve, but only if the first attempt at
   answering is thin*. Sketch it, and say what it costs in latency. That is Lab 6.3.
"""),
]


# =========================================================================== #
# Lab 6.3 -- adequacy, re-querying and the hop budget
# =========================================================================== #
LAB3 = [
    header(3, "Adequacy, Re-querying and Multi-Hop", "Advanced", 40,
           ["Judge your own retrieval &mdash; does it actually contain what was asked for?",
            "Build the retrieve &rarr; judge &rarr; re-query loop as a compiled "
            "<code>StateGraph</code>",
            "Re-query with a term the first hop taught you",
            "Stop: a hop budget, a repeat detector, and &lsquo;I could not find it&rsquo; as a "
            "real outcome"],
           "> **This is what makes it agentic.** A pipeline retrieves once. Everything in this lab\n"
           "> is the loop a pipeline cannot have, and the stops that keep it from running away."),
    setup(3),
    code(CORPUS),
    code(EMBEDDINGS),
    code(RETRIEVAL_STACK),

    md("""
## Concept

The first retrieval usually returns something. The question is whether it returns *enough*, and
on this corpus that is answerable without a model: **did what came back contain the things the
question asked about?**

When it did not, the results still tell you something &mdash; they hand you the corpus's own
vocabulary, which is exactly what the second query needed. That loop is a graph:

```
retrieve -> judge -> (adequate?) -> report
                  \\-> re-query -> retrieve -> ...
```

You built graphs in Module 3. Same `StateGraph`, same conditional edge, same reason for a budget:
a cycle without one is a bill.
"""),

    md("""
## Section 1 &mdash; Was that enough?

An adequacy test that is honest has to be able to say no. `coverage` measures how much of what
the question needed actually appeared; you decide how much is enough.
"""),
    code('''
def coverage(need_terms: list, results: list) -> float:
    """How much of what the question needed actually appeared in what came back, 0.0 to 1.0."""
    need = {t.lower() for t in need_terms}
    if not need:
        return 0.0
    covered = set(content_words(" ".join(r["text"] for r in results)))
    return len(need & covered) / len(need)


def missing(need_terms: list, results: list) -> list:
    """What the question asked about that the results never mention."""
    covered = set(content_words(" ".join(r["text"] for r in results)))
    return sorted(t for t in need_terms if t.lower() not in covered)


def adequate(need_terms: list, results: list) -> bool:
    """Is this retrieval enough to answer from?"""
    if not results:
        return False
    # TODO: how much of what was asked for has to be there? This is a policy decision, and the
    # self-check below states the bar: a retrieval that covers half of it must NOT pass.
    return coverage(need_terms, results) >= BLANK
''', '''
def coverage(need_terms: list, results: list) -> float:
    """How much of what the question needed actually appeared in what came back, 0.0 to 1.0."""
    need = {t.lower() for t in need_terms}
    if not need:
        return 0.0
    covered = set(content_words(" ".join(r["text"] for r in results)))
    return len(need & covered) / len(need)


def missing(need_terms: list, results: list) -> list:
    """What the question asked about that the results never mention."""
    covered = set(content_words(" ".join(r["text"] for r in results)))
    return sorted(t for t in need_terms if t.lower() not in covered)


def adequate(need_terms: list, results: list) -> bool:
    """Is this retrieval enough to answer from?"""
    if not results:
        return False
    return coverage(need_terms, results) >= 1.0     # every term, or it is not an answer yet
'''),
    code('''
# --- Self-check: Section 1   (over the real index -- no model)
SANCTIONS = ("what does a sanctions review need", ["sanctions", "compliance"])
HEDGING   = ("what is the JPY hedging policy",    ["hedging"])
HALF      = ("what does a sanctions review need", ["sanctions", "hedging"])   # one of two present

check("an empty retrieval is never adequate",
      lambda: adequate(["anything"], []) is False)
check("a retrieval that covers everything asked for IS adequate",
      lambda: adequate(SANCTIONS[1], search(SANCTIONS[0], k=3)) is True)
check("one that covers nothing is NOT, even though it returned rows",
      lambda: adequate(HEDGING[1], search(HEDGING[0], k=3)) is False,
      "three chunks came back and none is about hedging -- a length check would pass this")
check("half covered is not enough either",
      lambda: adequate(HALF[1], search(HALF[0], k=3)) is False,
      "that is the bar: a partial retrieval is a wrong answer waiting to be written")
check("coverage is a number you can log, not just a verdict",
      lambda: abs(coverage(HALF[1], search(HALF[0], k=3)) - 0.5) < 1e-9)
check("and it names what was missing",
      lambda: missing(HEDGING[1], search(HEDGING[0], k=3)) == ["hedging"],
      "'the corpus has nothing on hedging' is a useful answer; 'I don't know' is not")
'''),

    md("""
## Section 2 &mdash; The re-query, and the reason it is not a rewrite

Hop two's query contains a word you could not have known before hop one ran. That is what
&ldquo;multi-hop&rdquo; means &mdash; not three searches, but three searches where each is written from the
last one's answer.

On this corpus the handle is the reason code. `follow_up` reads the retrieved text and returns
the first code it did not already ask about.
"""),
    code('''
CODE_RE = re.compile(r"\\b(R\\d{2}|[A-Z]{2,}_[A-Z_]+)\\b")

def follow_up(results: list, asked: str):
    """A query built from a term the results just taught you, or None if they taught nothing."""
    found = []
    for r in results:
        found += CODE_RE.findall(r["text"])
    fresh = [f for f in found if f.lower() not in (asked or "").lower()]
    return fresh[0] if fresh else None
'''),
    code('''
# --- Self-check: Section 2   (string handling over real retrievals -- no model)
IBAN_Q = "what happens with a wrong beneficiary iban"

check("the first hop on a beneficiary question finds section 3.3",
      lambda: search(IBAN_Q, k=2)[0]["section"].startswith("3.3"))
check("and the results teach it a code the question never contained",
      lambda: follow_up(search(IBAN_Q, k=2), IBAN_Q) in ("INVALID_IBAN", "R04"),
      "that term came out of the corpus -- no rewrite of the question could have produced it")
check("a term already in the query is not chased again",
      lambda: follow_up(search("INVALID_IBAN", k=2), "INVALID_IBAN") != "INVALID_IBAN")
check("results that name no codes teach nothing, and say so",
      lambda: follow_up([{"text": "no codes here at all"}], "x") is None,
      "None, not an empty string -- the loop below branches on it")
'''),

    md("""
## Section 3 &mdash; The loop, as a graph

Four nodes. `retrieve` searches, `judge` scores the result, `requery` swaps in the new query, and
`report` writes the outcome. The conditional edge after `judge` is where every stop lives:
adequate, nothing left to chase, budget spent, or asking the same thing twice.

`hops` uses the `Annotated[list, add]` reducer from Module 3, so each pass appends its trace
instead of replacing it.
"""),
    code('''
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

MAX_HOPS = 3

class HuntState(TypedDict):
    question: str
    need: list
    query: str                          # the query THIS hop will send
    hops: Annotated[list, add]          # one entry per hop, appended
    results: list
    adequate: bool
    next_query: str | None
    outcome: str


def retrieve(state: HuntState) -> dict:
    hits = search(state["query"], k=3)
    return {"results": hits,
            "hops": [{"query": state["query"], "sections": [h["section"] for h in hits]}]}


def judge(state: HuntState) -> dict:
    ok = adequate(state["need"], state["results"])
    return {"adequate": ok,
            "next_query": None if ok else follow_up(state["results"], state["query"])}


def requery(state: HuntState) -> dict:
    return {"query": state["next_query"]}
'''),
    code('''
def report(state: HuntState) -> dict:
    """The last node. Say what happened, and hand back only what you may answer from."""
    if state["adequate"]:
        return {"outcome": "answered"}
    # TODO: this run did NOT find what was asked for. What should `results` be now?
    # Whatever is in it is what the answering step will read.
    return {"outcome": "not found", "results": BLANK}


def next_step(state: HuntState) -> str:
    """The conditional edge: go round again, or stop? Returns the KEY of the next branch."""
    if state["adequate"]:
        return "report"
    if state["next_query"] is None:                                   # nothing left to chase
        return "report"
    if len(state["hops"]) >= MAX_HOPS:                                # budget spent
        return "report"
    if any(h["query"] == state["next_query"] for h in state["hops"]): # asked that already
        return "report"
    return "requery"
''', '''
def report(state: HuntState) -> dict:
    """The last node. Say what happened, and hand back only what you may answer from."""
    if state["adequate"]:
        return {"outcome": "answered"}
    return {"outcome": "not found", "results": []}   # inadequate evidence is not evidence


def next_step(state: HuntState) -> str:
    """The conditional edge: go round again, or stop? Returns the KEY of the next branch."""
    if state["adequate"]:
        return "report"
    if state["next_query"] is None:                                   # nothing left to chase
        return "report"
    if len(state["hops"]) >= MAX_HOPS:                                # budget spent
        return "report"
    if any(h["query"] == state["next_query"] for h in state["hops"]): # asked that already
        return "report"
    return "requery"
'''),
    code('''
def build_hunt():
    """Wire the four nodes into a graph with one cycle, and compile it."""
    g = StateGraph(HuntState)
    g.add_node("retrieve", retrieve)
    g.add_node("judge", judge)
    g.add_node("requery", requery)
    g.add_node("report", report)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "judge")
    g.add_conditional_edges("judge", next_step, {"requery": "requery", "report": "report"})
    g.add_edge("requery", "retrieve")        # the backward edge -- this is the cycle
    g.add_edge("report", END)
    return g.compile()


def hunt(question: str, need: list) -> dict:
    """Run the loop for one question."""
    return build_hunt().invoke({"question": question, "need": need, "query": question,
                                "hops": [], "results": [], "adequate": False,
                                "next_query": None, "outcome": ""})
'''),
    code('''
# --- Self-check: Section 3   (a REAL compiled graph, running the real index -- still no model)
check("the graph compiles",
      lambda: build_hunt() is not None)
check("an answerable question is answered on the first hop",
      lambda: hunt(*SANCTIONS)["outcome"] == "answered" and len(hunt(*SANCTIONS)["hops"]) == 1)
check("and it hands back the evidence it answered from",
      lambda: len(hunt(*SANCTIONS)["results"]) == 3)
check("an unanswerable question ends as 'not found', not as a wrong answer",
      lambda: hunt(*HEDGING)["outcome"] == "not found")
check("and it hands back NOTHING to answer from",
      lambda: hunt(*HEDGING)["results"] == [],
      "handing the irrelevant chunks back anyway is how a refusal becomes a hallucination")
check("it went round again before giving up",
      lambda: len(hunt(*HEDGING)["hops"]) >= 2,
      "hop two used a term the corpus supplied -- it failed honestly, not lazily")
check("the hop budget is never exceeded",
      lambda: len(hunt(*HEDGING)["hops"]) <= MAX_HOPS)
check("the cycle cannot ask the same thing twice",
      lambda: len({h["query"] for h in hunt(*HEDGING)["hops"]})
              == len(hunt(*HEDGING)["hops"]))
check("the trace records every query it sent",
      lambda: all(set(h) == {"query", "sections"} for h in hunt(*HEDGING)["hops"]),
      "this is the log line you will want when someone asks why it said no")

def _traces():
    for question, need in (SANCTIONS, HEDGING, (IBAN_Q, ["r04", "originator"])):
        out = hunt(question, need)
        print(f"  {out['outcome']:10} {question[:44]}")
        for i, h in enumerate(out["hops"], 1):
            print(f"      hop {i}: {h['query'][:38]:40} -> {h['sections']}")
        if out["outcome"] == "not found":
            print(f"      missing: {missing(need, search(question, k=3))}")
guard(_traces)
'''),

    md("""
## Run it for real &mdash; let the model be the judge

Same two questions, but `adequate` is now the model. The one to watch is the second: a model asked
&ldquo;is this enough?&rdquo; about three irrelevant chunks has every incentive to say yes.
"""),
    code('''
if llm_ready():
    def _model_judge():
        for question, need in (SANCTIONS, HEDGING):
            results = search(question, k=3)
            context = "\\n".join(f"- [{r['section']}] {r['text'][:150]}" for r in results)
            verdict = ask(f"Question: {question}\\n\\nRetrieved:\\n{context}\\n\\n"
                          "Can this question be answered from the retrieved text alone? "
                          "Reply YES or NO, then one short sentence.",
                          system="Begin your reply with YES or NO.")
            print(f"  {question}")
            print(f"      model     : {verdict.strip()[:140]}")
            print(f"      coverage  : {coverage(need, results):.0%} -> "
                  f"{'adequate' if adequate(need, results) else 'not adequate'}")
            print()
    guard(_model_judge)
'''),
    md("""
### Read it

If the model says YES to the hedging question, you have watched the failure this lab exists to
prevent: the retrieval was inadequate, the judge was the same kind of thing that will write the
answer, and nothing stopped it.

The coverage check is crude and cannot be talked round. It also cannot tell a paraphrase from a
gap, which is why it is a floor and not a ceiling &mdash; in production you want both, the cheap
mechanical check underneath and the model for the judgements it is too blunt to make.

Notice what the graph bought you beyond the loop itself: every stop is one line in `next_step`,
and every run leaves a trace of exactly which queries were sent. Both of those are the difference
between an agent you can operate and one you can only demo.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Lower the adequacy bar to 0.5 and re-run `hunt(*HALF)`. It now answers. Read the evidence it
   answered from and decide whether you would sign that answer.
2. `follow_up` chases reason codes because that is what this corpus is made of. What is the
   equivalent handle in your corpus &mdash; a ticket id, a product code, a section number? Write the
   regex and see how far a chain gets.
3. Add a wall-clock deadline to `next_step` as well as the hop budget, then make one search slow.
   Which stop fires first, and which one would you actually have wanted?
"""),
]


# =========================================================================== #
# Lab 6.4 -- citations bound to spans, and refusing
# =========================================================================== #
LAB4 = [
    header(4, "Citations Bound to Spans, and Refusing", "Advanced", 40,
           ["Declare a citation as a <strong>Pydantic</strong> schema the model has to fill",
            "Pick the parser that actually rejects a bad one &mdash; one of the two does not",
            "Bind every claim to the exact characters that support it, and drop the ones that "
            "cannot be bound",
            "Refuse when the corpus cannot answer &mdash; structurally, and then in the prompt"],
           "> **Extractive grounding.** Every claim here is a quotation, so the binding is exact\n"
           "> and a citation is checkable by string comparison. Looser generation needs the\n"
           "> faithfulness score from Lab 6.5 &mdash; but this is the version you can prove."),
    setup(4),
    code(CORPUS),
    code(EMBEDDINGS),
    code(RETRIEVAL_STACK),

    md("""
## Concept

Two behaviours a regulated client will ask about, and both have to be **mechanisms** rather than
requests, because a request is something the model can decline to honour on any given run.

- **Citation** &mdash; not &ldquo;here are the documents that were in context&rdquo;, but *this claim came
  from these characters of that section*.
- **Refusal** &mdash; not &ldquo;the model decided it did not know&rdquo;, but *nothing cleared the floor, so
  there is nothing to answer from*.

The schema and the parser are how you state the first one to the model. The floor from Lab 6.1 is
how you get the second without asking for it.
"""),

    md("""
## Section 1 &mdash; Declare what a citation is

A Pydantic model is two things at once: the shape you validate against, and &mdash; through the
parser's format instructions &mdash; the description the model reads. The `Field` descriptions are
sent to the model verbatim, so they are instructions, not comments.
"""),
    code('''
from pydantic import BaseModel, Field

class Citation(BaseModel):
    """One claim, bound to the text that supports it."""

    claim: str = Field(description="BLANK")
    # TODO (claim): one line the model can follow. What is a claim here -- a whole answer, or
    # a single assertion that one span of one section can support on its own?

    quote: str = Field(description="BLANK")
    # TODO (quote): this is what gets matched against the source, character for character.
    # Say that it must be copied EXACTLY -- use the word "exactly" -- and never paraphrased.

    source: str = Field(description="the file the quote came from, e.g. 'ops-runbook-v4.md'")
    section: str = Field(description="the section heading the quote came from, e.g. '3.2 Limit breaches'")
''', '''
from pydantic import BaseModel, Field

class Citation(BaseModel):
    """One claim, bound to the text that supports it."""

    claim: str = Field(description="a single assertion, in one sentence, that one span of one "
                                   "section supports on its own -- not a whole answer")

    quote: str = Field(description="the supporting text copied exactly from that section, "
                                   "character for character, never paraphrased or shortened")

    source: str = Field(description="the file the quote came from, e.g. 'ops-runbook-v4.md'")
    section: str = Field(description="the section heading the quote came from, e.g. '3.2 Limit breaches'")
'''),
    code('''
# --- Self-check: Section 1   (the schema object -- no model)
def _rejects(fn) -> bool:
    """True if fn() refused its input. NameError is re-raised so a blank still prints [TODO]."""
    try:
        fn()
    except NameError:
        raise
    except Exception:
        return True
    return False

def _field_desc(name: str) -> str:
    d = (Citation.model_fields[name].description or "").strip()
    if d == "BLANK" or not d:
        raise NameError(f"{name} still has the placeholder description")
    return d

check("the schema declares all four fields",
      lambda: set(Citation.model_fields) == {"claim", "quote", "source", "section"})
check("every field carries a description the model will be shown",
      lambda: all(_field_desc(f) for f in Citation.model_fields))
check("the claim description says a claim is ONE assertion",
      lambda: any(w in _field_desc("claim").lower() for w in ("one ", "single", "a single")),
      "a citation attached to a whole paragraph cannot be checked against a span")
check("the quote description demands an exact copy",
      lambda: "exact" in _field_desc("quote").lower(),
      "a paraphrased quote cannot be found in the source, so it cannot be verified")
check("a well-formed citation validates",
      lambda: Citation(claim="c", quote="q", source="s", section="3.2").quote == "q")
check("and a citation with no section does not",
      lambda: _rejects(lambda: Citation(claim="c", quote="q", source="s")))
''', None),

    md("""
## Section 2 &mdash; The parser that actually rejects a bad one

LangChain gives you two parsers that both take `pydantic_object=`. Only one of them validates
against it. The other writes the format instructions and then hands you back whatever JSON the
model produced, missing fields and all &mdash; which looks identical right up to the moment
something downstream reads `citation["section"]`.
"""),
    code('''
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException

def citation_parser():
    """The parser used on model output. It must REJECT a citation that is missing a field."""
    # TODO: PydanticOutputParser or JsonOutputParser? The self-check below is the experiment --
    # try the other one and read what it does with `MISSING_SECTION`.
    return BLANK(pydantic_object=Citation)


GOOD_JSON = json.dumps({"claim": "Payments above USD 500,000 require Treasury approval",
                        "quote": "Payments above USD 500,000 require Treasury approval",
                        "source": "ops-runbook-v4.md", "section": "3.2 Limit breaches"})
MISSING_SECTION = json.dumps({"claim": "Payments above USD 500,000 require Treasury approval",
                              "quote": "Payments above USD 500,000 require Treasury approval",
                              "source": "ops-runbook-v4.md"})
''', '''
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException

def citation_parser():
    """The parser used on model output. It must REJECT a citation that is missing a field."""
    return PydanticOutputParser(pydantic_object=Citation)


GOOD_JSON = json.dumps({"claim": "Payments above USD 500,000 require Treasury approval",
                        "quote": "Payments above USD 500,000 require Treasury approval",
                        "source": "ops-runbook-v4.md", "section": "3.2 Limit breaches"})
MISSING_SECTION = json.dumps({"claim": "Payments above USD 500,000 require Treasury approval",
                              "quote": "Payments above USD 500,000 require Treasury approval",
                              "source": "ops-runbook-v4.md"})
'''),
    code('''
# --- Self-check: Section 2   (parser objects, on fixed strings -- no model)
check("a well-formed citation parses into a Citation object",
      lambda: isinstance(citation_parser().parse(GOOD_JSON), Citation),
      "a parser that hands back a dict has validated nothing")
check("A CITATION MISSING ITS SECTION IS REJECTED",
      lambda: _rejects(lambda: citation_parser().parse(MISSING_SECTION)),
      "JsonOutputParser(pydantic_object=...) accepts this quietly -- try it and see")
check("the format instructions carry your field descriptions to the model",
      lambda: "quote" in citation_parser().get_format_instructions()
              and "section" in citation_parser().get_format_instructions())
check("the instructions are worth sending -- they are the schema in words",
      lambda: len(citation_parser().get_format_instructions()) > 200)
'''),

    md("""
## Section 3 &mdash; No span, no claim

A citation that names a document proves nothing: the document was in the context whatever the
model wrote. A *span* is checkable &mdash; you can slice the source and compare.

`compose` is the control: a claim whose quote cannot be found in what was retrieved is
**dropped**, not flagged. Anything less and you have added a field, not a control.
"""),
    code('''
def normalise(text: str) -> str:
    return " ".join((text or "").split()).lower()


def find_span(quote: str, chunk: dict):
    """The (start, end) character range in the chunk that contains this quote, or None."""
    hay, needle = normalise(chunk["text"]), normalise(quote)
    i = hay.find(needle)
    return (i, i + len(needle)) if i >= 0 and needle else None


def bind(citation: Citation, results: list):
    """Attach the first retrieved chunk that actually contains this citation's quote."""
    for r in results:
        span = find_span(citation.quote, r)
        if span:
            return {"claim": citation.claim, "source": r["source"],
                    "section": r["section"], "span": span}
    return None


def compose(citations: list, results: list) -> dict:
    """Keep only the claims that can name their source. Report what was dropped."""
    bound = [(c, bind(c, results)) for c in citations]
    kept = [b for c, b in bound if b is not None]
    dropped = [c.claim for c, b in bound if b is None]
    return {"claims": [b["claim"] for b in kept], "dropped": dropped,
            "citations": [f"{b['source']}#{b['section']} [{b['span'][0]}:{b['span'][1]}]"
                          for b in kept]}
'''),
    code('''
# --- Self-check: Section 3   (span binding over real retrievals -- no model)
LIMIT_Q = "limit breach approval above USD 500,000"
LIMIT_HITS = search(LIMIT_Q, k=3)

SUPPORTED = Citation(claim="Large payments need Treasury approval",
                     quote="Payments above USD 500,000 require Treasury approval before release",
                     source="ops-runbook-v4.md", section="3.2 Limit breaches")
EXCEPTION = Citation(claim="Intra-group transfers are exempt",
                     quote="This does not apply to intra-group transfers",
                     source="ops-runbook-v4.md", section="3.2 Limit breaches")
INVENTED  = Citation(claim="The duty manager may release it",
                     quote="Payments above USD 500,000 may be released by the duty manager",
                     source="ops-runbook-v4.md", section="3.2 Limit breaches")

check("a supported citation finds its span",
      lambda: bind(SUPPORTED, LIMIT_HITS) is not None)
check("and the span points into the right section",
      lambda: bind(SUPPORTED, LIMIT_HITS)["section"].startswith("3.2"))
check("the span is a real character range you can slice",
      lambda: normalise(SUPPORTED.quote) in
              normalise(next(r["text"] for r in LIMIT_HITS
                             if r["section"] == bind(SUPPORTED, LIMIT_HITS)["section"])),
      "an auditor follows one link; the check is a string comparison, not a judgement")
check("AN INVENTED QUOTE BINDS TO NOTHING",
      lambda: bind(INVENTED, LIMIT_HITS) is None,
      "it is plausible, it is about the retrieved topic, and it is not in the text")
check("whitespace differences do not break a real citation",
      lambda: bind(Citation(claim="c", quote="Payments above USD 500,000\\n   require Treasury",
                            source="s", section="3.2"), LIMIT_HITS) is not None)
check("compose keeps the two supported claims and drops the invented one",
      lambda: compose([SUPPORTED, EXCEPTION, INVENTED], LIMIT_HITS)["dropped"]
              == [INVENTED.claim])
check("every surviving claim has a citation with a section and a range",
      lambda: all("#3.2" in c and "[" in c
                  for c in compose([SUPPORTED, EXCEPTION], LIMIT_HITS)["citations"]))
check("the exception survives alongside the rule, because Lab 6.1 chunked them together",
      lambda: EXCEPTION.claim in compose([SUPPORTED, EXCEPTION], LIMIT_HITS)["claims"],
      "chunk them apart and this claim becomes uncitable, so this control would delete it")
'''),

    md("""
## Section 4 &mdash; Refuse, usefully

There are two refusals here and they are not alternatives.

The **structural** one needs no co-operation: the floor from Lab 6.1 empties the result set, so
there is no context and nothing to be wrong from.

The **prompted** one matters when there *is* context and it still does not answer the question.
Measured on this sandbox: with an explicit refusal clause the model flagged it 3/3; without one,
0/3. The clause is not decoration.
"""),
    code('''
def refusal_clause() -> str:
    """The sentence in the system prompt that makes refusing an available answer."""
    # TODO: write it. It has to (a) tell the model to answer only from the context, (b) tell it
    # what to do when the context does not cover the question, and (c) make the refusal
    # MACHINE-READABLE by requiring the exact token INSUFFICIENT_CONTEXT in that case.
    return "BLANK"


def refused(reply: str) -> bool:
    """Did the model refuse? Read the token, not the tone."""
    return "INSUFFICIENT_CONTEXT" in (reply or "")


def respond(question: str, citations=None, floor: float = FLOOR) -> dict:
    """Answer from the corpus, or refuse and say what was missing. No model involved."""
    results = search(question, k=3, floor=floor)
    if not results:
        nearest = search(question, k=1)          # what we would have used, had we allowed it
        near = nearest[0]["section"] if nearest else "nothing"
        topic = ", ".join(sorted(set(content_words(question)))[:4])
        return {"answered": False, "citations": [],
                "why": f"nothing in the corpus clears the bar for [{topic}]; "
                       f"the closest section is {near}"}
    out = compose(citations or [], results)
    if not out["claims"]:
        return {"answered": False, "citations": [],
                "why": f"retrieved {len(results)} section(s) but no claim could name a span"}
    return {"answered": True, "citations": out["citations"],
            "claims": out["claims"], "dropped": out["dropped"]}
''', '''
def refusal_clause() -> str:
    """The sentence in the system prompt that makes refusing an available answer."""
    return ("Answer only from the CONTEXT below. If the context does not contain the answer, "
            "do not answer from your own knowledge: reply with the single word "
            "INSUFFICIENT_CONTEXT followed by one sentence saying what is missing.")


def refused(reply: str) -> bool:
    """Did the model refuse? Read the token, not the tone."""
    return "INSUFFICIENT_CONTEXT" in (reply or "")


def respond(question: str, citations=None, floor: float = FLOOR) -> dict:
    """Answer from the corpus, or refuse and say what was missing. No model involved."""
    results = search(question, k=3, floor=floor)
    if not results:
        nearest = search(question, k=1)          # what we would have used, had we allowed it
        near = nearest[0]["section"] if nearest else "nothing"
        topic = ", ".join(sorted(set(content_words(question)))[:4])
        return {"answered": False, "citations": [],
                "why": f"nothing in the corpus clears the bar for [{topic}]; "
                       f"the closest section is {near}"}
    out = compose(citations or [], results)
    if not out["claims"]:
        return {"answered": False, "citations": [],
                "why": f"retrieved {len(results)} section(s) but no claim could name a span"}
    return {"answered": True, "citations": out["citations"],
            "claims": out["claims"], "dropped": out["dropped"]}
'''),
    code('''
# --- Self-check: Section 4   (the clause as a string, the floor as a mechanism -- no model)
FX_Q = "what is the FX hedging policy for JPY exposure"

def _clause() -> str:
    c = refusal_clause().strip()
    if c == "BLANK" or not c:
        raise NameError("refusal_clause() is still the placeholder")
    return c

check("the clause is a real instruction, not a word",
      lambda: len(_clause()) > 60)
check("it confines the model to the context",
      lambda: "context" in _clause().lower())
check("and it names the token your code reads",
      lambda: "INSUFFICIENT_CONTEXT" in _clause(),
      "'say you do not know' is unreadable by machine -- refused() has to detect something exact")
check("refused() detects that token and not a mood",
      lambda: refused("INSUFFICIENT_CONTEXT nothing here covers FX") is True
              and refused("I'm not really sure about that") is False)
check("an answerable question is answered, with a citation",
      lambda: respond(LIMIT_Q, citations=[SUPPORTED])["answered"] is True
              and len(respond(LIMIT_Q, citations=[SUPPORTED])["citations"]) == 1)
check("a question the corpus cannot answer is REFUSED before any model sees it",
      lambda: respond(FX_Q, citations=[SUPPORTED])["answered"] is False,
      "the floor emptied the result set -- there is no context to be wrong from")
check("the refusal names what was searched for and what is nearby",
      lambda: "hedging" in respond(FX_Q)["why"] and "closest section" in respond(FX_Q)["why"],
      "that sentence is a work item for whoever owns the corpus")
check("retrieving something and supporting nothing also refuses",
      lambda: respond(LIMIT_Q, citations=[INVENTED])["answered"] is False,
      "the second gate: results cleared the floor, and still no claim could name a span")
'''),

    md("""
## Run it for real &mdash; part 1: ask the model to cite

The schema's format instructions go into the prompt, the model answers, your parser validates it,
and `bind` checks the quote against the text it claims to come from.
"""),
    code('''
if llm_ready():
    def _cited_answer():
        parser = citation_parser()
        results = search(LIMIT_Q, k=3)
        context = "\\n\\n".join(f"[{r['source']} #{r['section']}]\\n{r['text']}" for r in results)
        raw = ask(f"CONTEXT:\\n{context}\\n\\nQuestion: What approval does a payment above "
                  f"USD 500,000 need?\\n\\n{parser.get_format_instructions()}",
                  system="Reply with the JSON object only.")
        print("  raw reply:", raw.strip()[:200].replace("\\n", " "))
        try:
            cited = parser.parse(raw)
        except Exception as exc:
            print(f"  parser REJECTED it: {type(exc).__name__} -- and that is the parser working")
            return
        bound = bind(cited, results)
        print(f"  claim   : {cited.claim[:90]}")
        print(f"  quote   : {cited.quote[:90]}")
        print(f"  binds to: {bound['section'] + ' ' + str(bound['span']) if bound else 'NOTHING'}")
    guard(_cited_answer)
'''),
    md("""
## Run it for real &mdash; part 2: the refusal clause is the whole difference

The same unanswerable question, three times: with no context at all, with the four nearest chunks
and no refusal clause, and with the same four chunks and your clause.
"""),
    code('''
if llm_ready():
    def _refusals():
        nearest = search(FX_Q, k=4)                    # deliberately NO floor -- the wrong context
        context = "\\n".join(f"- [{r['section']}] {r['text'][:160]}" for r in nearest)
        arms = [
            ("floor applied: no context",  "(no documents were retrieved)", _clause()),
            ("context, no refusal clause", context, "Be brief."),
            ("context, refusal clause",    context, _clause()),
        ]
        for label, ctx, system in arms:
            reply = ask(f"CONTEXT:\\n{ctx}\\n\\nQuestion: {FX_Q}", system=system)
            print(f"  [{label:28}] refused={refused(reply)}")
            print(f"      {reply.strip()[:180]}")
            print()
    guard(_refusals)
'''),
    md("""
### Read it

**The citation.** If the parser rejected the reply, that is the parser doing its job &mdash; note that
`JsonOutputParser` would have accepted the same string and handed you a dict with a missing key.
If it parsed and then bound to nothing, look at the quote: it will be a *paraphrase*. That is why
the `quote` description says &ldquo;exactly&rdquo;, and it is the commonest way a citation stops being
checkable.

**The refusals.** The first arm is structural: no context, nothing to be wrong from, and it does not
depend on the model behaving. The third arm is the clause working. The middle arm is the one to
stare at &mdash; four chunks about payment limits in front of a question about FX, and nothing but
the model's own judgement between you and an answer stitched out of the nearest available prose.

Sometimes the middle arm refuses anyway. That is worth noticing and not worth relying on: you
cannot put &ldquo;the model was sensible&rdquo; in a control document, and it is not the same sentence in
the next model version.
"""),

    code('''
score()
'''),
    md("""
## Your turn

1. Swap `citation_parser()` to `JsonOutputParser` and re-run the Section 2 checks. Then find the
   line of code downstream that would have crashed in production instead.
2. Extractive grounding is the strictest kind and the least fluent. Let the model paraphrase, then
   decide how you would still bind a claim to a span &mdash; and what you lose when the match stops
   being exact.
3. `respond` drops unsupported claims silently. Log them instead, and after a day of traffic read
   the log: the claims a model keeps trying to make and cannot support are a map of what your
   corpus is missing.
"""),
]


# =========================================================================== #
# Lab 6.5 -- challenge: score retrieval and generation apart
# =========================================================================== #
LAB5 = [
    header(5, "Challenge: Score the Two Halves Apart", "Advanced &middot; challenge", 40,
           ["Label an eval set: which section <em>should</em> have come back, for each question",
            "Measure retrieval with recall@k and precision@k, and choose k with the numbers",
            "Measure generation with faithfulness and answer relevance &mdash; no labels needed",
            "Put a run in the 2&times;2 and read off which half to fix"],
           "> **The bridge into Day 3.** One end-to-end score cannot tell a lucky answer from a\n"
           "> grounded one. Two scores can, and the fixes are unrelated."),
    setup(5),
    code(CORPUS),
    code(EMBEDDINGS),
    code(RETRIEVAL_STACK),

    md("""
## Concept

A RAG system is two systems. They fail differently, they are fixed differently, and a single
end-to-end score cannot tell you which one broke.

The worst box in the 2&times;2 is **right answer, wrong evidence** &mdash; the model knew it already, or
guessed well. It passes every demo, and it stops working the day the model changes.
"""),

    md("""
## Section 1 &mdash; Label the eval set, then choose k

Anything measured `@k` needs a ground truth: for this question, which section *should* come back?
Producing those labels is the actual work, and there is no shortcut.

Then k. Raising it always helps recall and always hurts precision, so &ldquo;pick a good k&rdquo; is not a
judgement call &mdash; it is a rule you state and then read off the sweep.
"""),
    code('''
# (question, the section that answers it, one true claim quoted from that section)
LABELLED = [
    ("What approval is needed above USD 500,000?", "3.2 Limit breaches",
     "Payments above USD 500,000 require Treasury approval before release"),
    ("What happens to a payment returned INVALID_IBAN?", "3.3 Invalid beneficiary details",
     "A payment returned INVALID_IBAN is returned to the originator with code R04"),
    ("Who decides on a payment held for sanctions review?", "3.4 Sanctions review",
     "A payment held for SANCTIONS_REVIEW is decided by Compliance"),
    ("How much may a duty manager approve?", "1 Approval authority",
     "A duty manager may approve a release up to USD 250,000"),
    ("What happens when a payment is returned INSUFFICIENT_FUNDS?", "3.1 Insufficient funds",
     "A payment returned INSUFFICIENT_FUNDS is retried once after 24 hours"),
    ("How long before an unanswered approval escalates?", "2 Escalation timers",
     "If an approver has not responded within 15 minutes, escalate to the Treasury lead"),
]

def retrieved_sections(question: str, k: int) -> list:
    return [r["section"] for r in search(question, k=k)]


def recall_at_k(k: int) -> float:
    """Fraction of questions whose answering section came back at all."""
    return sum(1 for q, want, _ in LABELLED
               if want in retrieved_sections(q, k)) / len(LABELLED)


def precision_at_k(k: int) -> float:
    """Of everything returned across the eval set, what fraction was the right section?"""
    returned = sum(len(retrieved_sections(q, k)) for q, _, _ in LABELLED)
    correct = sum(1 for q, want, _ in LABELLED if want in retrieved_sections(q, k))
    return correct / returned if returned else 0.0
'''),
    code('''
# The sweep. Read it before you choose anything.
def _sweep():
    for k in (1, 2, 3, 4, 5):
        print(f"  k={k}   recall {recall_at_k(k):.0%}   precision {precision_at_k(k):.0%}")
guard(_sweep)
'''),
    code('''
def chosen_k() -> int:
    """How many chunks to retrieve.

    The rule, stated before looking: the SMALLEST k that finds the answering section for
    every question in the eval set. Anything larger only adds noise; anything smaller
    loses an answer.
    """
    return BLANK      # TODO: read it off the sweep above
''', '''
def chosen_k() -> int:
    """How many chunks to retrieve.

    The rule, stated before looking: the SMALLEST k that finds the answering section for
    every question in the eval set. Anything larger only adds noise; anything smaller
    loses an answer.
    """
    return 3          # recall is 100% from k=3 up, and precision only falls after that
'''),
    code('''
# --- Self-check: Section 1   (labels and metrics over the real index -- no model)
check("every labelled section really exists in the index",
      lambda: all(want in [m["section"] for m in store().get()["metadatas"]]
                  for _, want, _ in LABELLED))
check("every labelled claim really appears in its section",
      lambda: all(" ".join(claim.split()).lower() in
                  " ".join(next(d for d, m in zip(store().get()["documents"],
                                                  store().get()["metadatas"])
                                if m["section"] == want).split()).lower()
                  for _, want, claim in LABELLED),
      "a mislabelled ground truth measures your labelling, not your retriever")
check("recall never decreases as k grows",
      lambda: recall_at_k(5) >= recall_at_k(3) >= recall_at_k(1))
check("precision does the opposite -- more results, more noise",
      lambda: precision_at_k(1) > precision_at_k(5),
      "that trade is the whole of retrieval tuning")
check("at k=1 precision and recall are the same number",
      lambda: abs(precision_at_k(1) - recall_at_k(1)) < 1e-9)
check("your k finds every answering section",
      lambda: recall_at_k(chosen_k()) == 1.0)
check("and it is the smallest k that does",
      lambda: recall_at_k(chosen_k() - 1) < 1.0,
      "a larger k passes the check above too, and pays for chunks nothing needed")
'''),

    md("""
## Section 2 &mdash; The half you can measure without labels

Faithfulness and answer relevance need only what is already in a trace: the question, the
retrieved text, and the answer. No ground truth, so this is the half you can start measuring on
Monday.
"""),
    code('''
def normalise(t: str) -> str:
    return " ".join((t or "").split()).lower()


def faithfulness(claims: list, results: list) -> float:
    """Fraction of the answer's claims that the retrieved text actually contains."""
    if not claims:
        # TODO: an answer that asserts nothing. Is that perfectly faithful, or not faithful
        # at all? Whichever you choose, every empty answer will score it.
        return BLANK
    context = normalise(" ".join(r["text"] for r in results))
    return sum(1 for c in claims if normalise(c) in context) / len(claims)


def answer_relevance(question: str, claims: list) -> float:
    """How much of what the question asked about the answer actually addresses."""
    asked = set(content_words(question))
    if not asked:
        return 0.0
    answered = set(content_words(" ".join(claims)))
    return len(asked & answered) / len(asked)
''', '''
def normalise(t: str) -> str:
    return " ".join((t or "").split()).lower()


def faithfulness(claims: list, results: list) -> float:
    """Fraction of the answer's claims that the retrieved text actually contains."""
    if not claims:
        return 0.0      # saying nothing is not the same as saying only supported things
    context = normalise(" ".join(r["text"] for r in results))
    return sum(1 for c in claims if normalise(c) in context) / len(claims)


def answer_relevance(question: str, claims: list) -> float:
    """How much of what the question asked about the answer actually addresses."""
    asked = set(content_words(question))
    if not asked:
        return 0.0
    answered = set(content_words(" ".join(claims)))
    return len(asked & answered) / len(asked)
'''),
    code('''
# --- Self-check: Section 2   (metrics on fixed strings and real retrievals -- no model)
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
      "score it 1.0 and every refusal becomes your best-performing answer")
check("an on-topic answer is relevant",
      lambda: answer_relevance(Q0, [TRUE0]) > 0.5)
check("a perfectly grounded answer to a DIFFERENT question is faithful and irrelevant",
      lambda: faithfulness([LABELLED[2][2]], search(LABELLED[2][0], k=3)) == 1.0
              and answer_relevance(Q0, [LABELLED[2][2]]) < 0.4,
      "the two metrics are independent, which is exactly why you need both")
check("neither metric needed a label",
      lambda: faithfulness([TRUE0], HITS0) == 1.0 and answer_relevance(Q0, [TRUE0]) > 0.0)
'''),

    md("""
## Section 3 &mdash; Read the 2&times;2

Two questions per run: *did the right evidence come back?* and *is the answer supported by what
did?* Four combinations, and each one names a different half to fix.
"""),
    code('''
def boxes() -> dict:
    """(right evidence?, faithful?) -> (what happened, which half to fix)."""
    return {
        (True,  True):  ("grounded",                   "nothing"),
        (True,  False): ("ignored the evidence",       "generation"),
        # TODO: the answer quoted the text in front of it -- and it was the wrong text.
        # Fixing the model cannot help. Which half is broken?
        (False, True):  ("faithful to the wrong text", BLANK),
        (False, False): ("unsupported",                "both"),
    }


def diagnose(question: str, want_section: str, claims: list, k: int = 3) -> dict:
    """Which of the four boxes is this run in, and what should be fixed?"""
    results = search(question, k=k)
    got_evidence = want_section in [r["section"] for r in results]
    faithful = faithfulness(claims, results) == 1.0
    box, fix = boxes()[(got_evidence, faithful)]
    return {"box": box, "fix": fix, "evidence": got_evidence, "faithful": faithful}
''', '''
def boxes() -> dict:
    """(right evidence?, faithful?) -> (what happened, which half to fix)."""
    return {
        (True,  True):  ("grounded",                   "nothing"),
        (True,  False): ("ignored the evidence",       "generation"),
        (False, True):  ("faithful to the wrong text", "retrieval"),
        (False, False): ("unsupported",                "both"),
    }


def diagnose(question: str, want_section: str, claims: list, k: int = 3) -> dict:
    """Which of the four boxes is this run in, and what should be fixed?"""
    results = search(question, k=k)
    got_evidence = want_section in [r["section"] for r in results]
    faithful = faithfulness(claims, results) == 1.0
    box, fix = boxes()[(got_evidence, faithful)]
    return {"box": box, "fix": fix, "evidence": got_evidence, "faithful": faithful}
'''),
    code('''
# --- Self-check: Section 3   (the diagnosis, on real retrievals -- no model)
def missed_section(question, k=3):
    """A real section this question does NOT retrieve -- computed, not assumed.

    Hard-coding one is how you write a check that passes for the wrong reason: the section
    you picked as 'wrong' may well be in the top k.
    """
    got = retrieved_sections(question, k)
    return next(m["section"] for m in store().get()["metadatas"] if m["section"] not in got)

def quoted_from_top(question, k=3):
    """A claim lifted verbatim from whatever DID come back, so it is faithful by construction."""
    return search(question, k=k)[0]["text"][:60]

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
check("AN END-TO-END SCORE CANNOT SEPARATE THESE TWO RUNS",
      lambda: diagnose(Q0, WANT0, [TRUE0])["box"]
              != diagnose(Q0, missed_section(Q0), [quoted_from_top(Q0)])["box"],
      "both return a fluent, supported-looking answer; only two scores tell them apart")

def _scorecard():
    print(f"  chosen k = {chosen_k()}   recall {recall_at_k(chosen_k()):.0%}   "
          f"precision {precision_at_k(chosen_k()):.0%}\\n")
    for q, want, claim in LABELLED:
        d = diagnose(q, want, [claim], k=chosen_k())
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
            results = search(q, k=chosen_k())
            context = "\\n".join(f"- {r['text'][:170]}" for r in results)
            reply = ask(f"Context:\\n{context}\\n\\nQuestion: {q}\\n\\n"
                        "Answer in one sentence, quoting the context as closely as you can.",
                        system="Be brief and stay inside the context.")
            claims = [reply.strip()]
            d = diagnose(q, want, claims, k=chosen_k())
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
   paraphrase and rejects `LIE0`. How confident are you in that threshold on six examples?
2. Add a seventh labelled question whose answer spans **two** sections. What does recall@k mean
   now, and what did you have to decide about the label?
3. Sweep the floor from Lab 6.4 across 0.1 to 0.5 and plot refusals against correct answers. The
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
