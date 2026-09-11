#!/usr/bin/env python3
"""
Generate Module 9 lab notebooks and their solutions from one source.

Every code cell is declared once. Where the lab and the solution differ, the cell
carries both variants, so a blank can never drift from the answer that grades it.

    python3 gen_labs.py          # writes ../lab-9-0N-*.ipynb and ../solutions/

Design rules (rebuilt 2026-09-09 to match the framework-forward Day 1 rebuild):

  * Self-checks assert on framework OBJECTS -- a FastAPI route table, a Pydantic
    request contract, a LangChain @tool, a manifest dict -- not on hand-rolled
    stand-ins. Building any of those needs no endpoint, so the checks stay
    deterministic. Only model INVOCATION needs the gateway, and that lives in
    "Run it for real" cells, which are observed, not scored.
  * Blanks ask a DESIGN DECISION, never a Python idiom. If the answer is a
    comprehension, a slice, a dict lookup or an f-string, the code is given and the
    blank moves to what only understanding answers: which probe owns a condition,
    which status code a refusal deserves, which finding blocks a release, which
    traces tail sampling must keep, which signal an autoscaler should watch.
  * "BLANK" marks a blank; an unfilled blank raises NameError and prints [TODO].
    NOT three underscores: IPython PREDEFINES _, __ and ___ as its output history
    (they start as ""), so under a real Jupyter kernel that token is a defined empty
    string, not an undefined name. The NameError never fires, [TODO] silently becomes
    [FAIL], and a blank used as a loop guard is falsy forever -- lab 1.1 once spun in
    `while True` until the pod was OOM-killed. Plain-exec verifiers cannot see any of
    this, which is why verify_labs.py runs cells through a real kernel too.
  * Blanks live INSIDE function bodies. A module-level `x = BLANK` crashes the cell
    instead of printing [TODO], and so does a module-level CALL into a blanked
    function -- so anything that would trip one is built lazily or wrapped in guard().
  * A blank inside a STRING is not a blank: "BLANK" is a defined literal, so it can
    never raise NameError. Where one is unavoidable, raise it by hand.

MODULE 9 KEEPS ITS OWN RULE ON TOP OF ALL THAT: **a graded cell must never need a
cluster.** Module 9 is about deployment, and deployment means Kubernetes -- but the
manifests here are ordinary Python dicts, and the notebook writes them with json.dump.
That is not a simplification for teaching: kubectl accepts JSON, because a Kubernetes
manifest IS JSON and YAML is a surface syntax over it. The linting is therefore stdlib,
offline and exact, and the same object is what `kubectl apply` receives in the "Run it
for real" cell. `--dry-run=server` never appears in a graded cell, and the namespace is
read ONLY from APP_NAMESPACE, never derived from a hostname.

Langfuse is live-only, in every module. No graded cell touches it.
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
# Lab 9.{num} &mdash; {title}

**Level:** {level} &nbsp;|&nbsp; **Est. time:** {minutes} min &nbsp;|&nbsp; **Day 3 &middot; Module 9 &mdash; Deployment &amp; AgentOps**

### What you'll do
{items}

> **How this lab works.** You write real FastAPI, Pydantic, LangChain and Kubernetes-manifest
> code. Fill every `BLANK`, then run the **Self-check** cell under each section &mdash; those
> assert on the *objects you built* (a route table, a request contract, a compiled tool, a
> manifest dict), so they are deterministic. **No graded cell needs a cluster, a running server
> or a model.** Cells marked **Run it for real** put your code in front of the sandbox model,
> your own namespace or the tracing backend; if any of those is unreachable they print how to
> fix it instead of crashing. The score line is feedback, not a grade.

{note}
""")


SETUP_COMMON = '''
# ---------------------------------------------------------------- Setup: run me first
import os, json, time, math, textwrap
from typing import Any, Callable, Optional

WORK = os.path.join("/tmp", "awmas-lab-9-{num:02d}")
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

# The served model can reason before it answers, and the reasoning is billed as completion
# tokens. Off is the default here because a deployment lab makes a lot of small calls.
NO_THINK = {{"chat_template_kwargs": {{"enable_thinking": False}}}}

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
        _llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                          temperature=temperature, extra_body=NO_THINK)
    return _llm

def ask(prompt: str, system: str | None = None) -> str:
    """One stateless call. Returns text, or an error string -- never raises."""
    try:
        msgs = ([("system", system)] if system else []) + [("human", prompt)]
        return get_llm().invoke(msgs).content
    except Exception as exc:
        return f"<model unavailable: {{type(exc).__name__}}: {{exc}}>"

# ---- your own namespace --------------------------------------------------
# You deploy into your own namespace, published at your own host. Both are injected into
# the sandbox, so nothing here is hardcoded and nothing here needs them to be set.
#
# Read ONLY from APP_NAMESPACE, never derived from the hostname. A cell below runs
# kubectl against whatever this says, and a namespace guessed from a machine name is
# the wrong thing to point kubectl at.
APP_NS   = os.environ.get("APP_NAMESPACE", "")
APP_HOST = os.environ.get("APP_HOST", "")

print("work dir :", WORK)
print("model    :", LLM_MODEL or "(not configured -- the object-level self-checks still work)")
print("namespace:", APP_NS or "(unknown -- no graded cell needs it)")
'''


def setup(num, extra=""):
    return code(SETUP_COMMON.format(num=num) + extra)


# --------------------------------------------------------------------------- #
# the app being deployed
# --------------------------------------------------------------------------- #
# FrontDesk AI is the multi-agent support desk that runs on the same LiteLLM
# gateway your notebooks use. It is already built and published; this lab is
# about getting it running in YOUR namespace, which is the part the course has
# not covered yet.
INFO_CLUSTER = """
## What you are deploying, and where

**FrontDesk AI** &mdash; a FastAPI service wrapping the LangGraph support desk you have been
studying all week: supervisor &rarr; RAG &rarr; a domain worker with tools &rarr; escalation
&rarr; QA gate. The image is already built and published, so there is nothing to compile here.

You are putting it in **your own namespace**, on **your own hostname**, and a browser will
reach it. That is a different job from running a notebook, and most of it is decided before
`kubectl` is involved.

### Your namespace is not an unlimited machine

Read these off your own quota any time with
`kubectl -n $APP_NAMESPACE describe quota`. Three of them force a decision below.

| Limit | Value | Why it matters |
|---|---|---|
| `limits.memory` | **1Gi &mdash; for the whole namespace** | one pod at a 1Gi limit uses all of it |
| `services.nodeports` | **0** | not "a few". Zero |
| `services.loadbalancers` | **0** | so there is exactly one way in |
| `services` | 3 | |
| `persistentvolumeclaims` | 2 | |
| `pods` | 8 | |
| `requests.storage` | 5Gi | |

### And it can only reach four things

Its NetworkPolicy is `participant-egress`. This is **not** the sandbox your notebook runs
in &mdash; that one does have general internet access. Yours does not.

| Allowed out | For |
|---|---|
| `kube-system` :53/UDP | DNS |
| `llm-serving` any port | the LiteLLM gateway &mdash; how the app reaches a model |
| `ingress-nginx` pods | so your Ingress can reach back in |
| `tempo` :4317 &middot; `langfuse` :3000 | telemetry |

Nothing else on the internet. Not PyPI, not Hugging Face, not a model vendor.

### Three Secrets are already there for you

You do not create these and you never copy their values out:
`<your-namespace>-llm` (gateway URL, model name, your own capped key),
`<your-namespace>-langfuse` (tracing keys), and `frontdeskai-secret`
(`SECRET_KEY` and the first-login password, created during the deploy).
"""


INFO_OBJECTS = """
## Section 1 &mdash; The five objects, and the five answers

A deployment of this app is five Kubernetes objects. You do not type them out &mdash; they are
written for you in the cell after next &mdash; but you do have to decide five things inside
them, and each answer is forced by a number in the panel above.

| Object | What it carries |
|---|---|
| **ConfigMap** | non-secret settings: which provider, which model, where spans go. Never a credential |
| **PersistentVolumeClaim** | 1Gi at `/shared` for SQLite and the vector store |
| **Deployment** | one replica, the image, both probes, requests and limits, and `envFrom` |
| **Service** | gives the pod a stable name inside the cluster, port 80 &rarr; 8000 |
| **Ingress** | claims your public hostname and sends it to the Service |

### The four decisions

| Decision | The fact that forces it | What to weigh |
|---|---|---|
| `update_strategy()` | `limits.memory` is 1Gi for the **whole namespace** | a `RollingUpdate` starts the replacement pod *before* stopping the old one, so for a moment two pods each want 1Gi. The second is quota-denied and the rollout hangs with no useful message |
| `service_type()` | `nodeports: 0`, `loadbalancers: 0` | an Ingress reaches a Service from *inside* the cluster, so it does not need either |
| `llm_provider()` | egress reaches `llm-serving` and nothing else | `groq`, `ollama` and `openrouter` each call a vendor over the internet. `litellm` points at an OpenAI-compatible gateway and reads its URL and key from the environment |
| `gateway_secret()` | the three Secrets listed above | which one holds the gateway credential |

### And one trap

`home_override()` looks like housekeeping and is not. The image bakes ChromaDB's embedding
model into `/opt/appcache` and points `HOME` there, because `chromadb` resolves its cache from
`Path.home()` and **downloads the model on first use** &mdash; which your namespace has no
egress to do. The app also mounts a PVC at `/shared`, which is a very tempting `HOME`.

Set `HOME` to `/shared` and the app crash-loops on `httpx.ConnectError` during startup
indexing, while every manifest is valid and both probes are configured correctly. It has
already broken this app once.
"""


INFO_APPLY = """
## Section 2 &mdash; Put it on the cluster

A Kubernetes manifest **is** JSON, and YAML is just a friendlier surface over it &mdash; so the
dicts you linted are exactly what `kubectl` receives. Nothing is translated in between, which
is why linting them offline was worth doing.

These cells are marked **Run it for real**. They need `APP_NAMESPACE`, and they print what to
do instead if it is not set. The next two write the five objects to disk as JSON, create the
app Secret, and apply the set.
"""


MD_PROBE = """
### Waiting, and asking the probe its own question

The first image pull is the slow part, so the rollout can take a couple of minutes.

Then the cell asks `/health` **from inside the running container**. The obvious way would be to
`kubectl run` a small curl pod &mdash; try it, and it is refused with
`exceeded quota: limits.memory=1Gi, used: 1Gi`, because your app is already holding the entire
namespace budget. A debugging pod is still a pod. `exec` borrows the container you have already
paid for, and this image ships Python.
"""


MD_ASK = """
### Asking the running service a real question

This is the whole course in one HTTP request: supervisor &rarr; RAG &rarr; a domain worker with
tools &rarr; escalation check &rarr; QA gate &rarr; a grounded answer. Watch the audit trail it
prints &mdash; that is the same trail you have been reading all week, now coming out of a
service instead of a notebook.

⚠️ **It asks the pod directly rather than your public hostname, and that is deliberate.**
Cloudflare sits in front of that host and gives up at about 100 seconds with a `524`; an agent
doing three tool-calling turns can exceed that whenever the gateway is busy, and the `524` looks
exactly like a broken app while the pod is still working. It also answers the default Python
user agent with `403 error code: 1010`. **Open the hostname in a browser** to see the real
thing, and let this cell use `kubectl exec`.
"""


MD_TRACE = """
### Reading your own trace back out

The request you just made emitted spans. This asks Tempo what it actually **received**.

That distinction is the point: *&ldquo;the exporter is configured&rdquo;* is not evidence.
`BatchSpanProcessor` swallows export failures, so a wrong endpoint looks exactly like a healthy
one from inside the app &mdash; no error, no warning, no spans. The only proof is reading them
back.

Spans are batched, so give it about ten seconds after a request. If the newest trace looks
partial, that is why: a trace's root span lands last.
"""



FACTS = '''
# ------------------------------------------------------- the few values code needs
# Everything else about this cluster is in the panel above -- read that, not this.
APP_IMAGE = "brainupgrade/frontdeskai:latest"
APP_PORT  = 8000            # what uvicorn listens on inside the container
APP_NAME  = "frontdeskai"   # every object in this lab is named after it

# Secrets already published into your namespace. You never copy their values.
SECRETS = {
    "llm":      "{ns}-llm",             # gateway base URL, model name, your capped key
    "langfuse": "{ns}-langfuse",        # tracing keys, scoped to your environment tag
    "app":      "frontdeskai-secret",   # SECRET_KEY + first-login password, made at deploy
}

# The three quota numbers the self-checks below actually compare against.
QUOTA = {"limits.memory": "1Gi", "services": 3, "persistentvolumeclaims": 2}

# Tempo is shared by the whole cohort: your namespace may SEND to 4317, your
# sandbox may READ from 3200, which is what the last cell of this lab uses.
TEMPO_OTLP  = "http://tempo.monitoring.svc.cluster.local:4317"
TEMPO_QUERY = "http://tempo.monitoring.svc.cluster.local:3200"

NS   = APP_NS   or "your-namespace"      # placeholder keeps every cell runnable offline
HOST = APP_HOST or f"{NS}-app.example"

print("deploying :", APP_IMAGE)
print("namespace :", NS)
print("host      :", HOST)
'''

# --------------------------------------------------------------------------- #
# the four decisions
# --------------------------------------------------------------------------- #
DECISIONS_LAB = '''
# Five answers. The reasoning for each one is in the panel above -- none of these
# is a Python puzzle, and the mechanics are already written.

def update_strategy() -> dict:
    """One replica, 1Gi limit, and 1Gi for the whole namespace."""
    return {"type": BLANK}              # "RollingUpdate" | "Recreate"


def service_type() -> str:
    """services.nodeports is 0, and so is services.loadbalancers."""
    return BLANK                        # "ClusterIP" | "NodePort" | "LoadBalancer"


def llm_provider() -> str:
    """Only the in-cluster gateway is reachable from this namespace."""
    return BLANK                        # "groq" | "ollama" | "openrouter" | "litellm"


def gateway_secret(ns: str) -> str:
    """Which Secret carries that credential. Return the key into SECRETS."""
    return SECRETS[BLANK].format(ns=ns) # "llm" | "langfuse" | "app"


def home_override() -> dict:
    """Extra env for HOME, if any. This is the trap."""
    return BLANK                        # {} | {"HOME": "/shared"}
'''

DECISIONS_SOL = (DECISIONS_LAB
    .replace('{"type": BLANK}              #', '{"type": "Recreate"}         #')
    .replace('return BLANK                        # "ClusterIP"',
             'return "ClusterIP"                  # "ClusterIP"')
    .replace('return BLANK                        # "groq"',
             'return "litellm"                    # "groq"')
    .replace('SECRETS[BLANK]', 'SECRETS["llm"]')
    .replace('return BLANK                        # {}',
             'return {}                           # {}'))


# --------------------------------------------------------------------------- #
# the five objects
# --------------------------------------------------------------------------- #
BUILDERS = '''
# The five objects, written for you. The five calls you filled in are the only
# things that vary.

def meta(name, ns):
    return {"name": name, "namespace": ns, "labels": {"app": APP_NAME}}


def build_configmap(ns):
    """Non-secret configuration. Nothing in here may be a credential."""
    env = {
        "LLM_PROVIDER":                llm_provider(),
        "LLM_MODEL":                   "qwen36-35b-a3b-lab",
        "LLM_FALLBACK_PROVIDER":       "",     # no second vendor is reachable from here
        "LLM_FALLBACK_MODEL":          "",     # empty model disables the fallback
        "SEED_DEMO_DATA":              "true",
        "SQLITE_DIR":                  "/shared/.sqlite",
        "OTEL_SERVICE_NAME":           f"{APP_NAME}-{ns}",
        "OTEL_EXPORTER_OTLP_ENDPOINT": TEMPO_OTLP,
        "LOG_LEVEL":                   "INFO",
        "ENV":                         "production",
    }
    env.update(home_override())
    return {"apiVersion": "v1", "kind": "ConfigMap",
            "metadata": meta(f"{APP_NAME}-config", ns), "data": env}


def build_deployment(ns):
    probe = lambda: {"httpGet": {"path": "/health", "port": APP_PORT}}
    return {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": meta(APP_NAME, ns),
        "spec": {
            "replicas": 1,
            "strategy": update_strategy(),
            "selector": {"matchLabels": {"app": APP_NAME}},
            "template": {
                "metadata": {"labels": {"app": APP_NAME}},
                "spec": {
                    "securityContext": {"runAsNonRoot": True, "runAsUser": 1000, "fsGroup": 1000},
                    "containers": [{
                        "name": APP_NAME,
                        "image": APP_IMAGE,
                        "imagePullPolicy": "Always",
                        "ports": [{"containerPort": APP_PORT, "name": "http"}],
                        "envFrom": [
                            {"configMapRef": {"name": f"{APP_NAME}-config"}},
                            {"secretRef": {"name": gateway_secret(ns), "optional": True}},
                            {"secretRef": {"name": SECRETS["app"]}},
                        ],
                        "volumeMounts": [{"name": "data", "mountPath": "/shared"}],
                        "livenessProbe":  {**probe(), "initialDelaySeconds": 30, "periodSeconds": 30},
                        "readinessProbe": {**probe(), "initialDelaySeconds": 15, "periodSeconds": 10},
                        "resources": {
                            "requests": {"cpu": "200m", "memory": "256Mi"},
                            "limits":   {"cpu": "500m", "memory": "1Gi"},
                        },
                        "securityContext": {"allowPrivilegeEscalation": False,
                                            "capabilities": {"drop": ["ALL"]}},
                    }],
                    "volumes": [{"name": "data",
                                 "persistentVolumeClaim": {"claimName": f"{APP_NAME}-pvc"}}],
                },
            },
        },
    }


def build_service(ns):
    return {"apiVersion": "v1", "kind": "Service", "metadata": meta(APP_NAME, ns),
            "spec": {"type": service_type(), "selector": {"app": APP_NAME},
                     "ports": [{"name": "http", "port": 80, "targetPort": APP_PORT}]}}


def build_ingress(ns, host):
    return {"apiVersion": "networking.k8s.io/v1", "kind": "Ingress",
            "metadata": {**meta(APP_NAME, ns), "annotations": {
                "nginx.ingress.kubernetes.io/proxy-read-timeout": "300",
                "nginx.ingress.kubernetes.io/proxy-body-size": "16m"}},
            "spec": {"ingressClassName": "nginx", "rules": [{"host": host, "http": {"paths": [
                {"path": "/", "pathType": "Prefix",
                 "backend": {"service": {"name": APP_NAME, "port": {"number": 80}}}}]}}]}}


def build_pvc(ns):
    return {"apiVersion": "v1", "kind": "PersistentVolumeClaim",
            "metadata": meta(f"{APP_NAME}-pvc", ns),
            "spec": {"accessModes": ["ReadWriteOnce"],
                     "resources": {"requests": {"storage": "1Gi"}}}}


def all_manifests(ns=None, host=None):
    """Built lazily, so an unfilled blank prints [TODO] instead of crashing the cell."""
    ns, host = ns or NS, host or HOST
    return [build_configmap(ns), build_pvc(ns), build_deployment(ns),
            build_service(ns), build_ingress(ns, host)]
'''

# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
CHECKS = '''
# --- Self-check: the manifest set  (pure dicts -- no cluster, no model, no network)
def by_kind(kind):
    return next(m for m in all_manifests() if m["kind"] == kind)

def container():
    return by_kind("Deployment")["spec"]["template"]["spec"]["containers"][0]

def mem_to_mi(v):
    return int(v[:-2]) * 1024 if v.endswith("Gi") else int(v[:-2])

# the four decisions
check("Recreate, because there is no room for a second pod",
      lambda: by_kind("Deployment")["spec"]["strategy"]["type"] == "Recreate",
      "a RollingUpdate starts the new pod before stopping the old one")

check("ClusterIP, because the quota allows no NodePort",
      lambda: by_kind("Service")["spec"]["type"] == "ClusterIP"
              and not any("nodePort" in json.dumps(m) for m in all_manifests()))

check("the app is pointed at the in-cluster gateway",
      lambda: by_kind("ConfigMap")["data"]["LLM_PROVIDER"] == "litellm",
      "the other three providers need internet egress this namespace does not have")

check("the gateway credential comes from your own -llm Secret",
      lambda: any(r.get("secretRef", {}).get("name", "").endswith("-llm")
                  for r in container()["envFrom"]))

# the trap
check("HOME is left exactly as the image set it",
      lambda: "HOME" not in by_kind("ConfigMap")["data"],
      "moving HOME hides the baked embedding model, and there is no egress to re-fetch it")

# credentials
check("config and credentials arrive by reference, not by value",
      lambda: not container().get("env"),
      "envFrom only -- a literal in a manifest is a credential in git")

# the pod can be told whether it is healthy
check("both probes exist and ask the port the container listens on",
      lambda: container()["livenessProbe"]["httpGet"]["port"] == APP_PORT == 8000
              and container()["readinessProbe"]["httpGet"]["port"] == APP_PORT)

check("the memory limit does not exceed the whole namespace budget",
      lambda: mem_to_mi(container()["resources"]["limits"]["memory"])
              <= mem_to_mi(QUOTA["limits.memory"]))

# the route in
check("the Service targets the container port, and the Ingress targets the Service",
      lambda: by_kind("Service")["spec"]["ports"][0]["targetPort"] == APP_PORT
              and by_kind("Ingress")["spec"]["rules"][0]["http"]["paths"][0]
                  ["backend"]["service"]["port"]["number"] == 80)

check("the Ingress claims your own host",
      lambda: by_kind("Ingress")["spec"]["rules"][0]["host"] == HOST)

# telemetry
check("spans are exported, not silently dropped",
      lambda: by_kind("ConfigMap")["data"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == TEMPO_OTLP,
      "an EMPTY endpoint disables the exporter; a WRONG one drops every span in silence")

check("your traces are findable among the whole cohort's",
      lambda: by_kind("ConfigMap")["data"]["OTEL_SERVICE_NAME"].endswith(NS),
      "one Tempo, 31 services -- the name has to carry your namespace")

check("every object lands in your namespace, and inside the quota",
      lambda: {m["metadata"]["namespace"] for m in all_manifests()} == {NS}
              and sum(1 for m in all_manifests() if m["kind"] == "Service") <= QUOTA["services"]
              and sum(1 for m in all_manifests()
                      if m["kind"] == "PersistentVolumeClaim") <= QUOTA["persistentvolumeclaims"])
'''


# --------------------------------------------------------------------------- #
# live cells
# --------------------------------------------------------------------------- #
WRITE_OUT = '''
# The manifest set, on disk, exactly as kubectl will receive it.
def write_manifests():
    doc = {"apiVersion": "v1", "kind": "List", "items": all_manifests()}
    path = os.path.join(WORK, "frontdeskai.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print("wrote", path)
    for m in doc["items"]:
        print(f"  {m['kind']:<22} {m['metadata']['name']}")
    print("\\napply it with:")
    print(f"  kubectl -n {NS} apply -f {path}")
    return path

MANIFEST_PATH = guard(write_manifests)
'''

APPLY = '''
# --- Run it for real: create the app Secret, then apply the set -------------
# SECRET_KEY is the Fernet key for encrypted per-skill config in the app's database.
# It must survive a redeploy, or previously stored skill credentials become unreadable,
# so this reuses the existing one when there is one.
import subprocess, secrets as _secrets

def kubectl(*args, **kw):
    return subprocess.run(("kubectl", "-n", APP_NS) + args,
                          capture_output=True, text=True, **kw)

def deploy():
    if not APP_NS:
        print("APP_NAMESPACE is unset in this kernel, so there is no namespace to deploy into.")
        print("In a sandbox terminal this is already exported. Skipping the cluster steps.")
        return False
    if not MANIFEST_PATH:
        print("No manifest file was written -- fill the blanks above and re-run that cell.")
        return False

    existing = kubectl("get", "secret", "frontdeskai-secret",
                       "-o", "jsonpath={.data.SECRET_KEY}")
    if existing.returncode == 0 and existing.stdout.strip():
        import base64
        key = base64.b64decode(existing.stdout).decode()
        print("reusing the existing SECRET_KEY")
    else:
        key = _secrets.token_urlsafe(32)
        print("generating a SECRET_KEY")
    made = kubectl("create", "secret", "generic", "frontdeskai-secret",
                   f"--from-literal=SECRET_KEY={key}",
                   "--from-literal=AUTH_PASSWORD=brainupgrade",
                   "--dry-run=client", "-o", "json")
    subprocess.run(("kubectl", "-n", APP_NS, "apply", "-f", "-"),
                   input=made.stdout, capture_output=True, text=True)

    out = kubectl("apply", "-f", MANIFEST_PATH)
    print(out.stdout.strip() or out.stderr.strip())
    return out.returncode == 0

APPLIED = guard(deploy) or False
'''

ROLLOUT = '''
# --- Run it for real: wait for it, then ask the readiness probe ------------
def wait_for_it():
    if not APPLIED:
        print("Nothing was applied, so there is nothing to wait for.")
        return
    print("waiting for the rollout (the first image pull is the slow part)...")
    out = kubectl("rollout", "status", "deployment/frontdeskai", "--timeout=300s")
    print(out.stdout.strip() or out.stderr.strip())

    pods = kubectl("get", "pods", "-l", "app=frontdeskai", "--no-headers")
    print(pods.stdout.strip())

    hit = kubectl("exec", f"deploy/{APP_NAME}", "--", "python3", "-c",
                  "import urllib.request as u;"
                  f"print(u.urlopen('http://127.0.0.1:{APP_PORT}/health').status)")
    print("GET /health ->", (hit.stdout.strip() or hit.stderr.strip()[:200]))
    print("\\nyour app:", "https://" + APP_HOST if APP_HOST else "(APP_HOST unset)")

guard(wait_for_it)
'''

TALK = '''
# --- Run it for real: ask the running service a question -------------------
def talk_to_it():
    if not APPLIED:
        print("The app is not deployed from this kernel, so there is nothing to ask.")
        return
    ask_py = (
        "import json,urllib.request,urllib.parse,http.cookiejar;"
        "j=http.cookiejar.CookieJar();"
        "o=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(j));"
        "p=lambda u,d: o.open('http://127.0.0.1:8000'+u,"
        " data=urllib.parse.urlencode(d).encode(), timeout=600);"
        "p('/login', {'email':'rajesh.kumar@unigps.in','password':'brainupgrade'});"
        "r=json.loads(p('/chat/send', {'message':'What is my current leave balance?'}).read());"
        "print(json.dumps({'category':r['category'],'audit':r['audit'],"
        "'response':r['response'][:400]}))"
    )
    out = kubectl("exec", f"deploy/{APP_NAME}", "--", "python3", "-c", ask_py)
    if out.returncode != 0:
        print("the request did not complete:", (out.stderr or out.stdout).strip()[:300])
        return
    try:
        answer = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        print(out.stdout.strip()[:400])
        return
    print("category:", answer["category"])
    for line in answer["audit"]:
        print("  ", line)
    print()
    print(answer["response"])

guard(talk_to_it)
'''


TRACES = '''
# --- Run it for real: read your own trace back out of Tempo ----------------
def read_my_traces():
    import urllib.request, urllib.parse
    if not APP_NS:
        print("APP_NAMESPACE is unset in this kernel, so there is no service to look up.")
        return
    svc = f"{APP_NAME}-{APP_NS}"

    def get(path):
        return json.load(urllib.request.urlopen(TEMPO_QUERY + path, timeout=30))

    try:
        q = urllib.parse.quote(f"service.name={svc}")
        found = get(f"/api/search?tags={q}&limit=5").get("traces", [])
    except Exception as exc:
        print(f"could not reach Tempo ({type(exc).__name__}). Spans are batched, so give")
        print("it about ten seconds after a request and run this cell again.")
        return

    if not found:
        print(f"Tempo has no traces for {svc} yet -- spans are batched. Wait ~10s, re-run.")
        return

    print(f"{len(found)} trace(s) for {svc}")
    complete = [t for t in found if t.get("rootTraceName")] or found
    newest = max(complete, key=lambda t: int(t.get("startTimeUnixNano", 0)))
    print(f"  {newest.get('rootTraceName')}  {newest.get('durationMs')} ms  {newest['traceID']}")
    print()

    rows = []
    for b in get(f"/api/traces/{newest['traceID']}").get("batches", []):
        for ss in b.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                ms = (int(sp["endTimeUnixNano"]) - int(sp["startTimeUnixNano"])) / 1e6
                attrs = {a["key"]: list(a["value"].values())[0]
                         for a in sp.get("attributes", [])}
                rows.append((ms, sp["name"], attrs))
    rows.sort(reverse=True)
    print("  slowest spans:")
    for ms, name, attrs in rows[:6]:
        extra = attrs.get("llm.tokens") or attrs.get("chat.category") or ""
        print(f"    {name:30} {ms:9.1f} ms  {extra}")
    print()
    print("  Which step dominates? That is what a trace answers and a latency metric")
    print(f"  does not. In Grafana: Explore -> Tempo -> service.name = {svc}")

guard(read_my_traces)
'''


# --------------------------------------------------------------------------- #
# the lab
# --------------------------------------------------------------------------- #
LAB1 = [
    header(1, "Deploy the FrontDesk Service", "Advanced", 45,
           ["Read what your namespace allows, and what it can reach",
            "Make the five choices those limits force &mdash; including the one that has "
            "already broken this app once",
            "Lint the whole manifest set offline, before anything is scheduled",
            "Apply it, wait for the rollout, and get a grounded answer out of the "
            "running service",
            "Read your own trace back out of Tempo &mdash; the only evidence telemetry works"],
           "> **Everything before this ran in a notebook.** This lab puts a multi-agent service\n"
           "> on a cluster, in your own namespace, on your own hostname, reachable from a\n"
           "> browser. You are not asked to type the manifests out: they are written for you,\n"
           "> and what you supply are the five decisions inside them."),
    setup(1),
    md(INFO_CLUSTER),
    code(FACTS),

    md(INFO_OBJECTS),
    code(DECISIONS_LAB, DECISIONS_SOL),
    code(BUILDERS),
    code(CHECKS),

    md(INFO_APPLY),
    code(WRITE_OUT),
    code(APPLY),

    md(MD_PROBE),
    code(ROLLOUT),

    md(MD_ASK),
    code(TALK),

    md(MD_TRACE),
    code(TRACES),

    code('''
score()
'''),
    md("""
## Your turn

1. **Break it deliberately.** Set `home_override()` to `{"HOME": "/shared"}`, redeploy, and read
   the crash. That is the exact failure this app hit on its first deployment here: chromadb
   resolves its cache from `Path.home()`, finds nothing, tries to download the model, and dies
   during startup indexing with a connection error &mdash; while every manifest is valid and
   every probe is configured. **A healthy manifest is not a healthy app.**
2. **Make the rollout hang.** Set `update_strategy()` to `{"type": "RollingUpdate"}` and apply.
   The new pod is quota-denied, the old one keeps serving, and `rollout status` waits until it
   times out. Nothing reports an error you would notice from the outside. Then find the message
   that does say what happened &mdash; `kubectl -n $APP_NAMESPACE describe rs` &mdash; and decide
   where in a pipeline you would surface it.
3. **Pin the image.** Replace `:latest` with the digest of the image you just deployed
   (`kubectl get pod -o jsonpath='{..imageID}'`), and say what that buys and what it costs.
4. Which of the eighteen checks above would have caught **only** a mistake, and which encode a
   fact about *this* cluster that would be wrong somewhere else? Those are two different kinds of
   rule and only one of them travels.

**What you take from Module 9:** the constraints of the environment are part of the design, the
manifest is where that design is written down, and a set of predicates over it is a review that
runs every time instead of a checklist someone remembers.

That is the last lab. The capstone puts all nine modules behind one endpoint.
"""),
]


# =========================================================================== #
# main
# =========================================================================== #
LABS = [
    ("lab-9-01-deploy-the-frontdesk-service", LAB1),
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
    print(f"\n{len(LABS)} lab, {len(LABS) * 2} notebooks written")


if __name__ == "__main__":
    main()
