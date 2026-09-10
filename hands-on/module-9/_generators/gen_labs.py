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
FACTS = '''
# ------------------------------------------------------- what you are deploying
# FrontDesk AI: a FastAPI service wrapping a LangGraph multi-agent support desk.
# Supervisor -> RAG -> a domain worker with tools -> escalation -> QA gate.
# Published image, nothing to build here.
APP_IMAGE = "brainupgrade/frontdeskai:latest"
APP_PORT  = 8000            # what uvicorn listens on inside the container
APP_NAME  = "frontdeskai"   # every object in this lab is named after it

# A real deployment pins a digest instead of a moving tag, so that two rollouts
# of "the same" version really are. `latest` is used here because the workshop
# rebuilds the image during the course.

# ------------------------------------------------------- what your namespace allows
# Read from your own ResourceQuota with:  kubectl -n $APP_NAMESPACE describe quota
# These are the numbers, not a simplification -- three of them drive a decision below.
QUOTA = {
    "pods":                   8,
    "requests.cpu":       "500m",
    "requests.memory":   "512Mi",
    "limits.cpu":            "1",
    "limits.memory":       "1Gi",     # <- the whole namespace, not per pod
    "services":               3,
    "services.nodeports":     0,      # <- zero, not "a few"
    "persistentvolumeclaims": 2,
    "requests.storage":    "5Gi",
}

# ------------------------------------------------------- what your namespace can reach
# NetworkPolicy `participant-egress`. This is NOT the sandbox namespace, which does
# have general internet egress -- yours does not.
EGRESS_ALLOWED = [
    "kube-system :53/UDP        (DNS)",
    "llm-serving  any port      (the LiteLLM gateway -- how the app reaches a model)",
    "ingress-nginx pods         (so your Ingress can reach back in)",
    "langfuse     :3000         (tracing)",
]

# ------------------------------------------------------- secrets already in your namespace
# Published for you. You do not create these and you never copy their values.
SECRETS = {
    "llm":       "{ns}-llm",             # gateway base URL, model name, your own capped key
    "langfuse":  "{ns}-langfuse",        # tracing keys, scoped to your environment tag
    "app":       "frontdeskai-secret",   # SECRET_KEY + first-login password, created at deploy
}

# ------------------------------------------------------- where telemetry goes
# Tempo lives in the `monitoring` namespace and is shared by the whole cohort.
# Your namespace is allowed to reach 4317 (send spans); your SANDBOX is allowed to
# reach 3200 (read them back), which is what the last cell of this lab uses.
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
# Four decisions and one trap. Each is forced by something printed above -- the
# quota, the network policy, or the image. None is a Python puzzle: the mechanics
# are given, the answer is the choice.

def update_strategy() -> dict:
    """Rollout strategy for one replica whose memory LIMIT is 1Gi.

    limits.memory for the WHOLE namespace is also 1Gi. A RollingUpdate starts the
    replacement pod before stopping the old one, so for a moment two pods each
    want 1Gi. What happens to the second one, and to your rollout?
    """
    return {"type": BLANK}                 # "RollingUpdate" | "Recreate"


def service_type() -> str:
    """How the Service is exposed.

    services.nodeports is 0 in your quota, and services.loadbalancers is 0 too.
    An Ingress reaches a Service from inside the cluster.
    """
    return BLANK                           # "ClusterIP" | "NodePort" | "LoadBalancer"


def llm_provider() -> str:
    """Which provider the app uses for its LLM calls.

    agents.py knows four. "groq", "ollama" and "openrouter" each reach a vendor
    over the internet. "litellm" points at an OpenAI-compatible gateway whose base
    URL and key it reads from the environment. Re-read EGRESS_ALLOWED before you
    answer -- this is the decision that makes the app work here at all.
    """
    return BLANK                           # "groq" | "ollama" | "openrouter" | "litellm"


def gateway_secret(ns: str) -> str:
    """Which existing Secret carries that credential, mounted with envFrom.

    SECRETS is printed above. Return the KEY into it; the formatting is done here.
    """
    return SECRETS[BLANK].format(ns=ns)    # "llm" | "langfuse" | "app"


def home_override() -> dict:
    """Extra env for HOME, if any. This one has bitten before.

    The image bakes ChromaDB's embedding model into /opt/appcache and points HOME
    at it, because chromadb resolves its cache from Path.home() and downloads the
    model on first use -- which your namespace has no egress to do. The app also
    has a PVC mounted at /shared for its database, which is a tempting home.
    """
    return BLANK                           # {} | {"HOME": "/shared"}
'''

DECISIONS_SOL = (DECISIONS_LAB
    .replace('{"type": BLANK}                 #', '{"type": "Recreate"}            #')
    .replace('return BLANK                           # "ClusterIP"',
             'return "ClusterIP"                     # "ClusterIP"')
    .replace('return BLANK                           # "groq"',
             'return "litellm"                       # "groq"')
    .replace('SECRETS[BLANK]', 'SECRETS["llm"]')
    .replace('return BLANK                           # {}',
             'return {}                              # {}'))


# --------------------------------------------------------------------------- #
# the five objects
# --------------------------------------------------------------------------- #
BUILDERS = '''
# The five objects, given. Read them: this is the same YAML you have seen, written
# as dicts -- a Kubernetes manifest IS JSON, and YAML is a surface syntax over it.
# kubectl takes either, so the thing you lint below is the thing you apply.

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
        # One Tempo serves all 31 of you, so the service name has to carry your
        # namespace or you cannot find your own traces among everyone else's.
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
                        # Config and credentials arrive whole, by reference. No value
                        # of a secret is ever written into a manifest.
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
SECRETISH = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

def by_kind(kind):
    return next(m for m in all_manifests() if m["kind"] == kind)

def container():
    return by_kind("Deployment")["spec"]["template"]["spec"]["containers"][0]

def mem_to_mi(v):
    return int(v[:-2]) * 1024 if v.endswith("Gi") else int(v[:-2])

check("the rollout fits the namespace's 1Gi limits budget",
      lambda: by_kind("Deployment")["spec"]["strategy"]["type"] == "Recreate",
      "a RollingUpdate needs a second pod, and there is no room for one")

check("the Service is a type the quota permits",
      lambda: by_kind("Service")["spec"]["type"] == "ClusterIP",
      "services.nodeports is 0")

check("no object anywhere asks for a NodePort",
      lambda: not any("nodePort" in json.dumps(m) for m in all_manifests()))

check("the app is pointed at the in-cluster gateway",
      lambda: by_kind("ConfigMap")["data"]["LLM_PROVIDER"] == "litellm",
      "the other three providers need internet egress this namespace does not have")

check("the fallback is disabled rather than left pointing at a vendor",
      lambda: by_kind("ConfigMap")["data"]["LLM_FALLBACK_MODEL"] == "")

check("HOME is left as the image set it",
      lambda: "HOME" not in by_kind("ConfigMap")["data"],
      "moving HOME hides the baked embedding model, and there is no egress to re-fetch it")

check("the gateway credential comes from the per-participant Secret",
      lambda: any(r.get("secretRef", {}).get("name", "").endswith("-llm")
                  for r in container()["envFrom"]))

check("config and credentials arrive by reference, not by value",
      lambda: not container().get("env"),
      "envFrom only -- a literal in a manifest is a credential in git")

check("no manifest contains a secret-looking literal",
      lambda: not any(s in k.upper() and isinstance(v, str) and v
                      for m in all_manifests() if m["kind"] == "ConfigMap"
                      for k, v in m["data"].items() for s in SECRETISH))

check("both probes exist",
      lambda: container()["livenessProbe"] and container()["readinessProbe"])

check("the probes ask the port the container actually listens on",
      lambda: container()["livenessProbe"]["httpGet"]["port"] == APP_PORT == 8000)

check("requests and limits are both stated",
      lambda: container()["resources"]["requests"] and container()["resources"]["limits"])

check("the memory limit does not exceed the whole namespace budget",
      lambda: mem_to_mi(container()["resources"]["limits"]["memory"])
              <= mem_to_mi(QUOTA["limits.memory"]))

check("the Service targets the container port, not the Service port",
      lambda: by_kind("Service")["spec"]["ports"][0]["targetPort"] == APP_PORT)

check("the Ingress backend names the Service and its port 80",
      lambda: (lambda b: b["service"]["name"] == APP_NAME and b["service"]["port"]["number"] == 80)(
          by_kind("Ingress")["spec"]["rules"][0]["http"]["paths"][0]["backend"]))

check("the Ingress claims your own host",
      lambda: by_kind("Ingress")["spec"]["rules"][0]["host"] == HOST)

check("spans are exported, not silently dropped",
      lambda: by_kind("ConfigMap")["data"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == TEMPO_OTLP,
      "an EMPTY endpoint disables the exporter; a WRONG one drops every span in silence")

check("your traces are findable among the whole cohort's",
      lambda: by_kind("ConfigMap")["data"]["OTEL_SERVICE_NAME"].endswith(NS),
      "one Tempo, 31 services -- the name has to carry your namespace")

check("every object lands in your namespace, read from APP_NAMESPACE",
      lambda: {m["metadata"]["namespace"] for m in all_manifests()} == {NS})

check("the object count stays inside the quota",
      lambda: sum(1 for m in all_manifests() if m["kind"] == "Service") <= QUOTA["services"]
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

    # Ask the readiness probe's own question -- from INSIDE the running container.
    #
    # The obvious way to do this is `kubectl run` a curl pod. Try it: it is refused
    # with `exceeded quota: limits.memory=1Gi, used: 1Gi`, because your app is already
    # holding the entire namespace budget. A debugging pod is a pod. `exec` borrows
    # the container you already paid for, and the app image ships Python.
    hit = kubectl("exec", f"deploy/{APP_NAME}", "--", "python3", "-c",
                  "import urllib.request as u;"
                  f"print(u.urlopen('http://127.0.0.1:{APP_PORT}/health').status)")
    print("GET /health ->", (hit.stdout.strip() or hit.stderr.strip()[:200]))
    print("\\nyour app:", "https://" + APP_HOST if APP_HOST else "(APP_HOST unset)")

guard(wait_for_it)
'''

TALK = '''
# --- Run it for real: ask the running service a question -------------------
# The whole course in one request: HTTP -> supervisor -> RAG -> a domain worker
# with tools -> escalation check -> QA gate -> a grounded answer.
#
# This asks the pod directly rather than going through your public hostname, for
# two reasons worth knowing. Cloudflare sits in front of that host and gives up
# at ~100s with a 524 -- an agent doing three tool-calling turns can exceed that
# when the gateway is busy, and the 524 looks like your app is broken when it is
# still working. It also answers the default Python user agent with 403 and
# `error code: 1010`. Open the URL in a BROWSER to see the real thing.
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
# The request above emitted spans. This asks Tempo what it actually received --
# which is the only way to know your telemetry works. "The exporter is
# configured" is not evidence: BatchSpanProcessor swallows export failures, so a
# misconfigured endpoint looks exactly like a healthy one from inside the app.
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
    # Prefer a trace whose ROOT span has already arrived. Spans are batched, so the
    # very newest trace is often still partial -- its root lands last.
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
           ["Read your namespace and host from the environment, never from a hostname",
            "Build the five objects a deployment needs, as dicts you can lint offline",
            "Make the four decisions this cluster forces &mdash; and avoid the trap that has "
            "already broken this app once",
            "Apply them, wait for the rollout, and get a grounded answer out of the "
            "running service",
            "Read your own trace back out of Tempo &mdash; the only evidence that telemetry "
            "works"],
           "> **Everything before this ran in a notebook.** This lab puts a multi-agent service\n"
           "> on a cluster, in your own namespace, on your own hostname, reachable from a\n"
           "> browser. The manifests are Python dicts and the self-checks are predicates over\n"
           "> them, so the whole design is graded before anything is applied."),
    setup(1),
    code(FACTS),

    md("""
## Concept

You have built agents for three days. Shipping one is a different job, and most of it is
decided before `kubectl` is involved.

A namespace is not an unlimited machine. Yours grants **1Gi of memory limits in total**, **zero
NodePorts**, three Services and two PVCs, and its NetworkPolicy reaches **DNS, the model gateway,
the ingress controller and the tracing backend &mdash; and nothing else on the internet**. Every
one of those numbers removes an option that would otherwise look reasonable, and two of them
remove the option that is the *default*.

So the manifest is where the design lives. Written as dicts it is also where the design can be
**checked** &mdash; every rule below is a predicate over an object, runs offline in
milliseconds, and fails before a pod is ever scheduled.
"""),

    md("""
## Section 1 &mdash; The manifest set

Four decisions and one trap. Read `FACTS` above before you answer any of them: each is forced by
a number or a policy printed there, and none of them is a Python puzzle.
"""),
    code(DECISIONS_LAB, DECISIONS_SOL),
    code(BUILDERS),
    code(CHECKS),

    md("""
## Section 2 &mdash; Put it on the cluster

The dicts you just linted are what `kubectl` receives. These cells are marked **Run it for
real**: they need your namespace, and they print what to do instead if it is not set.
"""),
    code(WRITE_OUT),
    code(APPLY),
    code(ROLLOUT),
    code(TALK),
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
