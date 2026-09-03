# Runtime prerequisites — Module 1 labs

The notebooks run on the **K3s cluster, namespace `agenticai`**. The cluster admin provides the
runtime; this file is the contract the notebooks expect. Nothing here is hardcoded in a notebook —
the endpoint is read from the environment, so the admin can point it anywhere without editing labs.

## 1. Python packages

| Package | Needed by | Notes |
|---|---|---|
| *(stdlib only)* | every **graded** cell | `os`, `json`, `time`, `textwrap`, `typing` — nothing else |
| `langchain` (1.x) | Lab 1.3 live cell | `langchain.agents.create_agent`, `langchain.tools.tool` |
| `langchain-openai` | Labs 1.1–1.5 live cells | `ChatOpenAI`, pointed at the in-cluster gateway |

Python **3.12** (the course stack). Every graded cell is stdlib-only and self-checks offline, so a
participant whose LLM access is not yet wired can still complete and score all five labs.

## 2. Environment variables

Set these in the notebook pod (JupyterHub `singleuser.extraEnv`, a ConfigMap, or the image):

```
LAB_LLM_BASE_URL   # OpenAI-compatible base URL, ending in /v1
LAB_LLM_MODEL      # model name the gateway serves
OPENAI_API_KEY     # any non-empty value if the in-cluster gateway does not authenticate
```

`OPENAI_BASE_URL` / `OPENAI_MODEL` are accepted as fallbacks, so if the image already exports the
standard OpenAI variables nothing further is needed.

An in-cluster gateway is reachable at the usual service DNS, for example:

```
LAB_LLM_BASE_URL=http://<service>.agenticai.svc.cluster.local:<port>/v1
```

**These values are deliberately not baked into the notebooks.** If they are unset, every live cell
prints the two `export` lines it needs and continues — it never raises.

## 3. Egress

None required. The labs reach only the in-cluster LLM service; the case file is synthetic and
inline, and no lab downloads anything.

## 4. Writable path

Each lab creates `/tmp/awmas-lab-1-0N/` for scratch output. `/tmp` inside the pod is enough — no
persistent volume is needed for Module 1.

## 5. Smoke test

From a notebook terminal in the `agenticai` namespace:

```bash
python3 -c "
import os
from langchain_openai import ChatOpenAI
m = ChatOpenAI(model=os.environ['LAB_LLM_MODEL'],
               base_url=os.environ['LAB_LLM_BASE_URL'],
               api_key=os.environ.get('OPENAI_API_KEY', 'sandbox'))
print(m.invoke('Reply with just: OK').content)
"
```

`OK` means the live cells will work. If it fails, the graded cells still do.

## 6. Verifying the labs themselves

```bash
_generators/regenerate.sh
```

Rebuilds all ten notebooks from the single source and checks both directions: every solution must
score full marks, and every untouched lab must survive *Run All* without an uncaught exception.
Runs offline — no cluster, no model, no network (it does start a local Jupyter kernel).

`verify_labs.py` executes each lab **twice — plain `exec()` and a real Jupyter kernel — and requires
the two to agree** on (todo, pass, fail). Agreement is the actual check. A notebook must not depend
on IPython-specific semantics, and the reason is concrete: IPython predefines `_`, `__` and `___` as
its output history, initialised to `""`. A blank spelled with underscores is therefore a *defined
empty string* in a notebook and raises no `NameError`, so `[TODO]` silently becomes `[FAIL]` and a
blank used as a loop guard never stops its loop. That is why the blank marker is **`BLANK`**, which
is undefined under both. Do not change it back.
