"""Generates the FrontDesk AI cost-and-workflow-legs dashboard.

Edit this file and re-run it; never hand-edit the JSON it writes.
Output path is resolved relative to __file__ on purpose: this repo is public and
its directory path names the client, so an absolute path trips the rule-3 scan.
"""
import json
import os

DS = {"type": "prometheus", "uid": "prometheus-nuc"}
NS = 'namespace=~"$namespace"'
panels, _id = [], [0]
def nid(): _id[0] += 1; return _id[0]

# ── the metrics ──────────────────────────────────────────────────────────────
TOK  = f'frontdeskai_llm_tokens_total{{{NS}}}'
CNT  = f'frontdeskai_llm_call_duration_seconds_count{{{NS}}}'
SUMD = f'frontdeskai_llm_call_duration_seconds_sum{{{NS}}}'
CAT  = f'frontdeskai_category_total{{{NS}}}'
REQS = f'frontdeskai_request_duration_seconds_sum{{{NS}}}'
REQC = f'frontdeskai_request_duration_seconds_count{{{NS}}}'

# The gateway's own billing, divided by the tokens it billed for. Self-calibrating:
# no price is hardcoded here, so a tariff change on the gateway shows up by itself.
# It is a BLENDED input+output rate, because the app counts total tokens only.
USD = 'scalar(sum(litellm_spend_metric_total) / sum(litellm_total_tokens_metric_total))'
REQ = f'scalar(sum({CAT}))'

def leg(inner):
    """Collapse the agent label into the five workflow legs.

    Agent names are <desk>_react_iter_<n>[_fb] / <desk>_worker_final / supervisor /
    manager. label_replace anchors its regex, so _fb cannot also match the plain
    react rule. The numeric prefix is what orders the legend by pipeline position.
    """
    r = inner
    for lbl, rx in (
        ("1 supervisor", "supervisor"),
        ("5 manager", "manager"),
        ("2 react loop", ".*_react_iter_[0-9]+"),
        ("3 tool-error fallback", ".*_react_iter_[0-9]+_fb"),
        ("4 final answer", ".*_worker_final"),
    ):
        r = f'label_replace({r}, "leg", "{lbl}", "agent", "{rx}")'
    return r

def desk(inner):
    return f'label_replace({inner}, "desk", "$1", "agent", "([a-z]+)_(?:react_iter|worker).*")'

def leg_agg(metric):
    """sum-by-leg of any per-agent metric expression."""
    return f'sum by (leg) ({leg(f"sum by (agent) ({metric})")})'

LEG_TOK  = leg_agg(TOK)
LEG_TIME = leg_agg(SUMD)
LEG_CALL = leg_agg(CNT)
LEG_USD  = f'{LEG_TOK} * {USD}'
LEGS = ["1 supervisor", "2 react loop", "3 tool-error fallback", "4 final answer", "5 manager"]

# ── panel helpers ────────────────────────────────────────────────────────────
def row(title, y):
    panels.append({"type": "row", "title": title, "collapsed": False, "id": nid(),
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []})

def tgt(expr, refid="A", legend=None, table=False, instant=False):
    t = {"expr": expr, "refId": refid, "datasource": DS}
    if legend is not None: t["legendFormat"] = legend
    if table: t["format"] = "table"
    if instant: t["instant"] = True
    return t

def stat(title, expr, x, y, w=4, h=4, unit="short", decimals=2, steps=None, desc=None):
    return {"type": "stat", "title": title, "id": nid(), "datasource": DS, "description": desc,
            "gridPos": {"h": h, "w": w, "x": x, "y": y}, "targets": [tgt(expr)],
            "fieldConfig": {"defaults": {
                "unit": unit, "decimals": decimals, "mappings": [],
                "color": {"mode": "thresholds"} if steps else {"mode": "fixed", "fixedColor": "text"},
                "thresholds": {"mode": "absolute", "steps": steps or [{"color": "text", "value": None}]}},
                "overrides": []},
            "options": {"colorMode": "background" if steps else "value", "graphMode": "none",
                        "justifyMode": "center", "textMode": "auto",
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}

def ts(title, targets, x, y, w, h, unit="short", decimals=None, desc=None, stack=False):
    d = {"unit": unit, "custom": {"fillOpacity": 40 if stack else 10, "lineWidth": 1,
                                  "showPoints": "never",
                                  **({"stacking": {"mode": "normal"}} if stack else {})}}
    if decimals is not None: d["decimals"] = decimals
    return {"type": "timeseries", "title": title, "id": nid(), "datasource": DS, "description": desc,
            "gridPos": {"h": h, "w": w, "x": x, "y": y}, "targets": targets,
            "fieldConfig": {"defaults": d, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}

def bargauge(title, expr, x, y, w, h, unit="short", decimals=2, desc=None, legend="{{leg}}"):
    return {"type": "bargauge", "title": title, "id": nid(), "datasource": DS, "description": desc,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": [tgt(expr, "A", legend=legend, instant=True)],
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                         "color": {"mode": "continuous-BlPu"},
                                         "thresholds": {"mode": "absolute",
                                                        "steps": [{"color": "text", "value": None}]}},
                            "overrides": []},
            "options": {"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
                        "valueMode": "color", "minVizWidth": 8, "sortBy": "Name", "sortOrder": "Ascending",
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}

def barchart(title, expr, x, y, w, h, legend, unit="short", decimals=2, desc=None):
    return {"type": "barchart", "title": title, "id": nid(), "datasource": DS, "description": desc,
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "targets": [tgt(f'sort_desc({expr})', "A", legend=legend, instant=True)],
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                         "custom": {"fillOpacity": 80, "lineWidth": 0}},
                            "overrides": []},
            "options": {"orientation": "horizontal", "showValue": "always", "xTickLabelRotation": 0,
                        "legend": {"showLegend": False}}}

def text(title, md, x, y, w, h):
    return {"type": "text", "title": title, "id": nid(),
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "options": {"mode": "markdown", "content": md}}

# ── 1. what one request costs ────────────────────────────────────────────────
row("What one request costs", 0)
panels += [
    stat("Cost per request", f'sum({TOK}) * {USD} / sum({CAT})', 0, 1, unit="currencyUSD", decimals=6,
         desc="Total tokens x the gateway's own blended rate, over requests handled. This is the number Module 7 says to put a ceiling on."),
    stat("Cost per 1,000 requests", f'sum({TOK}) * {USD} / sum({CAT}) * 1000', 4, 1,
         unit="currencyUSD", decimals=3,
         desc="The same number at a scale you can argue about in a planning meeting. Six decimal places is not a budget."),
    stat("Tokens per request", f'sum({TOK}) / sum({CAT})', 8, 1, decimals=0),
    stat("Seconds per request", f'sum({REQS}) / sum({REQC})', 12, 1, unit="s",
         desc="Mean end-to-end. Time and cost are different bills paid by different legs — that is what the rest of this dashboard is for."),
    stat("LLM calls per request", f'sum({CNT}) / sum({CAT})', 16, 1,
         desc="Every leg of the workflow is a billable call. Fan-out is the cost lever you control most directly."),
    stat("Gateway rate", f'{USD} * 1e6', 20, 1, unit="currencyUSD", decimals=4,
         desc="USD per MILLION tokens, derived live from the gateway's own spend divided by its own token count — nothing is hardcoded here. Blended across input and output, because the app records total tokens only."),
]

# ── 2. the workflow, pictorially ─────────────────────────────────────────────
row("The workflow, leg by leg", 5)

# One instant table query, then rowsToFields turns each leg into a NAMED FIELD --
# which is how a canvas element binds a value: config.text.field matches by name.
canvas_targets = [tgt(f'sum by (leg) (({LEG_USD}) / {REQ})', "A", table=True, instant=True)]

def canvas_box(name, field, left, top, w=176, h=86, color="#1e1b4b", border="#818cf8", to=None):
    el = {"type": "rectangle", "name": name,
          "config": {"align": "center", "valign": "middle",
                     "color": {"fixed": "#f8fafc"}, "size": 18,
                     "text": {"mode": "field", "field": field, "fixed": ""}},
          "background": {"color": {"fixed": color}},
          "border": {"color": {"fixed": border}, "width": 2},
          "constraint": {"horizontal": "left", "vertical": "top"},
          "placement": {"top": top, "left": left, "width": w, "height": h, "rotation": 0}}
    if to:
        el["connections"] = [{"source": {"x": 1, "y": 0}, "target": {"x": -1, "y": 0},
                              "targetName": to, "color": {"fixed": "#818cf8"},
                              "size": {"fixed": 2}, "path": "straight"}]
    return el

def canvas_label(name, txt, left, top, w=176, h=24, size=12, color="#a5b4fc"):
    return {"type": "text", "name": name,
            "config": {"align": "center", "valign": "middle", "color": {"fixed": color},
                       "size": size, "text": {"mode": "fixed", "fixed": txt}},
            "background": {"color": {"fixed": "transparent"}},
            "border": {"color": {"fixed": "transparent"}, "width": 0},
            "constraint": {"horizontal": "left", "vertical": "top"},
            "placement": {"top": top, "left": left, "width": w, "height": h, "rotation": 0}}

elements = []
for i, lg in enumerate(LEGS):
    left = 16 + i * 196
    nm = lg[2:]
    nxt = LEGS[i + 1][2:] if i + 1 < len(LEGS) else None
    elements.append(canvas_label(f"lbl-{nm}", nm, left, 26))
    elements.append(canvas_box(nm, lg, left, 54, to=nxt))
    elements.append(canvas_label(f"cap-{nm}", "USD per request", left, 146, size=11, color="#64748b"))

panels.append({
    "type": "canvas", "title": "Cost of each leg, per request", "id": nid(), "datasource": DS,
    "description": "The pipeline as it actually runs: the supervisor classifies, the worker's ReAct loop reads tools, a failed tool call falls back, the worker writes the answer, and the manager takes it if it escalates. Each box shows USD per request spent in that leg. A leg with no traffic reads 0.",
    "gridPos": {"h": 9, "w": 24, "x": 0, "y": 6},
    "targets": canvas_targets,
    "transformations": [
        {"id": "rowsToFields", "options": {"mappings": [
            {"fieldName": "leg", "handlerKey": "field.name"},
            {"fieldName": "Value", "handlerKey": "field.value"}]}},
    ],
    "fieldConfig": {"defaults": {"unit": "currencyUSD", "decimals": 6}, "overrides": []},
    "options": {"inlineEditing": False, "showAdvancedTypes": True, "panZoom": False,
                "infinitePan": False,
                "root": {"type": "frame", "name": "Element 1", "elements": elements,
                         "background": {"color": {"fixed": "transparent"}},
                         "border": {"color": {"fixed": "transparent"}, "width": 0},
                         "constraint": {"horizontal": "left", "vertical": "top"},
                         "placement": {"top": 0, "left": 0, "width": 100, "height": 100}}},
})

# ── 3. where the money and the time go ───────────────────────────────────────
row("Where the money goes, and where the time goes", 15)
panels += [
    bargauge("Cost by leg (USD)", LEG_USD, 0, 16, 8, 9, unit="currencyUSD", decimals=6,
             desc="Cumulative since the pod started."),
    bargauge("Time by leg (seconds)", LEG_TIME, 8, 16, 8, 9, unit="s", decimals=1,
             desc="The same five legs by wall clock. Compare with the bar to the left: the leg that costs most and the leg that takes longest are rarely the same, and caching one does nothing for the other."),
    {"type": "piechart", "title": "Share of spend by leg", "id": nid(), "datasource": DS,
     "description": "If one leg owns most of this pie, that is where a prompt or a model change pays for itself.",
     "gridPos": {"h": 9, "w": 8, "x": 16, "y": 16},
     "targets": [tgt(LEG_USD, "A", legend="{{leg}}", instant=True)],
     "fieldConfig": {"defaults": {"unit": "currencyUSD", "decimals": 6}, "overrides": []},
     "options": {"legend": {"displayMode": "table", "placement": "right", "showLegend": True,
                            "values": ["value", "percent"]},
                 "pieType": "donut",
                 "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}},
    {"type": "table", "title": "Leg scorecard", "id": nid(), "datasource": DS,
     "description": "One row per workflow leg. Cost per request is the column a gate should watch; calls per request tells you whether the leg is expensive because it is slow or because it runs too often.",
     "gridPos": {"h": 10, "w": 24, "x": 0, "y": 25},
     "targets": [
         tgt(LEG_CALL, "A", table=True, instant=True),
         tgt(f'{LEG_CALL} / {REQ}', "B", table=True, instant=True),
         tgt(LEG_TIME, "C", table=True, instant=True),
         tgt(LEG_TOK, "D", table=True, instant=True),
         tgt(LEG_USD, "E", table=True, instant=True),
         tgt(f'({LEG_USD}) / {REQ}', "F", table=True, instant=True),
         tgt(f'({LEG_USD}) / scalar(sum({LEG_USD}))', "G", table=True, instant=True),
     ],
     "transformations": [
         {"id": "joinByField", "options": {"byField": "leg", "mode": "outer"}},
         {"id": "organize", "options": {
             "excludeByName": {f"Time{s}": True for s in ("", " 1", " 2", " 3", " 4", " 5", " 6", " 7")},
             "renameByName": {"leg": "Leg", "Value #A": "Calls", "Value #B": "Calls per request",
                              "Value #C": "Seconds", "Value #D": "Tokens", "Value #E": "Cost",
                              "Value #F": "Cost per request", "Value #G": "Share of spend"}}},
         {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "Leg", "desc": False}]}},
     ],
     "fieldConfig": {"defaults": {"custom": {"align": "auto", "inspect": False}}, "overrides": [
         {"matcher": {"id": "byRegexp", "options": "Cost.*"},
          "properties": [{"id": "unit", "value": "currencyUSD"}, {"id": "decimals", "value": 6}]},
         {"matcher": {"id": "byName", "options": "Seconds"},
          "properties": [{"id": "unit", "value": "s"}, {"id": "decimals", "value": 1}]},
         {"matcher": {"id": "byName", "options": "Calls per request"},
          "properties": [{"id": "decimals", "value": 2}]},
         {"matcher": {"id": "byName", "options": "Share of spend"},
          "properties": [{"id": "unit", "value": "percentunit"}, {"id": "decimals", "value": 1},
                         {"id": "custom.cellOptions",
                          "value": {"type": "color-background", "mode": "gradient"}},
                         {"id": "thresholds", "value": {"mode": "absolute", "steps": [
                             {"color": "green", "value": None}, {"color": "orange", "value": 0.35},
                             {"color": "red", "value": 0.6}]}}]},
     ]},
     "options": {"showHeader": True, "footer": {"show": False}}},
]

# ── 4. which desk ────────────────────────────────────────────────────────────
row("Which specialist desk the money goes to", 35)
# supervisor and manager belong to no desk, so they are excluded here rather than
# collapsing into a blank bar -- this panel answers "which desk", not "which leg".
TOK_WORKERS = f'frontdeskai_llm_tokens_total{{{NS}, agent!~"supervisor|manager"}}'
DESK_USD = f'sum by (desk) ({desk(f"sum by (agent) ({TOK_WORKERS})")}) * {USD}'
panels += [
    barchart("Spend by desk (USD)", DESK_USD, 0, 36, 12, 9, "{{desk}}",
             unit="currencyUSD", decimals=6,
             desc="Worker spend only. The supervisor and the manager belong to no desk and are excluded, so this total is LESS than the app total — the routing and escalation legs are on the leg breakdown above."),
    barchart("Requests routed to each desk", f'sum by (category) ({CAT})', 12, 36, 12, 9,
             "{{category}}", decimals=0,
             desc="Read together with the bar on the left. A desk taking few requests but most of the spend is the one to look at first."),
]

# ── 5. over time, and across participants ────────────────────────────────────
row("Over time, and across participants", 45)
panels += [
    ts("Spend rate by leg", [tgt(f'{leg_agg(f"rate({TOK}[$__rate_interval])")} * {USD} * 3600',
                                 "A", legend="{{leg}}")],
       0, 46, 12, 8, unit="currencyUSD", decimals=4, stack=True,
       desc="USD per hour at the current rate, stacked by leg. Useful for seeing a change land, not for forecasting a bill."),
    ts("Cost per request over time",
       [tgt(f'sum(rate({TOK}[$__rate_interval])) * {USD} / sum(rate({CAT}[$__rate_interval]))',
            "A", legend="cost per request")],
       12, 46, 12, 8, unit="currencyUSD", decimals=6,
       desc="The number to watch after a prompt change. Cost per case moves quietly when a prompt grows."),
    barchart("Cost per request, by participant",
             f'sum by (namespace) ({TOK}) * {USD} / sum by (namespace) ({CAT})',
             0, 54, 12, 9, "{{namespace}}", unit="currencyUSD", decimals=6,
             desc="Select All in the Participant variable to compare. Everyone is running the same app against the same model, so a large spread is a difference in prompts, tools or traffic — not in the platform."),
    barchart("Total app spend, by participant",
             f'sum by (namespace) ({TOK}) * {USD}', 12, 54, 12, 9, "{{namespace}}",
             unit="currencyUSD", decimals=6,
             desc="Attributed from the app's own token counter, so this is the deployed app only."),
]

# ── 6. cross-check against the gateway's own billing ─────────────────────────
row("Cross-check, and the budget", 63)
KEYSPEND = ('sum by (namespace) (label_replace(sum by (api_key_alias) (litellm_spend_metric_total), '
            '"namespace", "agenticaiu$1", "api_key_alias", "agenticai-[0-9]+-u([0-9]+)"))')
panels += [
    stat("App spend (derived)", f'sum({TOK}) * {USD}', 0, 64, w=6, unit="currencyUSD", decimals=5,
         desc="From the app's own per-agent token counter. Resets when the pod restarts."),
    stat("Key spend (gateway)", f'sum({KEYSPEND} and on(namespace) ({CAT} > bool -1))', 6, 64, w=6,
         unit="currencyUSD", decimals=5,
         desc="What the gateway has billed this participant's key. It includes their NOTEBOOK work as well as the app, so it should be LARGER than the derived figure — if it is smaller, the derivation is wrong."),
    stat("Requests handled", f'sum({CAT}) or vector(0)', 12, 64, w=6, decimals=0),
    stat("Share of the $2.50 daily cap", f'sum({KEYSPEND}) / 2.5', 18, 64, w=6,
         unit="percentunit", decimals=1,
         steps=[{"color": "green", "value": None}, {"color": "orange", "value": 0.6},
                {"color": "red", "value": 0.9}],
         desc="Each participant key is capped at $2.50/day. Over budget the gateway returns HTTP 400 rather than falling back, so this is worth watching during a long session."),
    text("How the cost number is built", """
**There is no price list in this dashboard.** The rate comes from the gateway's own accounting —
`sum(litellm_spend_metric_total) / sum(litellm_total_tokens_metric_total)` — so it re-derives itself
if the tariff changes. Per-leg cost is then that rate times the tokens the app attributes to each
agent. Three consequences worth knowing before you quote a number from here:

- **It is a blended input/output rate.** The app records total tokens per call, not the split, so a
  leg with an unusual input:output ratio is approximated. The *ranking* of legs is solid; the
  absolute figure is an estimate.
- **"Per request" means per request on average.** Prometheus stores aggregates. For one specific
  request — which agent, in what order, how long each took — the span tree is in Grafana's Explore
  under the **Tempo** datasource, service name `frontdeskai-<your namespace>`.
- **Counters reset when the pod restarts**, so a redeploy is a clean experiment boundary: change one
  thing, replay the same cases, compare cost per request before and after.

**Module 7's point, in one line:** cost per case is a gate, not a report. Pick the ceiling while
nobody is under pressure, and fail the build on it — and remember the leg that costs the most and
the leg that takes the longest are usually different legs, so know which one you were asked to fix.
""".strip(), 0, 68, 24, 9),
]

dash = {
    "title": "FrontDesk AI — Cost & Workflow Legs",
    "uid": "frontdeskai-cost-workflow",
    "description": "Time and cost per request, broken down by workflow leg and specialist desk, per participant namespace. The cost rate is derived live from the gateway's own billing rather than hardcoded. Queries the NUC Prometheus (uid prometheus-nuc).",
    "tags": ["agenticai", "workshop", "frontdeskai", "cost", "evaluation"],
    "timezone": "Asia/Kolkata",
    "schemaVersion": 39,
    "editable": True,
    "graphTooltip": 1,
    "refresh": "30s",
    "time": {"from": "now-3h", "to": "now"},
    "templating": {"list": [{
        "name": "namespace", "label": "Participant", "type": "query", "datasource": DS,
        "query": {"query": 'label_values(up{job="frontdeskai-participants"}, namespace)', "refId": "var"},
        "definition": 'label_values(up{job="frontdeskai-participants"}, namespace)',
        "refresh": 2, "sort": 7, "multi": True, "includeAll": True, "allValue": ".*",
        "current": {"text": ["All"], "value": ["$__all"]}, "options": [],
    }]},
    "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "datasource", "uid": "grafana"},
                              "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)",
                              "name": "Annotations & Alerts", "type": "dashboard"}]},
    "panels": panels,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "frontdeskai-cost-workflow-dashboard.json")
json.dump(dash, open(out, "w"), indent=2)
open(out, "a").write("\n")
print("panels:", len(panels), "->", out)
