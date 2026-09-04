#!/usr/bin/env python3
"""The ledger, exposed over MCP. Stdio transport, JSON-RPC 2.0, Content-Length framing.

This is Lab 4.4's server with one change that matters: it is a **persistent loop**. The
lab's version read all of stdin, answered everything, and exited, which is right for one
notebook round-trip and useless for a service that answers a request every few seconds.
A long-lived server reads one framed message, replies, and waits for the next.

Run it directly to see it refuse to do anything interesting:

    python3 ledger_mcp.py           # then paste a framed initialize, or use mcp_client.py

Nothing here calls a model. It is a data source with a schema, and the schema is the
contract -- Module 4's whole point.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain import LEDGER, POLICY, WATCHLIST_NOTE       # noqa: E402

SPECS = [
    {"name": "lookup_payment",
     "description": ("Return the ledger record for one payment reference such as 'PMT-1002': "
                     "amount, currency, counterparty, status, value date and reason code. "
                     "Use when you need the facts about one specific payment. Not for "
                     "searching across payments."),
     "inputSchema": {"type": "object",
                     "properties": {"ref": {"type": "string",
                                            "description": "a payment reference, e.g. PMT-1002"}},
                     "required": ["ref"]}},
    {"name": "policy_for",
     "description": ("Return the operating policy for one failure reason code, e.g. "
                     "'LIMIT_BREACH'. Use after you know why a payment failed and need to "
                     "know what to do about it."),
     "inputSchema": {"type": "object",
                     "properties": {"reason_code": {"type": "string"}},
                     "required": ["reason_code"]}},
    {"name": "screen_counterparty",
     "description": ("Return the sanctions-screening status of a counterparty name. Use for "
                     "EVERY payment, not only the ones whose reason code mentions sanctions "
                     "-- screening is independent of why the payment failed."),
     "inputSchema": {"type": "object",
                     "properties": {"counterparty": {"type": "string"}},
                     "required": ["counterparty"]}},
]


def call(name, args):
    """Run one tool. Returns (text, is_error). A miss is an ANSWER, not an exception --
    Lab 4.1's contract: a tool that raises tells the agent nothing it can act on."""
    if name == "lookup_payment":
        ref = args.get("ref")
        rec = LEDGER.get(ref)
        if rec is None:
            return f"no payment found with reference {ref!r}", True
        return json.dumps({"ref": ref, **rec}), False
    if name == "policy_for":
        code = args.get("reason_code")
        if code in POLICY:
            return POLICY[code], False
        return f"no policy on file for reason code {code!r}", True
    if name == "screen_counterparty":
        from domain import SANCTIONS_WATCH
        cp = args.get("counterparty")
        listed = cp in SANCTIONS_WATCH
        return json.dumps({"counterparty": cp, "listed": listed,
                           "note": WATCHLIST_NOTE if listed else "not currently listed"}), False
    return f"no such tool: {name!r}", True


def handle(req):
    rid, method = req.get("id"), req.get("method")
    params = req.get("params") or {}
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                           "serverInfo": {"name": "ledger", "version": "2.0.0"}}}
    if method == "notifications/initialized":
        return None                       # a notification has no id and takes no reply
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": SPECS}}
    if method == "tools/call":
        text, is_error = call(params.get("name"), params.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": text}], "isError": is_error}}
    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"method not found: {method!r}"}}


def read_message(stream):
    """One Content-Length-framed message, or None at end of stream."""
    header = b""
    while b"\r\n\r\n" not in header:
        byte = stream.read(1)
        if not byte:
            return None
        header += byte
    n = int(re.search(rb"Content-Length:\s*(\d+)", header).group(1))
    return json.loads(stream.read(n))


def write_message(stream, msg):
    body = json.dumps(msg).encode("utf-8")
    stream.write(b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    stream.flush()


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        req = read_message(stdin)
        if req is None:
            return
        reply = handle(req)
        if reply is not None:
            write_message(stdout, reply)


if __name__ == "__main__":
    main()
