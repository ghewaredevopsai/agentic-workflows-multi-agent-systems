#!/usr/bin/env python3
"""A small MCP client: spawn the server, initialize once, then call tools.

Lab 4.4's client did one round trip and exited. A service keeps the connection, because
spawning a Python interpreter per tool call is a latency budget you do not have.

Thread safety matters here and it is easy to miss: one subprocess, one pair of pipes, and
several requests in flight the moment your service serves two callers at once. Interleave
two writes and both replies are garbage. Hence the lock.
"""
import json, os, re, subprocess, sys, threading


class MCPError(Exception):
    """The server answered, and the answer was an error. Distinct from a transport failure."""


class MCPClient:
    def __init__(self, script="ledger_mcp.py", cwd=None):
        self.cwd = cwd or os.path.dirname(os.path.abspath(__file__))
        self.script = os.path.join(self.cwd, script)
        self._lock = threading.Lock()
        self._id = 0
        self.proc = subprocess.Popen(
            [sys.executable, self.script], cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.server_info = self._rpc("initialize", {})["result"]["serverInfo"]
        self.tools = self._rpc("tools/list", {})["result"]["tools"]

    # ---- transport ----------------------------------------------------------
    def _rpc(self, method, params):
        with self._lock:
            self._id += 1
            body = json.dumps({"jsonrpc": "2.0", "id": self._id,
                               "method": method, "params": params}).encode()
            self.proc.stdin.write(b"Content-Length: " + str(len(body)).encode()
                                  + b"\r\n\r\n" + body)
            self.proc.stdin.flush()
            header = b""
            while b"\r\n\r\n" not in header:
                byte = self.proc.stdout.read(1)
                if not byte:
                    raise MCPError("the MCP server closed the connection")
                header += byte
            n = int(re.search(rb"Content-Length:\s*(\d+)", header).group(1))
            return json.loads(self.proc.stdout.read(n))

    # ---- the useful bit -----------------------------------------------------
    def call(self, name, **arguments):
        """Call one tool and return its text. Raises MCPError if the server said isError."""
        reply = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if "error" in reply:
            raise MCPError(reply["error"]["message"])
        result = reply["result"]
        text = "".join(c.get("text", "") for c in result.get("content", []))
        if result.get("isError"):
            raise MCPError(text)
        return text

    def tool_names(self):
        return [t["name"] for t in self.tools]

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


if __name__ == "__main__":
    c = MCPClient()
    print("server :", c.server_info)
    print("tools  :", c.tool_names())
    print("lookup :", c.call("lookup_payment", ref="PMT-1003"))
    print("policy :", c.call("policy_for", reason_code="LIMIT_BREACH"))
    print("screen :", c.call("screen_counterparty", counterparty="NORTHWIND"))
    try:
        c.call("lookup_payment", ref="PMT-9999")
    except MCPError as exc:
        print("miss   :", exc)
    c.close()
