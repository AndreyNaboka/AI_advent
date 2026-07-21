"""Small stdio MCP client used by the interactive application."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class MCPClientError(RuntimeError):
    pass


class MCPNewsClient:
    def __init__(self, server_path: Optional[Path] = None):
        self.server_path = server_path or Path(__file__).with_name("mcp_news_server.py")
        self.process: Optional[subprocess.Popen[str]] = None
        self.request_id = 0

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> Dict[str, Any]:
        if self.is_running:
            return {"already_running": True}
        self.process = subprocess.Popen(
            [sys.executable, str(self.server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        try:
            initialized = self._request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ai-advent-chat", "version": "1.0.0"},
                },
            )
            self._notify("notifications/initialized")
            return initialized
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
        self.process = None

    def list_tools(self) -> List[Dict[str, Any]]:
        self._require_running()
        return self._request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._require_running()
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def _require_running(self) -> None:
        if not self.is_running:
            raise MCPClientError("MCP-сервер не запущен. Используйте /mcp-start")

    def _send(self, payload: Dict[str, Any]) -> None:
        self._require_running()
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise MCPClientError(f"Связь с MCP-сервером потеряна: {error}") from error

    def _notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def _request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.request_id += 1
        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        self._send(payload)

        assert self.process is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise MCPClientError("MCP-сервер завершился без ответа")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise MCPClientError(f"MCP-сервер вернул некорректный JSON: {line!r}") from error
        if response.get("id") != self.request_id:
            raise MCPClientError("MCP-сервер вернул ответ с неверным id")
        if "error" in response:
            error = response["error"]
            raise MCPClientError(error.get("message", str(error)))
        result = response.get("result")
        if not isinstance(result, dict):
            raise MCPClientError("В ответе MCP-сервера отсутствует result")
        return result


class MCPPeriodicSummaryClient(MCPNewsClient):
    def __init__(self, server_path: Optional[Path] = None):
        super().__init__(
            server_path or Path(__file__).with_name("mcp_summary_server.py")
        )


class MCPCodeReviewClient(MCPNewsClient):
    def __init__(self, server_path: Optional[Path] = None):
        super().__init__(
            server_path or Path(__file__).with_name("mcp_code_review_server.py")
        )


class MCPProjectClient(MCPNewsClient):
    """Client for the read-only MCP server exposing the current project context."""

    def __init__(self, server_path: Optional[Path] = None):
        super().__init__(
            server_path or Path(__file__).with_name("mcp_project_server.py")
        )


class MCPCRMClient(MCPNewsClient):
    """Client for the read-only JSON CRM used by the support assistant."""

    def __init__(self, server_path: Optional[Path] = None):
        super().__init__(
            server_path or Path(__file__).with_name("mcp_crm_server.py")
        )
