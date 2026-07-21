import json
import threading
import unittest
import urllib.request
from http.server import HTTPServer

from mcp_client import MCPCRMClient
from support_service import (
    DEFAULT_KB_DIR,
    SupportAssistant,
    SupportRagIndex,
    make_http_handler,
)


QUESTION = "Почему не работает авторизация?"


class CrmMcpTests(unittest.TestCase):
    def test_mcp_exposes_users_and_ticket_context(self):
        client = MCPCRMClient()
        try:
            initialized = client.start()
            self.assertEqual(initialized["serverInfo"]["name"], "json-crm")
            tools = {tool["name"] for tool in client.list_tools()}
            self.assertEqual(tools, {"get_user", "get_ticket_context", "list_tickets"})
            result = client.call_tool("get_ticket_context", {"ticket_id": "TCK-1001"})
            self.assertEqual(result["structuredContent"]["user"]["id"], "USR-1001")
            self.assertEqual(
                result["structuredContent"]["ticket"]["diagnostic_code"],
                "AUTH_ACCOUNT_LOCKED",
            )
        finally:
            client.stop()


class SupportRagTests(unittest.TestCase):
    def test_indexes_faq_and_product_documentation(self):
        index = SupportRagIndex(DEFAULT_KB_DIR)
        sources = {chunk.source for chunk in index.chunks}
        self.assertEqual(sources, {"faq.md", "authentication.md"})
        hits = index.search("AUTH_EMAIL_UNVERIFIED email_verified false")
        self.assertEqual(hits[0].chunk.section.split(" — ")[0], "AUTH_EMAIL_UNVERIFIED")


class SupportAssistantTests(unittest.TestCase):
    def setUp(self):
        self.assistant = SupportAssistant()

    def tearDown(self):
        self.assistant.close()

    def test_same_question_uses_different_ticket_context(self):
        locked = self.assistant.ask("TCK-1001", QUESTION)
        unverified = self.assistant.ask("TCK-1002", QUESTION)
        sso = self.assistant.ask("TCK-1003", QUESTION)

        self.assertIn("AUTH_ACCOUNT_LOCKED", locked.answer)
        self.assertIn("пяти неверных попыток", locked.answer)
        self.assertIn("AUTH_EMAIL_UNVERIFIED", unverified.answer)
        self.assertIn("подтверждения адреса", unverified.answer)
        self.assertIn("AUTH_SSO_DOMAIN_MISMATCH", sso.answer)
        self.assertIn("обновить SSO", sso.answer)
        self.assertNotEqual(locked.answer, unverified.answer)

    def test_response_contains_safe_crm_context_and_rag_sources(self):
        response = self.assistant.ask("TCK-1001", QUESTION)
        serialized = json.dumps(response.to_dict(), ensure_ascii=False)
        self.assertEqual(response.crm_context["user"]["plan"], "pro")
        self.assertTrue(response.crm_context["ticket"]["signals"]["account_locked"])
        self.assertTrue(response.sources)
        self.assertIn("faq.md", {source["source"] for source in response.sources})
        self.assertNotIn("anna.demo@example.test", serialized)
        self.assertNotIn("Анна Демо", serialized)

    def test_http_support_endpoint(self):
        server = HTTPServer(("127.0.0.1", 0), make_http_handler(self.assistant))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/health", timeout=5
            ) as health_response:
                health = json.loads(health_response.read().decode("utf-8"))
            self.assertEqual(health["status"], "ok")
            self.assertGreater(health["rag_chunks"], 0)

            payload = json.dumps(
                {"ticket_id": "TCK-1002", "question": QUESTION},
                ensure_ascii=False,
            ).encode("utf-8")
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/support",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertEqual(body["ticket_id"], "TCK-1002")
            self.assertIn("AUTH_EMAIL_UNVERIFIED", body["answer"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
