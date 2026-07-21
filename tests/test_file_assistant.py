import os
import shutil
import tempfile
import unittest
from pathlib import Path

from file_assistant import FileAssistant
from mcp_client import MCPClientError, MCPFileToolsClient


FIXTURE = Path(__file__).parent / "fixtures" / "file_assistant" / "project"
USAGE_GOAL = 'Найди все использования компонента "PaymentGateway" и подготовь отчёт'
DOCS_GOAL = "Обнови документацию по структуре Python-модулей проекта"


class FileAssistantIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name) / "project"
        shutil.copytree(FIXTURE, self.project)
        self.previous_root = os.environ.get("AI_ADVENT_FILES_ROOT")
        os.environ["AI_ADVENT_FILES_ROOT"] = str(self.project)
        self.assistant = FileAssistant()

    def tearDown(self):
        self.assistant.close()
        if self.previous_root is None:
            os.environ.pop("AI_ADVENT_FILES_ROOT", None)
        else:
            os.environ["AI_ADVENT_FILES_ROOT"] = self.previous_root
        self.temporary.cleanup()

    def test_mcp_tools_read_search_and_reject_path_escape(self):
        client = MCPFileToolsClient()
        try:
            initialized = client.start()
            self.assertEqual(initialized["serverInfo"]["name"], "project-files")
            tools = {tool["name"] for tool in client.list_tools()}
            self.assertEqual(
                tools,
                {"list_project_files", "read_file", "search_text", "write_file"},
            )
            result = client.call_tool("search_text", {"query": "PaymentGateway"})
            self.assertGreaterEqual(len(result["structuredContent"]["matching_files"]), 3)
            with self.assertRaises(MCPClientError):
                client.call_tool("read_file", {"path": "../outside.txt"})
        finally:
            client.stop()

    def test_goal_finds_usage_across_files_and_writes_report(self):
        result = self.assistant.execute(USAGE_GOAL)
        report = self.project / "assistant_outputs" / "usage_paymentgateway.md"
        self.assertEqual(result.scenario, "component_usage")
        self.assertGreaterEqual(len(result.files_read), 3)
        self.assertTrue(result.changed)
        self.assertIn("+++ b/assistant_outputs/usage_paymentgateway.md", result.diff)
        self.assertTrue(report.is_file())
        content = report.read_text(encoding="utf-8")
        self.assertIn("app/payment_api.py", content)
        self.assertIn("app/order_service.py", content)
        self.assertIn("tests/test_order_service.py", content)

        repeated = self.assistant.execute(USAGE_GOAL)
        self.assertFalse(repeated.changed)
        self.assertEqual(repeated.diff, "")

    def test_goal_updates_code_inventory_and_is_idempotent(self):
        first = self.assistant.execute(DOCS_GOAL)
        inventory = self.project / "docs" / "generated_code_inventory.md"
        self.assertEqual(first.scenario, "update_code_inventory")
        self.assertEqual(len(first.files_read), 3)
        self.assertTrue(inventory.is_file())
        content = inventory.read_text(encoding="utf-8")
        self.assertIn("`PaymentGateway`", content)
        self.assertIn("`OrderService`", content)
        self.assertIn("`Order`", content)

        second = self.assistant.execute(DOCS_GOAL)
        self.assertFalse(second.changed)
        self.assertEqual(second.diff, "")

    def test_dry_run_returns_diff_without_writing(self):
        result = self.assistant.execute(DOCS_GOAL, dry_run=True)
        inventory = self.project / "docs" / "generated_code_inventory.md"
        self.assertTrue(result.changed)
        self.assertTrue(result.dry_run)
        self.assertIn("+++ b/docs/generated_code_inventory.md", result.diff)
        self.assertFalse(inventory.exists())


if __name__ == "__main__":
    unittest.main()
