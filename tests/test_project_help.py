from pathlib import Path
import unittest

from mcp_client import MCPProjectClient
from project_help import ProjectHelpAssistant, ProjectRagIndex, ROOT


class ProjectRagIndexTests(unittest.TestCase):
    def test_indexes_readme_and_docs(self):
        index = ProjectRagIndex(ROOT)
        sources = {chunk.source for chunk in index.chunks}
        self.assertIn("README.md", sources)
        self.assertIn("docs/project_assistant.md", sources)

    def test_finds_project_structure(self):
        hits = ProjectRagIndex(ROOT).search("структура проекта main.py MCP")
        self.assertTrue(hits)
        self.assertIn(hits[0].chunk.source, {"README.md", "docs/project_assistant.md"})


class ProjectMcpTests(unittest.TestCase):
    def test_git_branch_over_mcp(self):
        client = MCPProjectClient()
        try:
            initialized = client.start()
            self.assertEqual(initialized["serverInfo"]["name"], "ai-advent-project")
            tools = {tool["name"] for tool in client.list_tools()}
            self.assertEqual(tools, {"git_branch", "list_files", "git_diff"})
            result = client.call_tool("git_branch", {})
            self.assertTrue(result["structuredContent"]["branch"])
        finally:
            client.stop()

    def test_help_answer_contains_rag_and_mcp_context(self):
        assistant = ProjectHelpAssistant()
        try:
            answer = assistant.ask("Как устроена структура проекта?")
            self.assertIn("Источники RAG:", answer)
            self.assertIn("Контекст проекта (MCP): git branch =", answer)
            self.assertIn("README.md", answer)
        finally:
            assistant.close()


if __name__ == "__main__":
    unittest.main()
