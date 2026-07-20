import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from pr_review import (
    ReviewAnalyzer,
    ReviewKnowledgeIndex,
    format_review,
    main,
    parse_unified_diff,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pr_review"
DIFF_FILE = FIXTURE_ROOT / "problematic_pr.diff"
PROJECT_ROOT = FIXTURE_ROOT / "project"


class DiffParserTests(unittest.TestCase):
    def test_extracts_changed_file_and_line_numbers(self):
        files = parse_unified_diff(DIFF_FILE.read_text(encoding="utf-8"))
        self.assertEqual([item.path for item in files], ["app/payment_service.py"])
        self.assertGreater(len(files[0].added), 3)
        self.assertTrue(any(line.new_line for line in files[0].added))


class RagReviewTests(unittest.TestCase):
    def setUp(self):
        self.changed = parse_unified_diff(DIFF_FILE.read_text(encoding="utf-8"))
        self.index = ReviewKnowledgeIndex(PROJECT_ROOT)

    def test_rag_indexes_documentation_and_code(self):
        types = {chunk.source_type for chunk in self.index.chunks}
        self.assertEqual(types, {"documentation", "code"})
        hits = self.index.search("payment timeout validation")
        self.assertEqual({hit.chunk.source_type for hit in hits}, {"documentation", "code"})

    def test_review_has_all_required_categories(self):
        findings = ReviewAnalyzer(self.index).analyze(self.changed)
        categories = {finding.category for finding in findings}
        self.assertEqual(categories, {"bug", "architecture", "recommendation"})
        titles = {finding.title for finding in findings}
        self.assertIn("Секрет добавлен в исходный код", titles)
        self.assertIn("Удалена проверка входных данных или прав", titles)
        self.assertIn("Нет изменений в автоматических тестах", titles)

    def test_report_contains_diff_files_and_rag_sources(self):
        findings = ReviewAnalyzer(self.index).analyze(self.changed)
        report = format_review("fixture", self.changed, findings, self.index)
        self.assertIn("## Потенциальные баги", report)
        self.assertIn("## Архитектурные проблемы", report)
        self.assertIn("## Рекомендации", report)
        self.assertIn("app/payment_service.py", report)
        self.assertIn("RAG-контекст (документация + код)", report)

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.md"
            with redirect_stdout(io.StringIO()):
                exit_code = main([
                    "--project-root", str(PROJECT_ROOT),
                    "--diff-file", str(DIFF_FILE),
                    "--output", str(output),
                ])
            self.assertEqual(exit_code, 0)
            self.assertIn("Автоматическое AI-ревью", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
