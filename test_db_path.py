"""feedback.db must stay out of cloud-synced folders."""
import unittest
from pathlib import Path

import app


class ResolveDbPathTest(unittest.TestCase):
    def test_local_checkout_keeps_db_next_to_app(self):
        root = Path("/Users/someone/CCowork-Local-Apps/gemini-coaching-agent-starter")
        self.assertEqual(app.resolve_db_path(root, {}), root / "feedback.db")

    def test_cloud_synced_checkout_moves_db_to_local_disk(self):
        root = Path("/Users/someone/Library/CloudStorage/OneDrive-Personal/CCowork/gemini-coaching-agent-starter")
        got = app.resolve_db_path(root, {}, home="/Users/someone")
        self.assertEqual(got, Path("/Users/someone/Local-Infra/app-data/gemini-coaching-agent-starter/feedback.db"))
        self.assertNotIn("CloudStorage", got.parts)

    def test_env_override_wins(self):
        root = Path("/Users/someone/Library/CloudStorage/OneDrive-Personal/x")
        got = app.resolve_db_path(root, {"FEEDBACK_DB_PATH": "/tmp/custom/feedback.db"})
        self.assertEqual(got, Path("/tmp/custom/feedback.db"))

    def test_blank_override_is_ignored(self):
        root = Path("/srv/app")
        self.assertEqual(app.resolve_db_path(root, {"FEEDBACK_DB_PATH": "  "}), root / "feedback.db")

    def test_vercel_uses_tmp(self):
        self.assertEqual(app.resolve_db_path(Path("/var/task"), {"VERCEL": "1"}), Path("/tmp/feedback.db"))


if __name__ == "__main__":
    unittest.main()
