import unittest
from tempfile import NamedTemporaryFile

import transcription_service as svc


class TestChunkTextMerge(unittest.TestCase):
    def test_exact_overlap(self):
        merged = svc._merge_texts_with_overlap(
            [
                "前面内容。我们讨论财务洗澡和去库存，这个逻辑很重要。",
                "财务洗澡和去库存，这个逻辑很重要。后面继续看数据。",
            ],
            fuzzy=True,
        )

        self.assertEqual(
            merged,
            "前面内容。我们讨论财务洗澡和去库存，这个逻辑很重要。后面继续看数据。",
        )

    def test_fuzzy_overlap(self):
        merged = svc._merge_texts_with_overlap(
            [
                "前面内容。我们讨论财务洗澡和去库存，这个逻辑很重要。",
                "财务洗澡和去库存在这个逻辑很重要。后面继续看数据。",
            ],
            fuzzy=True,
        )

        self.assertEqual(
            merged,
            "前面内容。我们讨论财务洗澡和去库存，这个逻辑很重要。后面继续看数据。",
        )

    def test_no_overlap_keeps_both_chunks(self):
        merged = svc._merge_texts_with_overlap(
            [
                "我们讨论消费行业的库存压力。",
                "接下来再看科技公司的估值。",
            ],
            fuzzy=True,
        )

        self.assertEqual(merged, "我们讨论消费行业的库存压力。\n接下来再看科技公司的估值。")


class TestPromptContext(unittest.TestCase):
    def test_prompt_terms_file_ignores_comments_and_duplicates(self):
        with NamedTemporaryFile("w+", encoding="utf-8") as f:
            f.write("# comment\n\nDCF\nDCF\n财务洗澡\n")
            f.flush()

            self.assertEqual(svc._load_prompt_terms(f.name), ("DCF", "财务洗澡"))

    def test_default_finance_prompt_is_used_without_user_prompt(self):
        context = svc._build_transcription_context(None)

        self.assertIn("price in", context)
        self.assertIn("财务洗澡", context)
        self.assertIn("半导体", context)
        self.assertIn("去产能", context)

    def test_default_finance_prompt_has_many_terms_from_file(self):
        terms = svc._load_prompt_terms(svc.DEFAULT_FINANCE_PROMPT_PATH)

        self.assertGreaterEqual(len(terms), 120)

    def test_user_prompt_is_kept_before_default_finance_prompt(self):
        context = svc._build_transcription_context("重点识别公司简称。")

        self.assertTrue(context.startswith("重点识别公司简称。"))
        self.assertIn("price in", context)
        self.assertIn("无风险收益率", context)


if __name__ == "__main__":
    unittest.main()
