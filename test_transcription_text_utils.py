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


class TestTranscriptionOptions(unittest.TestCase):
    def test_default_options_keep_json_mode(self):
        options = svc.normalize_transcription_options()

        self.assertEqual(options.response_format, "json")
        self.assertEqual(options.timestamp_granularities, ())
        self.assertFalse(options.return_sentence_timestamps)

    def test_sentence_granularity_requires_verbose_json(self):
        with self.assertRaisesRegex(svc.InputValidationError, "response_format=verbose_json"):
            svc.normalize_transcription_options(
                response_format="json",
                timestamp_granularities=["sentence"],
            )

    def test_segment_alias_maps_to_sentence(self):
        options = svc.normalize_transcription_options(
            response_format="verbose_json",
            timestamp_granularities=["segment,sentence"],
        )

        self.assertEqual(options.timestamp_granularities, ("sentence",))
        self.assertTrue(options.return_sentence_timestamps)

    def test_word_granularity_is_rejected(self):
        with self.assertRaisesRegex(svc.InputValidationError, "word is not supported"):
            svc.normalize_transcription_options(
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

    def test_unknown_response_format_is_rejected(self):
        with self.assertRaisesRegex(svc.InputValidationError, "response_format"):
            svc.normalize_transcription_options(response_format="srt")


class TestSentenceTimestamps(unittest.TestCase):
    def test_chinese_sentence_timestamps(self):
        sentences = svc.build_sentence_timestamps(
            "第一句。第二句！",
            [
                {"text": "第", "start_time": 0.0, "end_time": 0.1},
                {"text": "一", "start_time": 0.1, "end_time": 0.2},
                {"text": "句", "start_time": 0.2, "end_time": 0.5},
                {"text": "第", "start_time": 0.7, "end_time": 0.8},
                {"text": "二", "start_time": 0.8, "end_time": 0.9},
                {"text": "句", "start_time": 0.9, "end_time": 1.2},
            ],
            duration=1.4,
        )

        self.assertEqual(
            sentences,
            [
                {"id": 0, "text": "第一句。", "start": 0.0, "end": 0.5},
                {"id": 1, "text": "第二句！", "start": 0.7, "end": 1.2},
            ],
        )

    def test_english_sentence_timestamps(self):
        sentences = svc.build_sentence_timestamps(
            "Hello, world. Next sentence.",
            [
                {"text": "Hello", "start_time": 0.0, "end_time": 0.3},
                {"text": "world", "start_time": 0.35, "end_time": 0.8},
                {"text": "Next", "start_time": 1.0, "end_time": 1.2},
                {"text": "sentence", "start_time": 1.25, "end_time": 1.8},
            ],
            duration=2.0,
        )

        self.assertEqual(sentences[0]["text"], "Hello, world.")
        self.assertEqual(sentences[0]["start"], 0.0)
        self.assertEqual(sentences[0]["end"], 0.8)
        self.assertEqual(sentences[1]["text"], "Next sentence.")
        self.assertEqual(sentences[1]["start"], 1.0)
        self.assertEqual(sentences[1]["end"], 1.8)

    def test_decimal_period_does_not_split_sentence(self):
        sentences = svc.build_sentence_timestamps(
            "增速是5.5%。继续看。",
            [
                {"text": "增", "start_time": 0.0, "end_time": 0.1},
                {"text": "速", "start_time": 0.1, "end_time": 0.2},
                {"text": "是", "start_time": 0.2, "end_time": 0.3},
                {"text": "5", "start_time": 0.3, "end_time": 0.35},
                {"text": "5", "start_time": 0.36, "end_time": 0.4},
                {"text": "继", "start_time": 0.7, "end_time": 0.8},
                {"text": "续", "start_time": 0.8, "end_time": 1.0},
                {"text": "看", "start_time": 1.0, "end_time": 1.2},
            ],
            duration=1.4,
        )

        self.assertEqual([item["text"] for item in sentences], ["增速是5.5%。", "继续看。"])

    def test_missing_alignment_falls_back_to_duration(self):
        sentences = svc.build_sentence_timestamps("没有对齐。", [], duration=2.0)

        self.assertEqual(sentences, [{"id": 0, "text": "没有对齐。", "start": 0.0, "end": 2.0}])


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
