import unittest

from text_normalize import normalize_zh_numbers


class TestNormalizeZhNumbers(unittest.TestCase):
    def test_year(self):
        self.assertEqual(normalize_zh_numbers("二零二零年"), "2020年")
        self.assertEqual(normalize_zh_numbers("二零二六年经济"), "2026年经济")

    def test_year_range(self):
        self.assertEqual(normalize_zh_numbers("二零二三至二零二五年"), "2023至2025年")
        self.assertEqual(normalize_zh_numbers("二零二三-二零二五年"), "2023至2025年")

    def test_percent(self):
        self.assertEqual(normalize_zh_numbers("百分之五点五"), "5.5%")
        self.assertEqual(normalize_zh_numbers("百分之五点五左右"), "5.5%左右")
        self.assertEqual(normalize_zh_numbers("百分之四点五至百分之五"), "4.5%至5%")
        self.assertEqual(normalize_zh_numbers("百分之四点五至百分之五点五"), "4.5%至5.5%")

    def test_counters(self):
        self.assertEqual(normalize_zh_numbers("下调省份多达十九个"), "下调省份多达19个")
        self.assertEqual(normalize_zh_numbers("七十代到六十代再到五十代"), "70代到60代再到50代")
        self.assertEqual(normalize_zh_numbers("连续四年未达标"), "连续四年未达标")
        self.assertEqual(normalize_zh_numbers("1个关键细节"), "一个关键细节")

    def test_no_false_positive(self):
        self.assertEqual(normalize_zh_numbers("一刀切式的环保政策"), "一刀切式的环保政策")


if __name__ == "__main__":
    unittest.main()
