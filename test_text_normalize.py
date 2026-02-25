import unittest

from text_normalize import normalize_zh_numbers


class TestNormalizeZhNumbers(unittest.TestCase):
    def test_year(self):
        self.assertEqual(normalize_zh_numbers("二零二零年"), "2020年")
        self.assertEqual(normalize_zh_numbers("二零二六年经济"), "2026年经济")
        self.assertEqual(normalize_zh_numbers("二零二六春节档票房"), "2026春节档票房")

    def test_year_range(self):
        self.assertEqual(normalize_zh_numbers("二零二三至二零二五年"), "2023至2025年")
        self.assertEqual(normalize_zh_numbers("二零二三-二零二五年"), "2023至2025年")

    def test_percent(self):
        self.assertEqual(normalize_zh_numbers("百分之五点五"), "5.5%")
        self.assertEqual(normalize_zh_numbers("百分之五点五左右"), "5.5%左右")
        self.assertEqual(normalize_zh_numbers("百分之四点五至百分之五"), "4.5%至5%")
        self.assertEqual(normalize_zh_numbers("百分之四点五至百分之五点五"), "4.5%至5.5%")

    def test_dates(self):
        self.assertEqual(normalize_zh_numbers("二月十五日"), "2月15日")
        self.assertEqual(normalize_zh_numbers("二月15日"), "2月15日")
        self.assertEqual(normalize_zh_numbers("2月十五日"), "2月15日")
        self.assertEqual(normalize_zh_numbers("从一月开始"), "从1月开始")

    def test_money_units(self):
        self.assertEqual(normalize_zh_numbers("五十七点五二亿元"), "57.52亿元")
        self.assertEqual(normalize_zh_numbers("九十五点一亿"), "95.1亿")
        self.assertEqual(normalize_zh_numbers("六十七点五八亿"), "67.58亿")
        self.assertEqual(normalize_zh_numbers("57个亿"), "57亿")

    def test_counters(self):
        self.assertEqual(normalize_zh_numbers("下调省份多达十九个"), "下调省份多达19个")
        self.assertEqual(normalize_zh_numbers("七十代到六十代再到五十代"), "70代到60代再到50代")
        self.assertEqual(normalize_zh_numbers("连续四年未达标"), "连续四年未达标")
        self.assertEqual(normalize_zh_numbers("1个关键细节"), "一个关键细节")

    def test_no_false_positive(self):
        self.assertEqual(normalize_zh_numbers("一刀切式的环保政策"), "一刀切式的环保政策")


if __name__ == "__main__":
    unittest.main()
