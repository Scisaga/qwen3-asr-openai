import re


_ZH_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "两": 2,
}

_ZH_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_ZH_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}


def _is_zh_digit_seq(s: str) -> bool:
    return bool(s) and all(ch in _ZH_DIGITS for ch in s)


def _zh_digit_seq_to_int_str(s: str) -> str:
    return "".join(str(_ZH_DIGITS[ch]) for ch in s)


def _zh_int_to_int(s: str) -> int | None:
    s = (s or "").strip()
    if not s:
        return None
    if _is_zh_digit_seq(s):
        return int(_zh_digit_seq_to_int_str(s))

    total = 0
    section = 0
    number = 0
    has_any = False
    for ch in s:
        if ch in _ZH_DIGITS:
            number = _ZH_DIGITS[ch]
            has_any = True
            continue
        if ch in _ZH_SMALL_UNITS:
            unit = _ZH_SMALL_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
            has_any = True
            continue
        if ch in _ZH_LARGE_UNITS:
            unit = _ZH_LARGE_UNITS[ch]
            section = (section + number) * unit
            total += section
            section = 0
            number = 0
            has_any = True
            continue
        return None
    if not has_any:
        return None
    return total + section + number


def _zh_num_to_str(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None

    if "点" in s:
        left, right = s.split("点", 1)
        left_v = _zh_int_to_int(left) if left else 0
        if left_v is None:
            return None
        if not right:
            return str(left_v)
        if not _is_zh_digit_seq(right):
            return None
        dec = "".join(str(_ZH_DIGITS[ch]) for ch in right)
        dec = dec.rstrip("0")
        return f"{left_v}.{dec}" if dec else str(left_v)

    iv = _zh_int_to_int(s)
    if iv is None:
        return None
    return str(iv)


_RE_ZH_YEAR = re.compile(r"(?P<y>[零〇一二三四五六七八九]{4})年")
_RE_ZH_PERCENT_RANGE = re.compile(
    r"百分之(?P<a>[零〇一二三四五六七八九两十百千万亿点]+)"
    r"(?P<sep>至|到|—|–|－|-|~|～)"
    r"(?:百分之)?(?P<b>[零〇一二三四五六七八九两十百千万亿点]+)"
)
_RE_ZH_PERCENT = re.compile(r"百分之(?P<n>[零〇一二三四五六七八九两十百千万亿点]+)")
_RE_ZH_COUNTER = re.compile(r"(?P<n>[零〇一二三四五六七八九两十百千万亿点]+)(?P<u>个|年|月|日|天|次|省|市|项|条|位|名|代)")
_RE_ZH_YEAR_RANGE = re.compile(
    r"(?P<a>[零〇一二三四五六七八九]{4})"
    r"(?P<sep>至|到|—|–|－|-|~|～)"
    r"(?P<b>[零〇一二三四五六七八九]{4})年"
)
_RE_ARABIC_ONE_COUNTER = re.compile(r"(?<!\d)1(?P<u>个|年|月|日|天|次|省|市|项|条|位|名|代)")


def normalize_zh_numbers(text: str) -> str:
    """
    Normalize common Chinese numeric expressions in ASR outputs.
    - Years: 二零二六年 -> 2026年
    - Percentages: 百分之五点五 -> 5.5%
    - Some counters: 十九个 -> 19个, 连续四年 -> 连续4年
    """
    t = text or ""

    def repl_year(m: re.Match) -> str:
        return f"{_zh_digit_seq_to_int_str(m.group('y'))}年"

    def repl_year_range(m: re.Match) -> str:
        a = _zh_digit_seq_to_int_str(m.group("a"))
        b = _zh_digit_seq_to_int_str(m.group("b"))
        return f"{a}至{b}年"

    def repl_percent_range(m: re.Match) -> str:
        a = _zh_num_to_str(m.group("a"))
        b = _zh_num_to_str(m.group("b"))
        if a is None or b is None:
            return m.group(0)
        return f"{a}%至{b}%"

    def repl_percent(m: re.Match) -> str:
        n = _zh_num_to_str(m.group("n"))
        if n is None:
            return m.group(0)
        return f"{n}%"

    def repl_counter(m: re.Match) -> str:
        raw = m.group("n")
        if raw in _ZH_DIGITS and len(raw) == 1:
            return m.group(0)
        n = _zh_num_to_str(raw)
        if n is None:
            return m.group(0)
        return f"{n}{m.group('u')}"

    t = _RE_ZH_YEAR_RANGE.sub(repl_year_range, t)
    t = _RE_ZH_YEAR.sub(repl_year, t)
    t = _RE_ZH_PERCENT_RANGE.sub(repl_percent_range, t)
    t = _RE_ZH_PERCENT.sub(repl_percent, t)
    t = _RE_ZH_COUNTER.sub(repl_counter, t)
    t = _RE_ARABIC_ONE_COUNTER.sub(lambda m: f"一{m.group('u')}", t)
    return t
