"""Cleaning functions for the raw article table.

Parsers never raise on bad input. They return (value, status) so that failed
parses stay visible downstream instead of being dropped or treated as valid.
"""
import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

#### VALUE/AMOUNT/REVENUE VARIABLES ####

# Fixed conversion rates (currency -> USD), as defined in the assignment
FX_TO_USD = {
    "USD": Decimal(1),
    "EUR": Decimal("1.1"),
    "GBP": Decimal("1.27"),
    "JPY": Decimal(1) / Decimal(150),
}

# Symbol that comes before the amount -> currency code
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

# Suffix that comes after the amount -> how much it multiplies by
MULTIPLIERS = {
    "k": 10**3, "thousand": 10**3,
    "m": 10**6, "million": 10**6,
    "b": 10**9, "billion": 10**9,
}

# Values that mean "there is no value" (status "missing")
MISSING_TOKENS = {"", "null", "none", "nan"}

# Values that mean "there is a value, but it was not published" (status "undisclosed")
UNDISCLOSED_TOKENS = {"n/a", "na", "not disclosed", "undisclosed"}

AMOUNT_RE = re.compile(
    r"^(?P<symbol>[$€£¥])?\s*"                              # Regex for symbol
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*"                    # Regex for number
    r"(?P<multiplier>thousand|million|billion|[kmb])?\s*"   # Regex for multiplier
    r"(?P<code>usd|eur|gbp|jpy)?$",                         # Regex for currency code
    re.IGNORECASE,
)

######################################

#### DATE VARIABLES ####

# Splits a range like "$10M - $20M" into its two ends
RANGE_SPLIT_RE = re.compile(r"\s+(?:-|–|to)\s+", re.IGNORECASE)

# 2022-02-17 or 2021-09-11T00:00:00Z
ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:T.*)?$")

# 02/23/2023 or 23-02-2023 (keeps the separator, it helps to guess ambiguous dates)
NUMERIC_RE = re.compile(r"^(\d{1,2})([/-])(\d{1,2})[/-](\d{4})$")

# 21 Feb 2020 or October 19, 2022
TEXT_DATE_FORMATS = ("%d %b %Y", "%B %d, %Y")

######################################

#### CATEGORY VARIABLES ####

# Raw category (lowercase) -> standard category
CATEGORY_MAP = {
    "ai/ml": "AI_ML",
    "ai & ml": "AI_ML",
    "artificial intelligence": "AI_ML",
    "machine learning": "AI_ML",
    "cloud": "CLOUD",
    "cloud computing": "CLOUD",
    "cloud services": "CLOUD",
    "analytics": "DATA_ANALYTICS",
    "data analytics": "DATA_ANALYTICS",
    "big data": "DATA_ANALYTICS",
    "fintech": "FINTECH",
    "financial technology": "FINTECH",
    "finance": "FINTECH",
    "cybersecurity": "SECURITY",
    "infosec": "SECURITY",
    "security": "SECURITY",
    "software": "SOFTWARE",
    "enterprise software": "SOFTWARE",
    "saas": "SOFTWARE",
}

######################################


def _is_null(value) -> bool:
    """True for None and NaN (the two ways a null shows up in pandas)."""
    return value is None or (isinstance(value, float) and pd.isna(value))


def _parse_amount_usd(text: str) -> Decimal | None:
    """Parse a single amount like '€2.1B' or '500M USD' into USD."""

    # Gets the amount without symbols or letters
    match = AMOUNT_RE.match(text.strip())

    # If the text doesn't look like an amount, returns None
    if not match:
        return None

    # Currency from the symbol ($, €...) and from the code (USD, EUR...), if any
    symbol_currency = CURRENCY_SYMBOLS.get(match["symbol"])
    code_currency = match["code"].upper() if match["code"] else None

    # If regex and mapping get different currency codes, returns None
    if symbol_currency and code_currency and (symbol_currency != code_currency):
        return None

    # Gets currency (default USD)
    currency = symbol_currency or code_currency or "USD"

    # Format amount value
    amount = Decimal(match["number"].replace(",", ""))

    # Check if it has a multiplier
    if match["multiplier"]:
        amount *= MULTIPLIERS[match["multiplier"].lower()]

    # Returns the formated amount x currency convertion to USD
    return amount * FX_TO_USD[currency]


def parse_revenue(raw) -> tuple[int | None, str]:
    """Return (arr_usd, status); arr_usd is only set when status is 'ok'.

    A range such as '$10M - $20M' is converted to its USD midpoint.
    """

    # If value if None or missing, set value to None
    if _is_null(raw) or str(raw).strip().lower() in MISSING_TOKENS:
        return None, "missing"

    # If value is not disclosed, set value to None
    text = str(raw).strip()
    if text.lower() in UNDISCLOSED_TOKENS:
        return None, "undisclosed"

    # Checking if value is an interval
    # len(bounds) = 1 -> unique value
    # len(bounds) = 2 -> interval value (gets midpoint)
    # len(bounds) > 2 -> invalid interval value, set to None
    bounds = [_parse_amount_usd(part) for part in RANGE_SPLIT_RE.split(text)]

    if len(bounds) > 2 or None in bounds:
        return None, "unparseable"

    # Value in USD (if bounds is 1, value remains itself)
    usd = sum(bounds) / len(bounds)

    # Return whole USD, rounding ties away from zero (ROUND_HALF_UP):
    # 2.5 -> 3, 2.49 -> 2, and -2.5 -> -3.
    return int(usd.quantize(Decimal(1), rounding=ROUND_HALF_UP)), "ok"


def _make_date(year: int, month: int, day: int) -> date | None:
    """Build a date, or None if it doesn't exist (e.g. 2023-02-30)."""

    # date() raises ValueError for impossible dates, so we catch it
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date(raw) -> tuple[date | None, str]:
    """Return (date, status).

    For NN/NN/YYYY and NN-NN-YYYY, a number above 12 settles which one is the
    day. When both are 12 or below the date is ambiguous, and the separator
    decides, based on the unambiguous rows in this dataset:
      '/' -> MM/DD (the first number is never above 12 in any row)
      '-' -> DD-MM (the first number is above 12 in 63 rows, the second in 17)
    Ambiguous rows get status 'ambiguous_assumed_us' or 'ambiguous_assumed_eu'.
    """

    # If value is None or missing, set date to None
    if _is_null(raw) or str(raw).strip().lower() in MISSING_TOKENS:
        return None, "missing"

    text = str(raw).strip()

    # ISO format (2022-02-17 or 2021-09-11T00:00:00Z), always YYYY-MM-DD
    if match := ISO_RE.match(text):
        parsed = _make_date(*map(int, match.groups()))
        return (parsed, "ok") if parsed else (None, "invalid")

    # Numeric format (NN/NN/YYYY or NN-NN-YYYY), day and month can be in any order
    if match := NUMERIC_RE.match(text):
        first, separator, second, year = int(match[1]), match[2], int(match[3]), int(match[4])

        # First number can't be a month, so it is the day
        if first > 12:
            month, day, status = second, first, "ok"

        # Second number can't be a month (or both are equal), so there is no doubt
        elif second > 12 or first == second:
            month, day, status = first, second, "ok"

        # Both could be the month, so we guess from the separator and flag the row
        elif separator == "/":
            month, day, status = first, second, "ambiguous_assumed_us"
        else:
            month, day, status = second, first, "ambiguous_assumed_eu"

        # Invalid if the date doesn't exist (e.g. 31/31/2023)
        parsed = _make_date(year, month, day)
        return (parsed, status) if parsed else (None, "invalid")

    # Text format (21 Feb 2020 or October 19, 2022), tries each one until it fits
    for fmt in TEXT_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date(), "ok"
        except ValueError:
            pass

    # No format matched
    return None, "invalid"


def date_parts(value: date | None) -> tuple[int | None, int | None, int | None]:
    """Return (year, quarter, month)."""

    # If there is no date, there are no parts either
    if value is None:
        return None, None, None

    # Months 1-3 -> Q1, 4-6 -> Q2, 7-9 -> Q3, 10-12 -> Q4
    return value.year, (value.month - 1) // 3 + 1, value.month


def normalize_category(raw) -> str:
    """Map a raw category to the standard taxonomy; UNKNOWN if empty, OTHER if unmapped."""

    # If value is None or empty, there is no category to map
    if _is_null(raw) or not str(raw).strip():
        return "UNKNOWN"

    # Lowercase and single spaces, so "  Machine   Learning " matches the map
    key = " ".join(str(raw).lower().split())

    # If the category is not in the map, set it to OTHER
    return CATEGORY_MAP.get(key, "OTHER")


def clean_articles(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return the articles with cleaned columns added next to the raw ones.

    Raw columns are left untouched so every cleaned value can be audited
    against its source. (Note: "word_count" is the last raw column from .csv)
    """

    # Works on a copy, so the input table is not changed
    df_clean_articles = df_raw.copy()

    # Gets cleaned "renevue" column and a status for the conversion
    # If arr_parse_status <> "ok", something wrong happend on the conversion
    arr = [parse_revenue(value) for value in df_clean_articles["revenue"]]
    df_clean_articles["arr_usd"] = pd.array([value for value, _ in arr], dtype="Int64")
    df_clean_articles["arr_parse_status"] = [status for _, status in arr]

    # Gets cleaned "published_date" column and a status for the conversion
    # If date_parse_status <> "ok", something wrong happend on the conversion
    dates = [parse_date(value) for value in df_clean_articles["published_date"]]
    df_clean_articles["published_dt"] = pd.to_datetime([value for value, _ in dates])
    df_clean_articles["date_parse_status"] = [status for _, status in dates]

    # Gets year, quarter, month from cleaned "published_date"
    year, quarter, month = zip(*(date_parts(value) for value, _ in dates))
    df_clean_articles["pub_year"] = pd.array(year, dtype="Int64")
    df_clean_articles["pub_quarter"] = pd.array(quarter, dtype="Int64")
    df_clean_articles["pub_month"] = pd.array(month, dtype="Int64")

    # Gets normalized "category_std" for each company
    df_clean_articles["category_std"] = df_clean_articles["category"].map(normalize_category)

    # Cleans "word_count" to int number
    df_clean_articles["word_count"] = pd.to_numeric(
        df_clean_articles["word_count"], errors="coerce"
    ).astype("Int64")

    # Replaces empty "author" with N/A
    df_clean_articles["author"] = df_clean_articles["author"].replace("", pd.NA)

    return df_clean_articles
