from datetime import date

import pandas as pd
import pytest

from src.cleaning import (
    clean_articles,
    date_parts,
    normalize_category,
    parse_date,
    parse_revenue,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("5.2B", 5_200_000_000),
        ("$5,200,000,000", 5_200_000_000),
        ("5.2 billion", 5_200_000_000),
        ("$1.480 billion", 1_480_000_000),
        ("500M USD", 500_000_000),
        ("75000.0M USD", 75_000_000_000),
        ("$980.0M", 980_000_000),
        ("€2.1B", 2_310_000_000),  # x1.1
        ("£1,000,000", 1_270_000),  # x1.27
        ("¥150,000,000", 1_000_000),  # /150
        ("€1,254,545,455", 1_380_000_001),
        ("$10M - $20M", 15_000_000),
        ("$38475.0M - $42525.0M", 40_500_000_000),
    ],
)
# Valid formats: suffixes, words, currencies, ranges
def test_parse_revenue_ok(raw, expected):
    assert parse_revenue(raw) == (expected, "ok")


# The parsed value must be an int, not a Decimal or float
def test_parse_revenue_returns_int():
    value, _ = parse_revenue("€2.1B")
    assert isinstance(value, int)


@pytest.mark.parametrize(
    "raw, status",
    [
        (None, "missing"),
        (float("nan"), "missing"),
        ("", "missing"),
        ("  ", "missing"),
        ("null", "missing"),
        ("N/A", "undisclosed"),
        ("Not disclosed", "undisclosed"),
        ("lots of money", "unparseable"),
        ("$10M - ???", "unparseable"),
        ("€5M USD", "unparseable"),
    ],
)
# Missing, undisclosed and bad values never produce an ARR value
def test_parse_revenue_invalid_never_yields_value(raw, status):
    assert parse_revenue(raw) == (None, status)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2022-02-17", date(2022, 2, 17)),
        ("2021-09-11T00:00:00Z", date(2021, 9, 11)),
        ("21 Feb 2020", date(2020, 2, 21)),
        ("October 19, 2022", date(2022, 10, 19)),
        ("02/23/2023", date(2023, 2, 23)),  # second > 12 -> US
        ("23/02/2023", date(2023, 2, 23)),  # first > 12 -> EU
    ],
)
# Valid formats, including numeric dates that are not ambiguous
def test_parse_date_ok(raw, expected):
    assert parse_date(raw) == (expected, "ok")


# NN/NN/YYYY with both numbers <= 12 is read as MM/DD and flagged
def test_parse_date_ambiguous_slash_assumes_us_and_flags():
    assert parse_date("03/04/2023") == (date(2023, 3, 4), "ambiguous_assumed_us")


# NN-NN-YYYY with both numbers <= 12 is read as DD-MM and flagged
def test_parse_date_ambiguous_dash_assumes_eu_and_flags():
    assert parse_date("03-04-2023") == (date(2023, 4, 3), "ambiguous_assumed_eu")


@pytest.mark.parametrize(
    "raw, expected",
    [("23-02-2023", date(2023, 2, 23)), ("02-23-2023", date(2023, 2, 23))],
)
# With dashes, a number above 12 settles which one is the day
def test_parse_date_dash_unambiguous_either_order(raw, expected):
    assert parse_date(raw) == (expected, "ok")


# 05/05/2023 has only one reading, so it is not flagged
def test_parse_date_same_day_and_month_is_not_ambiguous():
    assert parse_date("05/05/2023") == (date(2023, 5, 5), "ok")


@pytest.mark.parametrize("raw", ["2023-02-30", "13/13/2023", "not a date", "31/31/2023"])
# Dates that don't exist or don't match any format
def test_parse_date_invalid(raw):
    assert parse_date(raw) == (None, "invalid")


@pytest.mark.parametrize("raw", [None, "", float("nan")])
# Empty values are missing, not invalid
def test_parse_date_missing(raw):
    assert parse_date(raw) == (None, "missing")


# Year, quarter and month (and nothing for a missing date)
def test_date_parts():
    assert date_parts(date(2023, 5, 12)) == (2023, 2, 5)
    assert date_parts(date(2023, 12, 1)) == (2023, 4, 12)
    assert date_parts(None) == (None, None, None)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("AI/ML", "AI_ML"),
        ("Artificial Intelligence", "AI_ML"),
        ("  machine   learning ", "AI_ML"),
        ("AI & ML", "AI_ML"),
        ("InfoSec", "SECURITY"),
        ("Big Data", "DATA_ANALYTICS"),
        ("Gardening", "OTHER"),
        ("", "UNKNOWN"),
        (None, "UNKNOWN"),
    ],
)
# Maps variations to the standard taxonomy; unmapped is OTHER, empty is UNKNOWN
def test_normalize_category(raw, expected):
    assert normalize_category(raw) == expected


# Cleaned columns are added and the raw ones stay untouched
def test_clean_articles_adds_columns_and_keeps_raw():
    df_raw = pd.DataFrame(
        {
            "article_id": ["A1", "A2", "A3"],
            "published_date": ["2023-05-12", "bad", "03/04/2023"],
            "category": ["AI/ML", "Cloud", "Gardening"],
            "revenue": ["€2.1B", "N/A", ""],
            "author": ["", "Bob", "Eve"],
            "word_count": ["100.0", "", "5"],
        }
    )
    df_clean_articles = clean_articles(df_raw)

    assert list(df_clean_articles["revenue"]) == list(df_raw["revenue"])  # raw untouched
    assert df_clean_articles["arr_usd"].tolist()[0] == 2_310_000_000
    assert pd.isna(df_clean_articles["arr_usd"].iloc[1]) and pd.isna(df_clean_articles["arr_usd"].iloc[2])
    assert list(df_clean_articles["arr_parse_status"]) == ["ok", "undisclosed", "missing"]
    assert list(df_clean_articles["date_parse_status"]) == ["ok", "invalid", "ambiguous_assumed_us"]
    assert pd.isna(df_clean_articles["published_dt"].iloc[1])
    assert df_clean_articles["pub_quarter"].tolist()[0] == 2
    assert list(df_clean_articles["category_std"]) == ["AI_ML", "CLOUD", "OTHER"]
    assert pd.isna(df_clean_articles["author"].iloc[0])
    assert df_clean_articles["word_count"].tolist()[0] == 100
    assert pd.isna(df_clean_articles["word_count"].iloc[1])
    assert len(df_raw.columns) == 6  # input frame not mutated
