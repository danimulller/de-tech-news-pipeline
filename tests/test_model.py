import duckdb
import pandas as pd

from src.cleaning import clean_articles
from src.enrich import enrich_articles
from src.model import build_ai_articles_enriched, build_model, export_tables, load_to_duckdb

RAW_COLUMNS = [
    "article_id", "title", "company_name", "published_date", "category",
    "revenue", "summary", "url", "author", "word_count",
]


def make_tables(df_companies, rows):
    """Run the cleaning, the enrichment and the model on a few raw rows.

    Each row is (article_id, company_name, published_date, revenue) and, optionally, the category.
    """

    df_raw = pd.DataFrame(
        [
            [a, f"Title {a}", c, d, category[0] if category else "AI/ML", r, f"Summary {a}", f"https://x/{a}", "", "100"]
            for a, c, d, r, *category in rows
        ],
        columns=RAW_COLUMNS,
    )
    df_enriched_articles = enrich_articles(clean_articles(df_raw), df_companies)

    return build_model(df_enriched_articles, df_companies, "tech_news.csv")


ROWS = [
    ("A1", "OpenAI", "2023-02-10", "$100M"),           # valid
    ("A2", "Open AI", "2023-03-20", "$150M"),          # valid, same quarter as A1
    ("A3", "OpenAI", "2023-05-01", "N/A"),             # undisclosed
    ("A4", "OpenAI", "2023-06-01", ""),                # missing
    ("A5", "Cohere", "2023-06-01", "$90M"),            # company not in the metadata (inferred)
    ("A6", "OpenAI", "2023-02-30", "$90M"),            # date that doesn't exist
    ("A7", "OpenAI", "2023-02-10", "lots of money"),   # value that can't be parsed
    ("A8", "Microsoft Azure", "2024-01-15", "$2B"),    # valid, resolved by alias
]

VALID_ARTICLES = {"A1", "A2", "A5", "A8"}


# Every article stays in fact_article, each excluded one with its reason
def test_fact_article_keeps_all_rows_with_exclusion_reason(df_companies):
    df_fact_article = make_tables(df_companies, ROWS)["fact_article"].set_index("article_id")

    assert len(df_fact_article) == len(ROWS)

    # Valid articles have no reason (empty), so they are shown as "" here
    assert df_fact_article["arr_exclusion_reason"].fillna("").to_dict() == {
        "A1": "", "A2": "", "A3": "undisclosed", "A4": "missing",
        "A5": "", "A6": "invalid_date", "A7": "unparseable", "A8": "",
    }


# A company that is not in the metadata gets its own row in dim_company, and its ARR is kept
def test_inferred_company_is_in_dim_company_and_has_observations(df_companies):
    tables = make_tables(df_companies, ROWS)
    df_dim_company = tables["dim_company"].set_index("company_id")

    assert len(df_dim_company) == 6
    assert df_dim_company.loc["cohere", "company_name"] == "Cohere"
    assert not df_dim_company.loc["cohere", "in_metadata"]
    assert pd.isna(df_dim_company.loc["cohere", "industry"])
    assert pd.isna(df_dim_company.loc["cohere", "employee_count"])
    assert df_dim_company.loc["openai", "in_metadata"]

    df_observation = tables["fact_arr_observation"].set_index("article_id")
    assert df_observation.loc["A5", "company_id"] == "cohere"

    # Every company in the facts exists in the dimension
    assert set(tables["fact_article"]["company_id"].dropna()) <= set(df_dim_company.index)


# A blank company name can't become an observation
def test_blank_company_name_is_excluded(df_companies):
    tables = make_tables(df_companies, [("A1", "", "2023-02-10", "$100M")])
    df_fact_article = tables["fact_article"]

    assert df_fact_article.loc[0, "arr_exclusion_reason"] == "missing_company"
    assert tables["fact_arr_observation"].empty


# Lineage columns: raw values stay next to the cleaned ones, and the source row is kept
def test_fact_article_keeps_raw_values_and_source_row(df_companies):
    df_fact_article = make_tables(df_companies, ROWS)["fact_article"].set_index("article_id")

    assert df_fact_article.loc["A2", "company_name_raw"] == "Open AI"
    assert df_fact_article.loc["A2", "company_id"] == "openai"
    assert df_fact_article.loc["A2", "revenue_raw"] == "$150M"
    assert df_fact_article.loc["A2", "source_row"] == 2
    assert df_fact_article.loc["A2", "source_file"] == "tech_news.csv"


# Only valid readings become observations, and each one links to its source article
def test_fact_arr_observation_only_has_valid_readings(df_companies):
    tables = make_tables(df_companies, ROWS)
    df_observation = tables["fact_arr_observation"]

    assert set(df_observation["article_id"]) == VALID_ARTICLES
    assert df_observation["arr_usd"].notna().all()

    # Every observation joins to exactly one article, with the same raw value
    df_joined = df_observation.merge(tables["fact_article"], on="article_id", suffixes=("", "_article"))
    assert len(df_joined) == len(df_observation)
    assert (df_joined["arr_raw"] == df_joined["revenue_raw"]).all()


# The CSV must be the same on every run, so rows are sorted by their keys
def test_tables_are_sorted_by_key(df_companies):
    tables = make_tables(df_companies, list(reversed(ROWS)))

    assert tables["fact_article"]["article_id"].is_monotonic_increasing
    assert tables["dim_company"]["company_id"].is_monotonic_increasing

    df_observation = tables["fact_arr_observation"]
    assert df_observation["company_id"].tolist() == sorted(df_observation["company_id"].tolist())


# Running the load twice gives the same rows (no duplicates)
def test_load_is_idempotent(df_companies, tmp_path):
    db_path = tmp_path / "test.duckdb"
    tables = make_tables(df_companies, ROWS)

    load_to_duckdb(tables, db_path)
    load_to_duckdb(tables, db_path)

    con = duckdb.connect(str(db_path))
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    con.close()

    assert counts == {"dim_company": 6, "fact_article": 8, "fact_arr_observation": 4}


# An observation that stops being valid is removed on the next load
def test_load_removes_observation_that_is_no_longer_valid(df_companies, tmp_path):
    db_path = tmp_path / "test.duckdb"
    load_to_duckdb(make_tables(df_companies, ROWS), db_path)

    # A2 is now "Not disclosed" in the source
    new_rows = [(a, c, d, "Not disclosed" if a == "A2" else r) for a, c, d, r in ROWS]
    load_to_duckdb(make_tables(df_companies, new_rows), db_path)

    con = duckdb.connect(str(db_path))
    observations = con.sql("SELECT article_id FROM fact_arr_observation ORDER BY article_id").fetchall()
    reason = con.execute("SELECT arr_exclusion_reason FROM fact_article WHERE article_id = 'A2'").fetchone()[0]
    con.close()

    assert observations == [("A1",), ("A5",), ("A8",)]
    assert reason == "undisclosed"


# One CSV per table, with dates as YYYY-MM-DD
def test_export_tables_writes_one_csv_per_table(df_companies, tmp_path):
    db_path = tmp_path / "test.duckdb"
    load_to_duckdb(make_tables(df_companies, ROWS), db_path)
    export_tables(db_path, tmp_path)

    assert sorted(p.name for p in tmp_path.glob("*.csv")) == [
        "dim_company.csv", "fact_arr_observation.csv", "fact_article.csv",
    ]

    df_observation = pd.read_csv(tmp_path / "fact_arr_observation.csv")
    assert df_observation["as_of_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()


# (article_id, company, published_date, revenue, category)
REPORT_ROWS = [
    ("R01", "Open AI", "2023-05-10", "$100M", "Cloud"),               # company industry is AI/ML -> in
    ("R02", "Microsoft", "2023-05-10", "$100M", "AI/ML"),             # article category is AI_ML -> in
    ("R03", "Microsoft", "2023-05-10", "$100M", "Cloud"),             # neither is AI -> out
    ("R04", "OpenAI", "2021-12-31", "$100M", "AI/ML"),                # before 2022 -> out
    ("R05", "OpenAI", "2025-01-01", "$100M", "AI/ML"),                # after 2024 -> out
    ("R06", "OpenAI", "2022-01-01", "$100M", "AI/ML"),                # first day of the period -> in
    ("R07", "OpenAI", "2024-12-31", "$100M", "AI/ML"),                # last day of the period -> in
    ("R08", "OpenAI", "2023-05-10", "$50M", "AI/ML"),                 # exactly the threshold -> out
    ("R09", "OpenAI", "2023-05-10", "$50.1M", "AI/ML"),               # just above the threshold -> in
    ("R10", "OpenAI", "2023-05-10", "N/A", "AI/ML"),                  # no valid ARR -> out
    ("R11", "Cohere", "2023-05-10", "$90M", "Machine Learning"),      # not in the metadata, AI category -> in
]

AI_ARTICLES_COLUMNS = [
    "article_id", "title", "company_name", "published_date", "category", "arr_usd", "summary", "url",
    "industry", "founded_year", "headquarters", "employee_count", "is_public", "stock_ticker",
    "company_age", "company_size_category", "embedding",
]


def make_ai_report(df_companies, tmp_path, rows=REPORT_ROWS):
    """Load the model into DuckDB and read the AI report from it."""

    db_path = tmp_path / "test.duckdb"
    load_to_duckdb(make_tables(df_companies, rows), db_path)

    return build_ai_articles_enriched(db_path)


# Category OR industry, between 2022 and 2024 (both included), valid ARR strictly above $50M
def test_ai_report_filters(df_companies, tmp_path):
    df_report = make_ai_report(df_companies, tmp_path)

    assert df_report["article_id"].tolist() == ["R01", "R02", "R06", "R07", "R09", "R11"]
    assert (df_report["arr_usd"] > 50_000_000).all()


# The columns are the ones asked in the assignment, in that order
def test_ai_report_columns(df_companies, tmp_path):
    df_report = make_ai_report(df_companies, tmp_path)

    assert list(df_report.columns) == AI_ARTICLES_COLUMNS


# The company name and the category are the standard ones, not the raw text of the article
def test_ai_report_uses_standard_company_and_category(df_companies, tmp_path):
    df_report = make_ai_report(df_companies, tmp_path).set_index("article_id")

    assert df_report.loc["R01", "company_name"] == "OpenAI"  # raw name was "Open AI"
    assert df_report.loc["R01", "category"] == "CLOUD"
    assert df_report.loc["R11", "category"] == "AI_ML"  # raw category was "Machine Learning"


# Company metadata is attached; a company that is not in the metadata keeps those columns empty
def test_ai_report_company_metadata(df_companies, tmp_path):
    df_report = make_ai_report(df_companies, tmp_path).set_index("article_id")

    assert df_report.loc["R01", ["industry", "founded_year", "company_size_category"]].tolist() == ["AI/ML", 2015, "Medium"]
    assert df_report.loc["R01", "company_age"] == 8
    assert pd.isna(df_report.loc["R11", "industry"])
    assert pd.isna(df_report.loc["R11", "company_age"])
    assert df_report.loc["R11", "company_name"] == "Cohere"


# The embedding column is part of the layout, but it is empty in every row
def test_ai_report_embedding_is_empty(df_companies, tmp_path):
    df_report = make_ai_report(df_companies, tmp_path)

    assert not df_report.empty
    assert df_report["embedding"].isna().all()


# No AI article: an empty report with the same columns, not an error
def test_ai_report_without_matching_articles(df_companies, tmp_path):
    rows = [("R01", "Microsoft", "2023-05-10", "$100M", "Cloud"), ("R02", "Microsoft", "2023-05-10", "$90M", "Cloud")]
    df_report = make_ai_report(df_companies, tmp_path, rows)

    assert df_report.empty
    assert list(df_report.columns) == AI_ARTICLES_COLUMNS


# The CSV files have every batch loaded so far, not only the last one
def test_export_has_the_rows_of_all_batches(df_companies, tmp_path):
    db_path = tmp_path / "test.duckdb"

    load_to_duckdb(make_tables(df_companies, ROWS[:4]), db_path)
    load_to_duckdb(make_tables(df_companies, ROWS[4:]), db_path)
    export_tables(db_path, tmp_path)

    df_fact_article = pd.read_csv(tmp_path / "fact_article.csv")
    df_observation = pd.read_csv(tmp_path / "fact_arr_observation.csv")

    assert df_fact_article["article_id"].tolist() == [f"A{i}" for i in range(1, 9)]

    # Sorted by company, then date: cohere (A5), microsoft (A8), openai (A1, A2)
    assert df_observation["article_id"].tolist() == ["A5", "A8", "A1", "A2"]


# Integer columns with empty values are written as integers (12793), not as floats (12793.0)
def test_export_writes_integers_without_decimals(df_companies, tmp_path):
    db_path = tmp_path / "test.duckdb"
    load_to_duckdb(make_tables(df_companies, ROWS), db_path)
    export_tables(db_path, tmp_path)

    # Cohere is not in the metadata, so employee_count has an empty value in this table
    df_dim_company = pd.read_csv(tmp_path / "dim_company.csv", dtype=str)

    assert "12793" in df_dim_company["employee_count"].tolist()
    assert not df_dim_company["employee_count"].str.contains(r"\.", na=False).any()
