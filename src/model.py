"""Warehouse tables for the ARR observations, plus the DuckDB load and the CSV export.

Tables (grain -> key):
    dim_company          one row per company                   -> company_id
    fact_article         one row per source article            -> article_id
    fact_arr_observation one row per valid ARR reading         -> article_id
    fact_article_embedding one row per source article          -> article_id

Relationships:
    fact_article.company_id         -> dim_company.company_id (null if the company was not matched)
    fact_arr_observation.article_id -> fact_article.article_id (source article of the observation)
    fact_arr_observation.company_id -> dim_company.company_id
    fact_article_embedding.article_id -> fact_article.article_id

Every article lands in fact_article, valid or not. Only articles with a valid ARR,
a valid date and a company become an observation; the reason for any exclusion is
kept in fact_article.arr_exclusion_reason.

Companies that are not in company_metadata.json also get a row in dim_company, with
in_metadata = false and no metadata (industry, founded year...), so their ARR is not lost.

How to derive the ARR views from fact_arr_observation:
ARR is a point-in-time number, so each period keeps the last reading of the period (not a sum
or an average). If a company has two readings on the same day, the highest article_id wins.

    -- Latest ARR of each company
    SELECT * FROM fact_arr_observation
    QUALIFY ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY as_of_date DESC, article_id DESC) = 1

    -- ARR of each company per quarter
    SELECT *, YEAR(as_of_date) AS year, QUARTER(as_of_date) AS quarter FROM fact_arr_observation
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY company_id, YEAR(as_of_date), QUARTER(as_of_date)
        ORDER BY as_of_date DESC, article_id DESC
    ) = 1
"""
import json

import duckdb
import numpy as np
import pandas as pd

from src.embeddings import build_article_texts, find_top_similar
from src.enrich import infer_companies

# Columns of each table, in the order they are written to the CSV files
DIM_COMPANY_COLUMNS = [
    "company_id", "company_name", "industry", "founded_year", "headquarters",
    "employee_count", "is_public", "stock_ticker", "company_size_category", "in_metadata",
]

FACT_ARTICLE_COLUMNS = [
    "article_id", "source_file", "source_row", "title",
    "company_name_raw", "company_id", "company_match_method",
    "published_date_raw", "published_date", "pub_year", "pub_quarter", "pub_month", "date_parse_status",
    "category_raw", "category_std",
    "revenue_raw", "arr_usd", "arr_parse_status", "arr_exclusion_reason",
    "summary", "url", "author", "word_count", "company_age",
]

FACT_ARR_OBSERVATION_COLUMNS = ["article_id", "company_id", "as_of_date", "arr_usd", "arr_raw"]

# Tables in the order they are loaded (the key of each one is used to replace its rows)
TABLE_KEYS = {
    "dim_company": "company_id",
    "fact_article": "article_id",
    "fact_arr_observation": "article_id",
    "fact_article_embedding": "article_id",
}

# Columns that hold a list; in the CSV files they are written as JSON, e.g. [0.1, 0.2]
LIST_COLUMNS = ["embedding", "top_similar_articles"]

# How many similar articles are kept for each article
TOP_SIMILAR_ARTICLES = 3

# AI report: articles published between these years (both included) with ARR above this value (USD)
AI_REPORT_FIRST_YEAR, AI_REPORT_LAST_YEAR = 2022, 2024
AI_REPORT_ARR_THRESHOLD_USD = 50_000_000

DDL = """
CREATE TABLE IF NOT EXISTS dim_company (
    company_id            VARCHAR PRIMARY KEY,
    company_name          VARCHAR NOT NULL,
    industry              VARCHAR,
    founded_year          INTEGER,
    headquarters          VARCHAR,
    employee_count        INTEGER,
    is_public             BOOLEAN,
    stock_ticker          VARCHAR,
    company_size_category VARCHAR,
    in_metadata           BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_article (
    article_id           VARCHAR PRIMARY KEY,
    source_file          VARCHAR NOT NULL,
    source_row           INTEGER NOT NULL,
    title                VARCHAR,
    company_name_raw     VARCHAR,
    company_id           VARCHAR,
    company_match_method VARCHAR,
    published_date_raw   VARCHAR,
    published_date       DATE,
    pub_year             INTEGER,
    pub_quarter          INTEGER,
    pub_month            INTEGER,
    date_parse_status    VARCHAR,
    category_raw         VARCHAR,
    category_std         VARCHAR,
    revenue_raw          VARCHAR,
    arr_usd              BIGINT,
    arr_parse_status     VARCHAR,
    arr_exclusion_reason VARCHAR,
    summary              VARCHAR,
    url                  VARCHAR,
    author               VARCHAR,
    word_count           INTEGER,
    company_age          INTEGER
);

CREATE TABLE IF NOT EXISTS fact_arr_observation (
    article_id VARCHAR PRIMARY KEY,
    company_id VARCHAR NOT NULL,
    as_of_date DATE    NOT NULL,
    arr_usd    BIGINT  NOT NULL,
    arr_raw    VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_article_embedding (
    article_id           VARCHAR PRIMARY KEY,
    embedding            FLOAT[] NOT NULL,
    top_similar_articles VARCHAR[] NOT NULL
);
"""


def build_dim_company(df_companies: pd.DataFrame, df_inferred_companies: pd.DataFrame) -> pd.DataFrame:
    """One row per company: the ones in the metadata plus the inferred ones.

    Sorted by company_id so the CSV is always the same.
    """

    # Inferred companies have no metadata, so those columns stay empty
    df_dim_company = pd.concat([df_companies, df_inferred_companies], ignore_index=True)
    df_dim_company = df_dim_company.reindex(columns=DIM_COMPANY_COLUMNS)

    # Types that accept empty values, otherwise the integers turn into floats
    df_dim_company = df_dim_company.astype({
        "founded_year": "Int64", "employee_count": "Int64", "is_public": "boolean", "in_metadata": "bool",
    })

    return df_dim_company.sort_values("company_id").reset_index(drop=True)


def build_fact_article(df_enriched_articles: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """One row per source article, with raw and cleaned values side by side."""

    df_fact_article = df_enriched_articles.rename(columns={
        "company_name": "company_name_raw",
        "published_date": "published_date_raw",
        "published_dt": "published_date",
        "category": "category_raw",
        "revenue": "revenue_raw",
        "match_method": "company_match_method",
    })

    # Where the article came from: file name and row number (the header is not counted)
    df_fact_article["source_file"] = source_file
    df_fact_article["source_row"] = np.arange(1, len(df_fact_article) + 1)

    # Why the article does NOT become an ARR observation (empty if it does)
    # The checks run in order: the first one that is true wins
    df_fact_article["arr_exclusion_reason"] = np.select(
        [
            df_fact_article["arr_parse_status"] != "ok",
            df_fact_article["company_id"].isna(),
            df_fact_article["published_date"].isna(),
        ],
        [
            df_fact_article["arr_parse_status"],
            "missing_company",
            "invalid_date",
        ],
        default="",
    )
    df_fact_article["arr_exclusion_reason"] = df_fact_article["arr_exclusion_reason"].replace("", pd.NA)

    df_fact_article = df_fact_article[FACT_ARTICLE_COLUMNS].sort_values("article_id")

    return df_fact_article.reset_index(drop=True)


def build_fact_arr_observation(df_fact_article: pd.DataFrame) -> pd.DataFrame:
    """One row per valid ARR reading: a company, a valid date and a parsed value."""

    # Only articles without an exclusion reason are valid observations
    df_valid = df_fact_article[df_fact_article["arr_exclusion_reason"].isna()]

    # The article_id is the link back to the source article
    df_fact_arr_observation = pd.DataFrame({
        "article_id": df_valid["article_id"],
        "company_id": df_valid["company_id"],
        "as_of_date": df_valid["published_date"],
        "arr_usd": df_valid["arr_usd"],
        "arr_raw": df_valid["revenue_raw"],
    })

    df_fact_arr_observation = df_fact_arr_observation.sort_values(["company_id", "as_of_date", "article_id"])

    return df_fact_arr_observation.reset_index(drop=True)


def build_fact_article_embedding(df_fact_article: pd.DataFrame, embed) -> pd.DataFrame:
    """One row per article: the embedding of title + summary and the ids of the 3 most similar articles.

    embed is the function that turns a list of texts into vectors (see src/embeddings.py).
    """

    article_ids = df_fact_article["article_id"].tolist()

    # The vectors are normalized, so the cosine similarity between two articles is a dot product
    embeddings = embed(build_article_texts(df_fact_article))
    top_similar_articles = find_top_similar(embeddings, article_ids, TOP_SIMILAR_ARTICLES)

    # Rounded to 6 decimals to keep the files small and equal on every run
    return pd.DataFrame({
        "article_id": article_ids,
        "embedding": [[round(float(value), 6) for value in vector] for vector in embeddings],
        "top_similar_articles": top_similar_articles,
    })


def build_model(
    df_enriched_articles: pd.DataFrame, df_companies: pd.DataFrame, source_file: str, embed
) -> dict:
    """Build the four tables, as a dict of table name -> DataFrame."""

    df_fact_article = build_fact_article(df_enriched_articles, source_file)

    # Companies found in the articles but not in the metadata
    df_inferred_companies = infer_companies(df_enriched_articles)

    return {
        "dim_company": build_dim_company(df_companies, df_inferred_companies),
        "fact_article": df_fact_article,
        "fact_arr_observation": build_fact_arr_observation(df_fact_article),
        "fact_article_embedding": build_fact_article_embedding(df_fact_article, embed),
    }


def load_to_duckdb(tables: dict, db_path: str) -> None:
    """Load the tables into DuckDB without creating duplicates, no matter how many times it runs.

    For each table, the rows with the same key as the new batch are deleted and the
    batch is inserted again. This also removes stale rows: an article that used to be
    a valid observation and is not anymore (e.g. after a fix in the parser) disappears.
    """

    con = duckdb.connect(str(db_path))

    try:
        con.execute(DDL)

        # Everything is in one transaction: either all tables are updated or none
        con.execute("BEGIN")

        for table, key in TABLE_KEYS.items():
            df_batch = tables[table]
            con.register("df_batch", df_batch)

            con.execute(f"DELETE FROM {table} WHERE {key} IN (SELECT {key} FROM df_batch)")
            con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM df_batch")

            con.unregister("df_batch")

        # Fact article rows that stop being observations must be removed from the observation table
        # (their key is not in the new observation batch, so the delete above did not touch them)
        con.execute(
            "DELETE FROM fact_arr_observation WHERE article_id IN "
            "(SELECT article_id FROM fact_article WHERE arr_exclusion_reason IS NOT NULL)"
        )

        con.execute("COMMIT")

    except Exception:
        con.execute("ROLLBACK")
        raise

    finally:
        con.close()


def export_tables(tables: dict, output_dir: str) -> None:
    """Write each table to its own CSV file (output_dir/<table name>.csv)."""

    for table, df_table in tables.items():
        df_table.pipe(lists_to_json).to_csv(f"{output_dir}/{table}.csv", index=False, date_format="%Y-%m-%d")


def lists_to_json(df_table: pd.DataFrame) -> pd.DataFrame:
    """Copy of the table with the list columns written as JSON text, so the CSV can be read back."""

    df_json = df_table.copy()

    for column in LIST_COLUMNS:
        if column in df_json:
            df_json[column] = df_json[column].map(lambda values: json.dumps([_to_python(v) for v in values]))

    return df_json


def _to_python(value):
    """numpy numbers come out of DuckDB; json only accepts plain python numbers and text."""

    return value.item() if isinstance(value, np.generic) else value


def build_ai_articles_enriched(db_path: str) -> pd.DataFrame:
    """The ai_articles_enriched report, read from the DuckDB tables (nothing is computed again).

    An article is in the report if its category OR the industry of its company is AI / ML,
    it was published between 2022 and 2024, and it has a valid ARR above $50M.
    A valid ARR is a row in fact_arr_observation.
    """

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        df_ai_articles = con.execute("""
            SELECT art.article_id, art.title, com.company_name, art.published_date, art.category_std as category,
                art.arr_usd, art.summary, art.url, com.industry, com.founded_year, com.headquarters,
                com.employee_count, com.is_public, com.stock_ticker, art.company_age, com.company_size_category,
                emb.embedding, emb.top_similar_articles
            FROM fact_arr_observation obs
            INNER JOIN fact_article art ON art.article_id = obs.article_id
            INNER JOIN dim_company com ON com.company_id = obs.company_id
            INNER JOIN fact_article_embedding emb ON emb.article_id = obs.article_id
            WHERE art.category_std = 'AI_ML'
              AND YEAR(obs.as_of_date) BETWEEN ? AND ?
              AND obs.arr_usd > ?
            ORDER BY art.article_id
        """, [AI_REPORT_FIRST_YEAR, AI_REPORT_LAST_YEAR, AI_REPORT_ARR_THRESHOLD_USD]).df()

    finally:
        con.close()

    return lists_to_json(df_ai_articles)
