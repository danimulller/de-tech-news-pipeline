"""Runs the whole pipeline: read, clean, enrich, model, export, load into DuckDB and build the AI report.

Usage:
    python -m src.pipeline
    python -m src.pipeline --data-dir data --output-dir data/output
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.cleaning import clean_articles
from src.embeddings import embed_texts
from src.enrich import enrich_articles, load_companies
from src.model import build_ai_articles_enriched, build_model, export_tables, load_to_duckdb

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTICLES_FILE = "tech_news.csv"
COMPANIES_FILE = "company_metadata.json"
DB_FILE = "tech_news.duckdb"
AI_REPORT_FILE = "ai_articles_enriched.csv"


def read_articles(path: Path) -> pd.DataFrame:
    """Read the raw articles as text, exactly as they are in the file.

    Everything is read as string and nothing is turned into NaN by pandas, because
    values like "N/A" and "" have a meaning for the cleaning step.
    """

    return pd.read_csv(path, dtype=str, keep_default_na=False)


def run_pipeline(
    data_dir: Path = PROJECT_ROOT / "data",
    output_dir: Path = PROJECT_ROOT / "data" / "output",
    db_path: Path | None = None,
    embed=embed_texts,
) -> dict:
    """Run every step and return the model: a dict of table name -> DataFrame.

    Side effects: one CSV per table and ai_articles_enriched.csv in output_dir
    (data/output by default), and the tables loaded into the DuckDB file at db_path
    (output_dir/tech_news.duckdb by default).

    embed is the function that turns a list of texts into vectors. The default uses
    sentence-transformers; tests pass a fake one so they don't need the model.

    Running it again on the same data gives the same files and no duplicates.
    """

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    db_path = Path(db_path) if db_path else output_dir / DB_FILE
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the two source files
    df_raw = read_articles(data_dir / ARTICLES_FILE)
    df_companies = load_companies(data_dir / COMPANIES_FILE)
    logger.info("Read %d articles and %d companies from the metadata", len(df_raw), len(df_companies))

    # Clean the values (ARR, dates, category) and add the company metadata
    df_clean_articles = clean_articles(df_raw)
    df_enriched_articles = enrich_articles(df_clean_articles, df_companies)

    # Build the warehouse tables (the embeddings are part of the model)
    tables = build_model(df_enriched_articles, df_companies, ARTICLES_FILE, embed)

    # Write the CSV files and load DuckDB
    export_tables(tables, output_dir)
    load_to_duckdb(tables, db_path)

    # The AI report is a query on the DuckDB tables
    df_ai_articles = build_ai_articles_enriched(db_path)
    df_ai_articles.to_csv(output_dir / AI_REPORT_FILE, index=False, date_format="%Y-%m-%d")
    logger.info("%-22s %5d rows", "ai_articles_enriched", len(df_ai_articles))

    log_summary(tables)

    return tables


def log_summary(tables: dict) -> None:
    """Log how many rows each table has and why articles were left out of the ARR table."""

    for table, df_table in tables.items():
        logger.info("%-22s %5d rows", table, len(df_table))

    reasons = tables["fact_article"]["arr_exclusion_reason"].value_counts().to_dict()
    logger.info("Articles without an ARR observation: %s", reasons or "none")

    # Companies that are not in company_metadata.json (dim_company.in_metadata = false)
    logger.info("Companies not in the metadata: %d", (~tables["dim_company"]["in_metadata"]).sum())


def main() -> None:
    """Command line entry point."""

    parser = argparse.ArgumentParser(description="Run the tech news ARR pipeline.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # The libraries that download the model log every request, which hides our own messages
    for noisy_logger in ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    run_pipeline(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
