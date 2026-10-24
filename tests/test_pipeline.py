import json

import duckdb
import pandas as pd
import pytest

from src.pipeline import run_pipeline
from tests.conftest import METADATA

ARTICLES_CSV = """article_id,title,company_name,published_date,category,revenue,summary,url,author,word_count
ART1,First,OpenAI,2023-02-10,AI/ML,$100M,Summary one,https://x/1,Ann,100.0
ART2,Second,Open AI,02/23/2023,Artificial Intelligence,€2B,Summary two,https://x/2,,200.0
ART3,Third,AWS,"March 5, 2023",Cloud,N/A,Summary three,https://x/3,Bob,
ART4,Fourth,Cohere,2023-07-01,Machine Learning,$90M,Summary four,https://x/4,Eve,300.0
"""


@pytest.fixture
def data_dir(tmp_path):
    """A small data folder with the two source files."""

    path = tmp_path / "data"
    path.mkdir()
    (path / "tech_news.csv").write_text(ARTICLES_CSV)
    (path / "company_metadata.json").write_text(json.dumps(METADATA))

    return path


# The function returns the model, ready to use
def test_run_pipeline_returns_the_model(data_dir, tmp_path):
    tables = run_pipeline(data_dir, tmp_path / "output")

    assert set(tables) == {"dim_company", "fact_article", "fact_arr_observation"}
    assert len(tables["fact_article"]) == 4
    assert set(tables["fact_arr_observation"]["article_id"]) == {"ART1", "ART2", "ART4"}

    # The "N/A" revenue is kept in fact_article, with its reason
    df_fact_article = tables["fact_article"].set_index("article_id")
    assert df_fact_article.loc["ART3", "arr_exclusion_reason"] == "undisclosed"


# One CSV per table, the AI report and the DuckDB file
def test_run_pipeline_writes_all_outputs(data_dir, tmp_path):
    output_dir = tmp_path / "output"
    run_pipeline(data_dir, output_dir)

    assert sorted(p.name for p in output_dir.iterdir()) == [
        "ai_articles_enriched.csv", "dim_company.csv", "fact_arr_observation.csv", "fact_article.csv",
        "tech_news.duckdb",
    ]

    # Companies that are not in the metadata are found in dim_company
    df_dim_company = pd.read_csv(output_dir / "dim_company.csv")
    assert df_dim_company[~df_dim_company["in_metadata"]]["company_name"].tolist() == ["Cohere"]


# The AI report is built by the pipeline: ART1, ART2 and ART4 have AI categories and ARR above $50M
def test_run_pipeline_writes_ai_report(data_dir, tmp_path):
    output_dir = tmp_path / "output"
    run_pipeline(data_dir, output_dir)

    df_report = pd.read_csv(output_dir / "ai_articles_enriched.csv")

    assert df_report["article_id"].tolist() == ["ART1", "ART2", "ART4"]
    assert df_report["published_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()


# Running twice gives the same CSV files and the same rows in DuckDB (no duplicates)
def test_run_pipeline_is_idempotent(data_dir, tmp_path):
    output_dir = tmp_path / "output"
    csv_names = [
        "dim_company.csv", "fact_article.csv", "fact_arr_observation.csv", "ai_articles_enriched.csv",
    ]

    run_pipeline(data_dir, output_dir)
    first_run = {name: (output_dir / name).read_bytes() for name in csv_names}

    run_pipeline(data_dir, output_dir)
    second_run = {name: (output_dir / name).read_bytes() for name in csv_names}

    assert first_run == second_run

    con = duckdb.connect(str(output_dir / "tech_news.duckdb"))
    counts = con.execute(
        "SELECT (SELECT COUNT(*) FROM fact_article), (SELECT COUNT(*) FROM fact_arr_observation)"
    ).fetchone()
    con.close()

    assert counts == (4, 3)


# The raw file is read as text, so "N/A" is not turned into NaN before the cleaning
def test_run_pipeline_reads_na_as_text(data_dir, tmp_path):
    tables = run_pipeline(data_dir, tmp_path / "output")
    df_fact_article = tables["fact_article"].set_index("article_id")

    assert df_fact_article.loc["ART3", "revenue_raw"] == "N/A"
    assert df_fact_article.loc["ART3", "arr_parse_status"] == "undisclosed"
