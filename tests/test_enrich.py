import json

import pandas as pd
import pytest

from src.enrich import (
    enrich_articles,
    infer_companies,
    load_companies,
    normalize_company_name,
    resolve_companies,
    size_category,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Mongo DB", "mongodb"),
        ("OpenAI Inc.", "openai"),
        ("Amazon Web Services (AWS)", "amazonwebservices"),
        ("Palantir Technologies", "palantir"),
        ("NVIDIA Corporation", "nvidia"),
    ],
)
# Legal suffixes, parentheses and spaces don't change the key
def test_normalize_company_name(raw, expected):
    assert normalize_company_name(raw) == expected


# Without the employee count there is no category (not "Large")
def test_size_category_without_employee_count():
    assert size_category(None) is None
    assert size_category(float("nan")) is None


@pytest.mark.parametrize(
    "employees, expected",
    [(9_999, "Small"), (10_000, "Medium"), (30_000, "Medium"), (30_001, "Large")],
)
# Checks the edges: 10,000 and 30,000 are Medium
def test_size_category_boundaries(employees, expected):
    assert size_category(employees) == expected


# One row per company, with company_id and size category
def test_load_companies(df_companies):

    # The fixture itself comes from load_companies (see conftest.py)
    assert df_companies["company_id"].tolist() == ["amazon_web_services", "openai", "microsoft", "spacex", "tesla"]
    assert df_companies.set_index("company_id")["company_size_category"].to_dict() == {
        "amazon_web_services": "Medium", "openai": "Medium", "microsoft": "Large", "spacex": "Large", "tesla": "Small",
    }


# Each name is resolved by exact match, alias, fuzzy match or inferred (not in the metadata)
def test_resolve_companies_methods(df_companies):
    names = ["OpenAI", "Open AI", "AWS", "Microsoft Azure", "Opn AI", "Cohere"]
    df_resolved = resolve_companies(names, df_companies).set_index("company_name_raw")

    assert df_resolved.loc["OpenAI", "match_method"] == "exact"
    assert df_resolved.loc["Open AI", "company_id"] == "openai"
    assert df_resolved.loc["AWS", ["company_id", "match_method"]].tolist() == ["amazon_web_services", "alias"]
    assert df_resolved.loc["Microsoft Azure", "company_id"] == "microsoft"
    assert df_resolved.loc["Opn AI", ["company_id", "match_method"]].tolist() == ["openai", "fuzzy"]
    assert df_resolved.loc["Cohere", ["company_id", "match_method"]].tolist() == ["cohere", "inferred"]


# "The Boring Company / SpaceX" is linked to SpaceX by alias
def test_boring_company_spacex_resolves_to_spacex(df_companies):
    df_resolved = resolve_companies(["The Boring Company / SpaceX", "SpaceX"], df_companies).set_index("company_name_raw")

    assert df_resolved.loc["The Boring Company / SpaceX", ["company_id", "match_method"]].tolist() == ["spacex", "alias"]
    assert df_resolved.loc["SpaceX", ["company_id", "match_method"]].tolist() == ["spacex", "exact"]


# Keeps every article, keeps the raw name and adds metadata and company_age
def test_enrich_articles(df_companies):
    df_clean_articles = pd.DataFrame(
        {
            "article_id": ["A1", "A2", "A3"],
            "company_name": ["AWS", "Open AI", "Cohere"],
            "pub_year": pd.array([2023, 2022, 2024], dtype="Int64"),
        }
    )
    df_enriched_articles = enrich_articles(df_clean_articles, df_companies)

    assert df_enriched_articles["article_id"].tolist() == ["A1", "A2", "A3"]  # no rows added or lost
    assert df_enriched_articles["company_name"].tolist() == ["AWS", "Open AI", "Cohere"]  # raw name kept
    assert df_enriched_articles["company_age"].tolist()[:2] == [17, 7]
    assert df_enriched_articles["industry"].tolist()[:2] == ["Cloud Computing", "AI/ML"]

    # Cohere is not in the metadata: it has its own company_id, but no metadata
    assert df_enriched_articles["company_id"].tolist() == ["amazon_web_services", "openai", "cohere"]
    assert pd.isna(df_enriched_articles["company_age"].iloc[2])
    assert pd.isna(df_enriched_articles["industry"].iloc[2])


# Spellings of the same company that is not in the metadata share one company_id
def test_inferred_company_spellings_share_one_id(df_companies):
    df_resolved = resolve_companies(["Hugging Face", "HuggingFace", "Hugging Face Inc."], df_companies)

    assert df_resolved["company_id"].tolist() == ["hugging_face"] * 3
    assert set(df_resolved["match_method"]) == {"inferred"}


# The company_id of an inferred company doesn't depend on the order of the names
def test_inferred_company_id_does_not_depend_on_order(df_companies):
    df_first = resolve_companies(["HuggingFace", "Hugging Face"], df_companies)
    df_second = resolve_companies(["Hugging Face", "HuggingFace"], df_companies)

    assert set(df_first["company_id"]) == set(df_second["company_id"]) == {"hugging_face"}


# A blank name has no company to create, so it stays unmatched
def test_blank_company_name_stays_unmatched(df_companies):
    df_resolved = resolve_companies(["", "OpenAI"], df_companies).set_index("company_name_raw")

    assert df_resolved.loc["", "match_method"] == "unmatched"
    assert pd.isna(df_resolved.loc["", "company_id"])


# One row per inferred company, with the first spelling as its name
def test_infer_companies(df_companies):
    df_clean_articles = pd.DataFrame(
        {
            "article_id": ["A1", "A2", "A3", "A4"],
            "company_name": ["Hugging Face", "HuggingFace", "Cohere", "OpenAI"],
            "pub_year": pd.array([2023] * 4, dtype="Int64"),
        }
    )
    df_inferred = infer_companies(enrich_articles(df_clean_articles, df_companies))

    assert df_inferred.to_dict("records") == [
        {"company_id": "cohere", "company_name": "Cohere", "in_metadata": False},
        {"company_id": "hugging_face", "company_name": "Hugging Face", "in_metadata": False},
    ]


# Two names that give the same company_id are an error, not a silent overwrite
def test_load_companies_rejects_repeated_company_id(tmp_path):
    metadata = {"Scale AI": {"employee_count": 100}, "Scale-AI": {"employee_count": 200}}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="same company_id"):
        load_companies(path)


# A company without the employee count is loaded, with an empty size category
def test_load_companies_without_employee_count(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"Acme": {"employee_count": None, "industry": "SaaS"}}))

    df_companies = load_companies(path)

    assert pd.isna(df_companies.loc[0, "company_size_category"])
