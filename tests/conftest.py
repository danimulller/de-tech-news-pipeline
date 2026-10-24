import json

import pytest

from src.enrich import load_companies

# Small metadata used by the tests (not the real file)
METADATA = {
    "Amazon Web Services": {"founded_year": 2006, "headquarters": "Seattle, WA", "employee_count": 27826,
                            "industry": "Cloud Computing", "is_public": True, "stock_ticker": "AMZN"},
    "OpenAI": {"founded_year": 2015, "headquarters": "San Francisco, CA", "employee_count": 12793,
               "industry": "AI/ML", "is_public": False, "stock_ticker": None},
    "Microsoft": {"founded_year": 1975, "headquarters": "Redmond, WA", "employee_count": 48646,
                  "industry": "SaaS", "is_public": True, "stock_ticker": "MSFT"},
    "SpaceX": {"founded_year": 2002, "headquarters": "Hawthorne, CA", "employee_count": 31018,
               "industry": "Aerospace", "is_public": False, "stock_ticker": None},
    "Tesla": {"founded_year": 2003, "headquarters": "Austin, TX", "employee_count": 5691,
              "industry": "AI/ML", "is_public": True, "stock_ticker": "TSLA"},
}


@pytest.fixture
def df_companies(tmp_path):
    """Company dimension built from METADATA, shared by all tests."""

    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(METADATA))

    return load_companies(path)
