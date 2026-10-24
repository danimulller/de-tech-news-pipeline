"""Company name resolution and metadata enrichment."""
import json
import re

import pandas as pd
from rapidfuzz import fuzz, process

# Minimum similarity (0-100) to accept a fuzzy match
FUZZY_THRESHOLD = 90

# Words dropped from company names before comparing them ("Databricks Inc." -> "databricks")
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "technologies", "ltd", "llc", "co"}

# Variations that string similarity can't resolve, keyed by normalized name.
# Azure is a Microsoft product, so its articles resolve to Microsoft.
# "The Boring Company / SpaceX" is also resolved to SpaceX.
# The raw company_name is kept on the article for audit.
COMPANY_ALIASES = {
    "aws": "Amazon Web Services",
    "azure": "Microsoft",
    "microsoftazure": "Microsoft",
    "deepmind": "Google DeepMind",
    "facebookairesearch": "Meta AI",
    "metaairesearch": "Meta AI",
    "theboringcompanyspacex": "SpaceX",
}


def normalize_company_name(name: str) -> str:
    """Lowercase, drop parentheticals, punctuation and legal suffixes, then squash spaces.

    'Mongo DB' and 'MongoDB' end up with the same key.
    """

    # Removes text between parentheses ("Amazon Web Services (AWS)" -> "amazon web services")
    name = re.sub(r"\(.*?\)", " ", str(name).lower())

    # Keeps only letters and numbers, then splits into words
    words = re.sub(r"[^a-z0-9 ]", " ", name).split()

    # Drops legal suffixes and joins the words without spaces
    return "".join(word for word in words if word not in LEGAL_SUFFIXES)


def company_id_from_name(name: str) -> str:
    """Make a slug to use as company_id ("Scale AI" -> "scale_ai")."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def size_category(employee_count: int) -> str:
    """Small below 10,000 employees, Medium from 10,000 to 30,000, Large above 30,000."""

    if employee_count < 10_000:
        return "Small"

    if employee_count <= 30_000:
        return "Medium"

    return "Large"


def load_companies(path: str) -> pd.DataFrame:
    """Build the company dimension, one row per company in the metadata file."""

    with open(path) as f:
        metadata = json.load(f)

    # The json keys are the company names, so they become a column
    df_companies = pd.DataFrame.from_dict(metadata, orient="index")
    df_companies.index.name = "company_name"
    df_companies = df_companies.reset_index()

    # Creates company_id for JOIN with .csv
    df_companies["company_id"] = df_companies["company_name"].map(company_id_from_name)

    df_companies["company_size_category"] = df_companies["employee_count"].map(size_category)

    # Every company here comes from company_metadata.json (inferred ones are added later)
    df_companies["in_metadata"] = True

    return df_companies


def resolve_companies(names, df_companies: pd.DataFrame) -> pd.DataFrame:
    """Match raw company names to the dimension.

    Tries, in order: exact normalized match, alias, fuzzy match.
    
    Names that match nothing are not in company_metadata.json, so they get their own
    company_id and "inferred" in the "match_method" column. Spellings of the same name
    ("Hugging Face" and "HuggingFace") share one company_id.

    Only a blank name keeps a null company_id ("unmatched").
    """

    # Lookups: normalized name -> company_id, and official name -> company_id
    names_and_ids = list(zip(df_companies["company_name"], df_companies["company_id"]))
    id_by_key = {normalize_company_name(name): company_id for name, company_id in names_and_ids}
    id_by_name = dict(names_and_ids)

    rows = []

    # Inferred companies: normalized name -> company_id
    inferred_id_by_key = {}

    # Each distinct raw name is resolved once, not once per article
    # Sorted, so the company_id of an inferred company doesn't depend on the order of the articles
    for name in sorted(pd.unique(pd.Series(names))):
        key = normalize_company_name(name)

        # Same name after normalizing ("Open AI" -> "openai")
        if key in id_by_key:
            company_id, method = id_by_key[key], "exact"

        # Known variation listed in COMPANY_ALIASES ("AWS")
        elif key in COMPANY_ALIASES:
            company_id, method = id_by_name[COMPANY_ALIASES[key]], "alias"

        # Fuzzy match: compares the key with every official key and gets the most similar one
        # Similarity goes from 0 (nothing in common) to 100 (identical)
        # It is only accepted if the score is at least FUZZY_THRESHOLD (90), to avoid wrong matches
        # Example: "Opn AI" -> "opnai" is 90.9 similar to "openai" (one letter missing), so it matches OpenAI
        # extractOne returns (company_id, score, key), or None if no key reaches the threshold
        elif best := process.extractOne(key, id_by_key, scorer=fuzz.ratio, score_cutoff=FUZZY_THRESHOLD):
            company_id, method = best[0], "fuzzy"

        # Not in the metadata: the company gets its own company_id (slug of the name)
        # The first spelling (in alphabetical order) defines the id, the others reuse it
        # If the company is added to the metadata later, the slug is the same, so the id doesn't change
        elif key:
            company_id = inferred_id_by_key.setdefault(key, company_id_from_name(name))
            method = "inferred"

        # Blank name: there is no company to create
        else:
            company_id, method = None, "unmatched"

        rows.append({"company_name_raw": name, "company_id": company_id, "match_method": method})

    return pd.DataFrame(rows)


def enrich_articles(df_clean_articles: pd.DataFrame, df_companies: pd.DataFrame) -> pd.DataFrame:
    """Add company_id, match_method and the company metadata to each article.
    """

    # Resolves each raw name and attaches the result to every article (LEFT JOIN keeps all articles)
    df_resolved = resolve_companies(df_clean_articles["company_name"], df_companies)
    df_enriched_articles = df_clean_articles.merge(
        df_resolved.rename(columns={"company_name_raw": "company_name"}), on="company_name", how="left"
    )

    # Attaches the company metadata (industry, founded_year, employees...)
    # Inferred companies are not in df_companies, so their metadata columns stay empty
    df_enriched_articles = df_enriched_articles.merge(
        df_companies.drop(columns=["company_name", "in_metadata"]), on="company_id", how="left"
    )

    # Age of the company when the article was published
    # Year of Article Published - Foundation Year (came from company_metadata.json) = Company Age
    df_enriched_articles["company_age"] = (
        df_enriched_articles["pub_year"] - df_enriched_articles["founded_year"]
    ).astype("Int64")

    return df_enriched_articles


def infer_companies(df_enriched_articles: pd.DataFrame) -> pd.DataFrame:
    """Companies that are in the articles but not in company_metadata.json, one row per company_id.

    They have no metadata, so only company_id and company_name are filled.
    """

    # Articles whose company was not found in the metadata
    df_inferred = df_enriched_articles[df_enriched_articles["match_method"] == "inferred"]

    # One row per company_id; the first spelling in alphabetical order is the company_name
    df_inferred = df_inferred.groupby("company_id", as_index=False).agg(company_name=("company_name", "min"))
    df_inferred["in_metadata"] = False

    return df_inferred
