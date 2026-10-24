# Data Architecture

How the pipeline turns 750 messy news articles into queryable company ARR observations,
and how it could run reliably beyond a local batch.

## 1. Overview

Every valid `revenue` value is treated as a **reported ARR observation** of the article's
company, as of the article's published date, converted to USD.

```
 tech_news.csv ──────────┐
                         ├─► clean ─► enrich ─► model ─► 3 tables (CSV + DuckDB)
 company_metadata.json ──┘                              ai_articles_enriched.csv
```

- `src/cleaning.py`: parses ARR, dates and categories. Parsers never raise, they return `(value, status)`.
- `src/enrich.py`: matches company names to the metadata and adds industry, age and size.
- `src/model.py`: builds the tables, loads DuckDB, exports the CSVs and the AI report.
- `src/pipeline.py`: `run_pipeline()` runs everything in order and returns the tables.

## 2. Data model

| Table | One row per... | Key | Rows |
|---|---|---|---|
| `dim_company` | company | `company_id` | 26 |
| `fact_article` | source article, valid or not | `article_id` | 750 |
| `fact_arr_observation` | valid ARR reading | `article_id` | 558 |

```
 dim_company ◄── company_id ── fact_article ◄── article_id ── fact_arr_observation
      ▲                                                               │
      └──────────────────────── company_id ───────────────────────────┘
```

The relations are documented, not enforced as foreign keys: DuckDB refuses to delete and
re-insert a row that another table references, which would break re-runs.

- **`dim_company`**: companies from the metadata file (`in_metadata = true`, with industry,
  founded year, headquarters, employees, public/private, ticker and size category). Companies
  that appear in articles but not in the metadata (5 today) also get a row, with
  `in_metadata = false` and empty metadata, so their ARR is kept. Their `company_id` is a slug
  of the name (`Mistral AI` is `mistral_ai`), so it does not change if they join the metadata later.
  Size: `Small` below 10,000 employees, `Medium` up to 30,000, `Large` above.
- **`fact_article`**: keeps the **raw value next to the cleaned one** (`revenue_raw` / `arr_usd`,
  `published_date_raw` / `published_date`, `category_raw` / `category_std`,
  `company_name_raw` / `company_id`), a status column for each cleaning step, `source_file`,
  `source_row`, `company_age` and `arr_exclusion_reason`.
- **`fact_arr_observation`**: `article_id`, `company_id`, `as_of_date`, `arr_usd` and `arr_raw`.
  An article becomes an observation only with a parsed ARR, a valid date and a company.
  Everything else stays in `fact_article` with the reason in `arr_exclusion_reason`.

## 3. Handling messy data

Nothing is silently dropped or invented. Every parser returns a value and a status.

**ARR.** Formats like `5.2B`, `$5,200,000,000`, `5.2 billion` and `500M USD` are parsed with
`Decimal` into an integer. EUR x 1.1, GBP x 1.27, JPY / 150. A range becomes its midpoint.

| `arr_parse_status` | Meaning | Articles |
|---|---|---|
| `ok` | Parsed, becomes an observation | 558 |
| `undisclosed` | `N/A`, `Not disclosed` | 126 |
| `missing` | Empty or null | 66 |
| `unparseable` | Not an amount | 0 |

**Dates.** ISO, ISO with time, `21 Feb 2020`, `October 19, 2022` and numeric dates are read.
In `NN/NN/YYYY` and `NN-NN-YYYY`, a number above 12 settles which one is the day. When both
are 12 or below (`03/04/2023`) the date is ambiguous, and the separator decides, based on the
unambiguous rows of this dataset: `/` is MM/DD (the first number is never above 12), and `-` is
DD-MM (the first number is above 12 in 63 rows, the second in 17). These rows are flagged:
657 `ok`, 56 `ambiguous_assumed_eu`, 37 `ambiguous_assumed_us`.

**Categories.** `CATEGORY_MAP` maps the 19 raw values to 6 (`AI_ML`, `CLOUD`, `DATA_ANALYTICS`,
`FINTECH`, `SECURITY`, `SOFTWARE`). Unmapped values become `OTHER`, empty ones `UNKNOWN`.

**Companies.** The raw name is matched in this order (`company_match_method`):

| Method | How | Articles |
|---|---|---|
| `exact` | Same name once normalized (`Mongo DB` = `MongoDB`, `Databricks Inc.` = `Databricks`) | 704 |
| `alias` | Listed in `COMPANY_ALIASES` (`AWS`, `Azure`, `DeepMind`...) | 21 |
| `fuzzy` | Very similar to an official name (score of 90+ out of 100), catches typos | 0 |
| `inferred` | Not in the metadata, gets its own `company_id` | 25 |
| `unmatched` | Blank name, no company can be created | 0 |

## 4. Lineage and queries

An observation points to its article by `article_id`, and the article points to the exact row
of the source file (`source_file`, `source_row`), next to the original text.

```sql
-- Source article of an ARR observation
SELECT o.article_id, o.as_of_date, o.arr_usd, a.revenue_raw, a.company_name_raw,
       a.title, a.url, a.source_file, a.source_row
FROM fact_arr_observation AS o
JOIN fact_article AS a ON a.article_id = o.article_id
WHERE o.article_id = 'ART0102';

-- ARR of a company over time (the main use case)
SELECT c.company_name, o.as_of_date, o.arr_usd, o.article_id
FROM fact_arr_observation AS o
JOIN dim_company AS c ON c.company_id = o.company_id
WHERE o.company_id = 'openai'
ORDER BY o.as_of_date;

-- Companies that are not in the metadata
SELECT * FROM dim_company WHERE NOT in_metadata;
```

**Latest and quarterly ARR.** ARR is a point-in-time number, so each period keeps **the last
reading of the period** (not a sum or an average). On a tie, the highest `article_id` wins
(6 company-days have two readings, always with the same value).

```sql
-- Latest ARR of each company
SELECT * FROM fact_arr_observation
QUALIFY ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY as_of_date DESC, article_id DESC) = 1;

-- ARR of each company per quarter
SELECT *, YEAR(as_of_date) AS year, QUARTER(as_of_date) AS quarter
FROM fact_arr_observation
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY company_id, YEAR(as_of_date), QUARTER(as_of_date)
    ORDER BY as_of_date DESC, article_id DESC
) = 1;
```

Filtering by date, category, industry and ARR is a join of `fact_article` and `dim_company`
(`category_std`, `published_date`, `industry`, `arr_usd`).

## 5. Re-runs and the AI report

**Idempotent load.** For each table, the rows with the same key as the new batch are deleted and
inserted again, in one transaction. The same data gives the same rows and byte-identical CSVs.
It also removes stale rows: if an article stops being a valid observation (after a parser fix, for
example), its old observation disappears. A plain upsert would leave it behind.

**`ai_articles_enriched.csv`** is a query on the tables, not a separate computation. It keeps
articles whose category is `AI_ML` **or** whose company industry is `AI/ML`, with a valid ARR above
$50M, dated 2022 to 2024. It has 124 rows (83 by category, 41 only by industry).
The `embedding` column is part of the layout and is always empty.

## 6. Assumptions and caveats

- **ARR from news is not company master data.** It is a claim made by one article on one day,
  so it lives in an observations table with its source, not as a column of `dim_company`.
- **Industry is used as given.** The metadata is synthetic (Anthropic is labeled `FinTech`).
- **Judgment calls**, visible as `alias`: Azure resolves to Microsoft, and
  "The Boring Company / SpaceX" to SpaceX. The raw name is always kept.
- **Companies outside the metadata** have no industry or size, so they reach the AI report
  only through their article category.
- **93 dates are guessed** (flagged `ambiguous_assumed_*`) and **FX rates are fixed**, as in the assignment.
- **Out of scope:** the bonus part (embeddings, similarity and hybrid search).

## 7. Running beyond a local batch

Today it is one Python process, two input files and a local DuckDB file. At scale:

- **Raw layer.** Land each batch as-is in object storage, partitioned by ingest date, and never edit it.
  Any table can be rebuilt from it. `source_file` and `source_row` already point to the raw record.
- **Retries and backfills.** Loading by key makes retries and duplicate batches harmless, and the newest
  version of an `article_id` wins. To fix a parser or an alias, change the code and re-run: wrong
  observations are replaced and stale ones removed. At scale, reprocess by date partition.
- **Schema changes.** Add new source columns to the raw layer first, then to the model, with versioned
  migrations (`CREATE TABLE IF NOT EXISTS` does not alter existing tables). Check the source schema before
  cleaning and quarantine a batch that breaks it. New values (a currency, a category) are already safe:
  they become `unparseable` or `OTHER` instead of crashing the run.
- **Orchestration and storage.** Run the steps as retryable tasks (Airflow, Dagster or Prefect) and swap
  the DuckDB file for a warehouse or Parquet on object storage. The model and the SQL stay the same.
- **Monitoring.** The status columns give the checks for free: alert on the share of `unparseable`,
  `invalid`, `unmatched`, `ambiguous_*` and `inferred` rows, on row counts against the previous run, and on
  the same company-day with different values.
- **Testing.** 90 tests cover the parsers, matching, model, idempotency and the report. Outputs are
  deterministic, so a regression shows up as a file diff, and `requirements.txt` pins every version.
