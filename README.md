# Tech News ARR Pipeline

A local Python pipeline that turns messy technology-news articles into:

- **Warehouse tables** with company ARR (annual recurring revenue) observations over time, one CSV file per table
  and a DuckDB database.
- **`ai_articles_enriched.csv`**, a report of AI-related articles with company metadata.

Every valid `revenue` value is treated as a reported ARR observation of the article's company, as of the
article's published date, converted to USD. See [ARCHITECTURE.md](ARCHITECTURE.md) for the data model and the design decisions.

## System requirements

| | |
|---|---|
| OS | Linux or WSL 2 (Ubuntu 24.04 recommended) |
| Python | **3.12 or newer** (the pinned `numpy` needs it) |
| Disk | About 300 MB for the dependencies |
| Memory | Under 1 GB. No GPU |
| Time | A few seconds per run |
| Internet | Only to install the dependencies. The pipeline runs offline |

## Installation and running

Run these steps in order, in a Linux or WSL 2 terminal. Steps 1 and 2 are needed only once.

1. Check that Python is 3.12 or newer:

   ```bash
   python3 --version
   ```

   If it is older (Ubuntu 22.04 comes with Python 3.10), install Python 3.12 or newer before going on.

2. Install git and the Python venv package (Ubuntu and WSL; skip it if they are already installed):

   ```bash
   sudo apt update && sudo apt install -y git python3 python3-venv
   ```

3. Clone the repository and enter the project folder:

   ```bash
   git clone https://github.com/danimulller/de-tech-news-pipeline.git
   cd de-tech-news-pipeline
   ```

4. Create and activate the virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install the dependencies (all versions are pinned):

   ```bash
   pip install -r requirements.txt
   ```

6. Run the pipeline. The files are written to `data/output/`:

   ```bash
   python -m src.pipeline
   ```

   Options: `--data-dir` (folder with the two input files, default `data`) and `--output-dir`
   (default `data/output`).

7. Run the tests:

   ```bash
   python -m pytest -q
   ```

In a new terminal, go to the project folder and run `source .venv/bin/activate` again before steps 6 and 7.

## Regenerating the outputs

Running the pipeline again rebuilds every file, and gives **identical results** for the same input.
It is safe to run as many times as you want: the load replaces rows by key, so nothing is duplicated.

The CSV files are always overwritten. To also start from an empty database, delete it first:

```bash
rm data/output/tech_news.duckdb
python -m src.pipeline
```

| File in `data/output/` | What it is | Rows |
|---|---|---|
| `dim_company.csv` | One row per company, with metadata | 26 |
| `fact_article.csv` | One row per source article, raw and cleaned values side by side | 750 |
| `fact_arr_observation.csv` | One row per valid ARR reading (the ARR over time table) | 558 |
| `ai_articles_enriched.csv` | AI-related articles, 2022 to 2024, ARR above $50M | 124 |
| `tech_news.duckdb` | The three tables in a DuckDB database | |

## Using the results

### Query the DuckDB database

```bash
python - <<'EOF'
import duckdb

con = duckdb.connect("data/output/tech_news.duckdb", read_only=True)

# ARR of a company over time
print(con.sql("""
    SELECT as_of_date, arr_usd, article_id
    FROM fact_arr_observation
    WHERE company_id = 'openai'
    ORDER BY as_of_date
""").df().head())
EOF
```

More queries (source article of an observation, latest ARR, quarterly ARR, filters) are in
[ARCHITECTURE.md](ARCHITECTURE.md#4-lineage-and-queries).

### Use the model from Python

```python
from src.pipeline import run_pipeline

tables = run_pipeline()                    # runs everything and writes the files
tables["fact_arr_observation"].head()      # the tables are pandas DataFrames
```

### Notebooks (optional)

The `notebooks/` folder has exploration notebooks. To use them, install the extra dependency and pick the
virtual environment as the kernel:

```bash
pip install -r requirements-dev.txt
```

## Project layout

```
data/
  tech_news.csv              input: 750 articles
  company_metadata.json      input: company metadata
  output/                    generated files (see the table above)
src/
  cleaning.py                ARR, date and category parsing
  enrich.py                  company matching and metadata
  model.py                   tables, DuckDB load, CSV export, AI report
  pipeline.py                run_pipeline(): runs all the steps
tests/                       unit and integration tests
notebooks/                   exploration notebooks
ARCHITECTURE.md              data model and design decisions
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named venv` or `ensurepip is not available` | `sudo apt install python3-venv`, then create the environment again |
| `Requires-Python >=3.12` during `pip install` | Your Python is too old. Install Python 3.12 or newer |
| `No module named src` | Run the command from the project root (the folder with `src/`), not from inside `src/` |
| `Could not set lock on file ... tech_news.duckdb` | Another program has the database open, often a notebook. Close its connection or restart its kernel, then run again |
