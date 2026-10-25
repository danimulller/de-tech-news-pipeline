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
| OS | Linux or WSL 2 (Ubuntu 24.04 recommended). On Windows, [install WSL](https://learn.microsoft.com/windows/wsl/install) first |
| Python | **3.11 or newer** (tested on 3.11, 3.12, 3.13 and 3.14) |
| Disk | About 300 MB for the dependencies |
| Memory | Under 1 GB. No GPU |
| Time | A few seconds per run |
| Internet | Only to install the dependencies. The pipeline runs offline |
| Docker | Optional. [Docker Desktop](https://www.docker.com/products/docker-desktop/) can replace the Python setup, see [Running with Docker](#running-with-docker-optional) |

## Installation and running (Linux or WSL)

This section is for a **Linux or WSL 2 terminal**. Run the steps in order; steps 1 and 2 are needed only once.
To use **Docker** instead, go to [Running with Docker](#running-with-docker-optional) below.

On Windows, there is no Linux terminal until WSL is installed. Follow the official
[Install WSL](https://learn.microsoft.com/windows/wsl/install) guide (it installs Ubuntu by default),
then open the Ubuntu terminal and start from step 1.

1. Check that Python is 3.11 or newer:

   ```bash
   python3 --version
   ```

   If it is older (Ubuntu 22.04 comes with Python 3.10), install Python 3.11 or newer before going on.

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

   To also see the test coverage (97% of the lines of `src/`), install the dev dependencies first:

   ```bash
   pip install -r requirements-dev.txt
   coverage run --branch --source=src -m pytest -q
   coverage report -m
   ```

In a new terminal, go to the project folder and run `source .venv/bin/activate` again before steps 6 and 7.

## Running with Docker (optional)

An alternative to the Linux/WSL steps above, on Windows, macOS or Linux. No Python setup needed. Install
[Docker Desktop](https://www.docker.com/products/docker-desktop/), **open it** and wait until it shows
"Engine running". Then, in a terminal:

```bash
git clone https://github.com/danimulller/de-tech-news-pipeline.git
cd de-tech-news-pipeline
docker compose up -d --build
```

This builds the image, runs the pipeline and keeps the container open. The CSV files and the DuckDB file
stay **inside Docker** (in a volume), not in the project folder.

| Command | What it does |
|---|---|
| `docker compose logs` | Shows the pipeline log |
| `docker compose exec pipeline bash` | Opens a terminal in the container (try `ls data/output`) |
| `docker compose exec pipeline python -m pytest -q` | Runs the tests |
| `docker compose cp pipeline:/app/data/output ./docker_output` | Copies the output files to your machine |
| `docker compose stop` | Stops the container |
| `docker compose start` | Starts it again (runs the pipeline again, safe) |
| `docker compose down -v --rmi local` | **Removes everything**: container, output files and image |

## Regenerating the outputs

Running the pipeline again is safe: the same input gives the same CSV files and no duplicate rows.
To see every file being recreated, delete everything in the output folder and run the pipeline again:

```bash
rm -f data/output/*
python -m src.pipeline
```

With Docker:

```bash
docker compose exec pipeline sh -c "rm -f data/output/*"
docker compose exec pipeline python -m src.pipeline
```

The `.duckdb` file is recreated too. It has the same rows, but its bytes can change, so it is not
versioned (it is in `.gitignore`).

| File in `data/output/` | What it is | Rows |
|---|---|---|
| `dim_company.csv` | One row per company, with metadata | 26 |
| `fact_article.csv` | One row per source article, raw and cleaned values side by side | 750 |
| `fact_arr_observation.csv` | One row per valid ARR reading (the ARR over time table) | 558 |
| `ai_articles_enriched.csv` | AI-related articles, 2022 to 2024, ARR above $50M. The `embedding` column is always empty (the bonus part is not implemented) | 124 |
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
Dockerfile, compose.yaml     optional Docker setup
ARCHITECTURE.md              data model and design decisions
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named venv` or `ensurepip is not available` | `sudo apt install python3-venv`, then create the environment again |
| `No matching distribution found for pandas==3.1.0` during `pip install` | Your Python is older than 3.11. Install Python 3.11 or newer |
| `No module named src` | Run the command from the project root (the folder with `src/`), not from inside `src/` |
| `failed to connect to the docker API` | Docker Desktop is closed. Open it, wait for "Engine running" and try again |
| `Could not set lock on file ... tech_news.duckdb` | Another program has the database open, often a notebook. Close its connection or restart its kernel, then run again |
