# Image to run the pipeline and the tests (the outputs stay in a Docker volume, see compose.yaml)
FROM python:3.12-slim

# No .pyc files, and logs show up right away
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so this layer is reused when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code, tests and the two input files (the generated data/output is not copied)
COPY src ./src
COPY tests ./tests
COPY data ./data

# Default command: run the pipeline
CMD ["python", "-m", "src.pipeline"]
