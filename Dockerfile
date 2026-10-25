# Image to run the pipeline and the tests (the data folder is mounted by compose.yaml)
FROM python:3.12-slim

# No .pyc files, and logs show up right away
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so this layer is reused when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and tests
COPY src ./src
COPY tests ./tests

# Default command: run the pipeline
CMD ["python", "-m", "src.pipeline"]
