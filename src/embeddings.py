"""Text embeddings of the articles and the most similar articles of each one."""
from functools import lru_cache

import numpy as np
import pandas as pd

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_model():
    """Load the sentence-transformers model once (the first time it downloads the model, so it needs internet)."""

    # Imported here because it is slow, and only this function needs it
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def build_article_texts(df_fact_article: pd.DataFrame) -> list[str]:
    """Text to embed for each article: title + summary."""

    return [f"{title}. {summary}" for title, summary in zip(df_fact_article["title"], df_fact_article["summary"])]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed the texts, one row per text.

    The vectors are normalized (length 1), so the cosine similarity of two texts
    is just the dot product of their vectors.
    """

    model = load_model()

    return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def find_top_similar(embeddings: np.ndarray, article_ids: list[str], top_k: int = 3) -> list[list[str]]:
    """For each article, the ids of the top_k most similar articles (never the article itself).

    The most similar comes first. If two articles have the same score, the one that
    comes first in article_ids wins, so the result is always the same.
    """

    # Cosine similarity between every pair of articles (vectors are normalized)
    similarity = embeddings @ embeddings.T

    # Rounded, so a tiny float error doesn't decide the order of equal scores
    # (articles with the same text have the same score, and must be sorted by position)
    similarity = np.round(similarity, 6)

    # An article is always the most similar to itself, so it is taken out of its own ranking
    np.fill_diagonal(similarity, -np.inf)

    # Can't return more neighbors than there are other articles
    top_k = min(top_k, len(article_ids) - 1)

    # Highest scores first; the stable sort keeps the original order when scores are tied
    ranking = np.argsort(-similarity, axis=1, kind="stable")[:, :top_k]

    return [[article_ids[j] for j in row] for row in ranking]
