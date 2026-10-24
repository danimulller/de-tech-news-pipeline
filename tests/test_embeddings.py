import numpy as np
import pandas as pd

from src.embeddings import build_article_texts, find_top_similar


# The text to embed is the title followed by the summary
def test_build_article_texts():
    df_fact_article = pd.DataFrame({"title": ["Big news", "Other"], "summary": ["It happened.", "Nothing."]})

    assert build_article_texts(df_fact_article) == ["Big news. It happened.", "Other. Nothing."]


# Most similar first, and an article is never in its own list
def test_find_top_similar_orders_by_similarity_and_excludes_itself():
    # Unit vectors at angles 0, 10, 20 and 90 degrees: the closer the angle, the more similar
    angles = np.radians([0, 10, 20, 90])
    embeddings = np.column_stack([np.cos(angles), np.sin(angles)])

    top_similar = find_top_similar(embeddings, ["A", "B", "C", "D"], top_k=3)

    assert top_similar == [
        ["B", "C", "D"],  # A: B is 10 degrees away, C 20, D 90
        ["A", "C", "D"],  # B: A is 10 degrees away, C 10 too (A wins the tie), D 80
        ["B", "A", "D"],  # C: B is 10 degrees away, A 20, D 70
        ["C", "B", "A"],  # D: C is 70 degrees away, B 80, A 90
    ]


# Same score: the article that comes first in the list wins, so the result never changes
def test_find_top_similar_breaks_ties_by_position():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    top_similar = find_top_similar(embeddings, ["A", "B", "C", "D"], top_k=2)

    assert top_similar[0] == ["B", "C"]
    assert top_similar[1] == ["A", "C"]
    assert top_similar[2] == ["A", "B"]


# Scores that only differ by a tiny float error are a tie too (articles with the same text)
def test_find_top_similar_ignores_tiny_float_errors():
    embeddings = np.array([[0.0, 1.0], [1.0, -1e-8], [1.0, 1e-8], [1.0, 0.0]])

    # Without rounding the order would be C, D, B (by the float error); with it, they tie and B, C, D is by position
    assert find_top_similar(embeddings, ["A", "B", "C", "D"], top_k=3)[0] == ["B", "C", "D"]


# With fewer articles than top_k + 1, it returns all the others
def test_find_top_similar_with_few_articles():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert find_top_similar(embeddings, ["A", "B"], top_k=3) == [["B"], ["A"]]
