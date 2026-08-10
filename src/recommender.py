import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def build_recommender(df, max_features=50000):
    """Train a TF-IDF + cosine-distance nearest-neighbor recommender."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(df["combined_features"])

    nn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(21, len(df)),
    )
    nn.fit(matrix)

    bundle = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "nn": nn,
        "titles": df["title"].tolist(),
        "genres": df["genres_list"].tolist(),
        "metadata": df[
            [
                "title",
                "primary_genre",
                "release_year",
                "vote_average",
                "vote_count",
                "popularity",
                "budget",
                "revenue",
                "runtime",
                "original_language",
                "overview",
            ]
        ].copy(),
    }
    return bundle


def save_recommender(bundle, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)


def load_recommender(model_path):
    return joblib.load(model_path)


def _find_title_index(titles, movie_title):
    """Find exact title first, then a case-insensitive partial match."""
    normalized = movie_title.strip().lower()
    exact = [i for i, t in enumerate(titles) if t.lower() == normalized]
    if exact:
        return exact[0]

    partial = [i for i, t in enumerate(titles) if normalized in t.lower()]
    if partial:
        return partial[0]

    raise ValueError(f"Movie not found: {movie_title}")


def recommend_movies(bundle, movie_title, n=10):
    """Return the top N similar movies, excluding the selected movie itself."""
    titles = bundle["titles"]
    index = _find_title_index(titles, movie_title)

    query = bundle["matrix"][index]
    distances, indices = bundle["nn"].kneighbors(
        query, n_neighbors=min(n + 1, len(titles))
    )

    rows = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == index:
            continue

        similarity = max(0.0, 1.0 - float(distance))
        record = bundle["metadata"].iloc[idx].to_dict()
        record["similarity"] = similarity
        rows.append(record)

        if len(rows) >= n:
            break

    return rows


def get_title_matches(bundle, query, limit=10):
    """Return titles matching a search query for the Streamlit autocomplete."""
    q = query.strip().lower()
    if not q:
        return bundle["titles"][:limit]

    matches = [t for t in bundle["titles"] if q in t.lower()]
    return matches[:limit]
