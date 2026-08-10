import random


def genre_precision_at_k(bundle, sample_size=100, k_values=(5, 10), seed=42):
    """
    Proxy evaluation for a content-based recommender.

    A recommendation is counted as relevant when it shares at least one genre
    with the query movie. This is NOT the same as user-level ground-truth
    evaluation; it is a transparent offline consistency check.
    """
    rng = random.Random(seed)
    n_movies = len(bundle["titles"])
    sample_indices = list(range(n_movies))
    rng.shuffle(sample_indices)
    sample_indices = sample_indices[: min(sample_size, n_movies)]

    scores = {k: [] for k in k_values}

    for idx in sample_indices:
        query_genres = set(bundle["genres"][idx])
        if not query_genres:
            continue

        query = bundle["matrix"][idx]
        max_k = max(k_values)

        distances, indices = bundle["nn"].kneighbors(
            query,
            n_neighbors=min(max_k + 1, n_movies),
        )

        rec_indices = [i for i in indices[0] if i != idx][:max_k]

        for k in k_values:
            selected = rec_indices[:k]
            if not selected:
                continue

            relevant = sum(
                bool(query_genres.intersection(set(bundle["genres"][i])))
                for i in selected
            )
            scores[k].append(relevant / len(selected))

    return {
        f"Genre Precision@{k}": round(sum(values) / len(values), 4)
        if values else 0.0
        for k, values in scores.items()
    }
