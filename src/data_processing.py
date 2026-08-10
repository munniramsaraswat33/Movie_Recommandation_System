import ast
import pandas as pd


NUMERIC_COLUMNS = [
    "budget",
    "revenue",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
]


def parse_genres(value):
    """Return a clean list of genre names from the dataset's string representation."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [
                item.get("name", "").strip()
                for item in parsed
                if isinstance(item, dict) and item.get("name")
            ]
    except (ValueError, SyntaxError, TypeError):
        pass
    return []


def load_and_clean_data(filepath, min_votes=100):
    """Load the movie metadata CSV and create analysis/recommendation features."""
    df = pd.read_csv(filepath, low_memory=False)

    required = ["title", "overview", "genres", "release_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["overview"] = df["overview"].fillna("").astype(str).str.strip()
    df["tagline"] = df.get("tagline", "").fillna("").astype(str).str.strip()
    df["original_title"] = df.get("original_title", "").fillna("").astype(str).str.strip()
    df["original_language"] = (
        df.get("original_language", "")
        .fillna("")
        .astype(str)
        .str.strip()
    )

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["genres_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genres_list"].apply(
        lambda x: x[0] if x else "Unknown"
    )

    # A transparent text feature built only from columns present in the supplied dataset.
    df["combined_features"] = (
        df["title"] + " "
        + df["overview"] + " "
        + df["tagline"] + " "
        + df["original_title"] + " "
        + df["genres_list"].apply(lambda x: " ".join(x)) + " "
        + df["original_language"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Remove records that cannot be useful to the recommender.
    df = df[
        (df["title"] != "")
        & (df["combined_features"].str.len() > 10)
        & (df["vote_count"] >= min_votes)
    ].copy()

    # Keep the first occurrence of a title to make the UI less confusing.
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)

    return df
