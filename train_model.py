from pathlib import Path

from src.data_processing import load_and_clean_data
from src.recommender import build_recommender, save_recommender


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "movies_metadata.csv"
MODEL_PATH = BASE_DIR / "models" / "movie_recommender.joblib"


def main():
    print("Loading and cleaning movie data...")
    df = load_and_clean_data(DATA_PATH, min_votes=100)

    print(f"Movies used for modelling: {len(df):,}")
    print("Building TF-IDF matrix and nearest-neighbor model...")

    bundle = build_recommender(df, max_features=50_000)
    save_recommender(bundle, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
