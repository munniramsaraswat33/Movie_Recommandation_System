from pathlib import Path

from src.recommender import load_recommender
from src.evaluation import genre_precision_at_k


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "movie_recommender.joblib"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python train_model.py"
        )

    bundle = load_recommender(MODEL_PATH)
    results = genre_precision_at_k(
        bundle,
        sample_size=100,
        k_values=(5, 10),
    )

    print("\nOffline proxy evaluation")
    print("------------------------")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    print(
        "\nNote: these are genre-overlap proxy metrics, not user-level "
        "ground-truth recommendation metrics."
    )


if __name__ == "__main__":
    main()
