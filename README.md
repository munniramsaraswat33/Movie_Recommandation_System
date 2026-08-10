# 🎬 Intelligent Movie Recommendation System

A complete Data Science project that combines exploratory data analysis,
content-based movie recommendation, TF-IDF feature engineering, cosine
similarity, offline proxy evaluation, and an interactive Streamlit dashboard.

## Project Objective

Build a movie analytics and recommendation application that:

1. Cleans and explores movie metadata.
2. Performs exploratory data analysis.
3. Converts movie text/metadata into TF-IDF vectors.
4. Finds similar movies using cosine distance.
5. Generates Top-N movie recommendations.
6. Provides a transparent offline genre-overlap evaluation.
7. Presents the results through Streamlit.

## Dataset

The project uses `data/movies_metadata.csv`.

The supplied dataset contains movie metadata such as:

- title
- overview
- genres
- tagline
- release date
- budget
- revenue
- popularity
- runtime
- vote average
- vote count
- original language

The recommendation model only uses columns that are actually present in this dataset.

## Machine Learning Approach

### 1. Feature Engineering

The following available text fields are combined:

- title
- overview
- tagline
- original title
- genres
- original language

### 2. TF-IDF

TF-IDF converts the combined movie text into numerical vectors.

### 3. Cosine Similarity

Cosine distance is used by `NearestNeighbors`; similarity is calculated as:

`similarity = 1 - cosine_distance`

### 4. Recommendation

For a selected movie, the system returns the Top-N most similar movies.

## Evaluation

Because this dataset does not provide user-level preference histories in the
current project files, standard collaborative-filtering RMSE cannot be honestly
computed from these files.

Instead, the project includes a transparent proxy:

**Genre Precision@K**

A recommendation is considered relevant when it shares at least one genre with
the query movie.

This should be described in the presentation as an **offline proxy evaluation**,
not as user-level recommendation accuracy.

## Folder Structure

```text
Movie_Recommendation_Data_Science_Project/
│
├── app.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── movies_metadata.csv
│
├── models/
│   └── movie_recommender.joblib   # generated after training
│
└── src/
    ├── __init__.py
    ├── data_processing.py
    ├── recommender.py
    └── evaluation.py
```

## Setup

Use Python 3.12 for the smoothest compatibility with common Data Science
libraries.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Train the model

Run this once:

```powershell
python train_model.py
```

This creates:

```text
models/movie_recommender.joblib
```

### Evaluate the model

```powershell
python evaluate_model.py
```

### Start the Streamlit application

```powershell
streamlit run app.py
```

The application contains:

- EDA Dashboard
- Recommendation System
- Offline Evaluation
- Data Explorer

## Important Project Claims

This is a **content-based recommendation system**.

Do not describe it as collaborative filtering unless a separate user-rating
dataset and collaborative-filtering model are added.

Do not claim RMSE/MAPE results unless a rating-prediction model and appropriate
ground-truth test data are implemented.

## Future Scope

- Add a user-rating dataset for collaborative filtering.
- Build a hybrid recommender combining content and user behavior.
- Add movie posters using a legitimate movie API.
- Add user accounts and personalized watch history.
- Deploy the Streamlit application.
- Add a larger-scale vector database for production search.

## Author

Munniram Saraswat

```text
models/movie_recommender.joblib
```

### Evaluate the model

```powershell
python evaluate_model.py
```

### Start the Streamlit application

```powershell
streamlit run app.py
```

The application contains:

- EDA Dashboard
- Recommendation System
- Offline Evaluation
- Data Explorer

## Important Project Claims

This is a **content-based recommendation system**.

Do not describe it as collaborative filtering unless a separate user-rating
dataset and collaborative-filtering model are added.

Do not claim RMSE/MAPE results unless a rating-prediction model and appropriate
ground-truth test data are implemented.

## Future Scope

- Add a user-rating dataset for collaborative filtering.
- Build a hybrid recommender combining content and user behavior.
- Add movie posters using a legitimate movie API.
- Add user accounts and personalized watch history.
- Deploy the Streamlit application.
- Add a larger-scale vector database for production search.

## Author

Munniram Saraswat
>>>>>>> 438400d (Initiatl commit)
