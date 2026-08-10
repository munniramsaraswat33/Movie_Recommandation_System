from pathlib import Path
import ast

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import load_and_clean_data
from src.recommender import load_recommender, recommend_movies
from src.evaluation import genre_precision_at_k


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "movies_metadata.csv"
MODEL_PATH = BASE_DIR / "models" / "movie_recommender.joblib"


st.set_page_config(
    page_title="Cinematically | Movie Data Science",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="metric-container"] {
        border: 1px solid rgba(128,128,128,.25);
        padding: 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_and_clean_data(DATA_PATH, min_votes=100)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_recommender(MODEL_PATH)


df = get_data()

st.title("🎬 MOVIE RECOMMANDATION SYSTEM")
st.caption(
    "Intelligent Movie Recommendation System using TF-IDF, cosine similarity, "
    "exploratory data analysis, and Streamlit."
)

if not MODEL_PATH.exists():
    st.error(
        "Recommendation model is not available yet. "
        "Run `python train_model.py` from the project root, then restart Streamlit."
    )
    st.stop()

bundle = get_model()

# ---------------- Sidebar ----------------
st.sidebar.header("🎛️ Dashboard Controls")

min_year = int(df["release_year"].dropna().min())
max_year = int(df["release_year"].dropna().max())

year_range = st.sidebar.slider(
    "Release year",
    min_year,
    max_year,
    (max(1990, min_year), max_year),
)

genre_options = sorted(df["primary_genre"].dropna().unique().tolist())
selected_genres = st.sidebar.multiselect(
    "Primary genre",
    genre_options,
    default=genre_options,
)

min_rating = st.sidebar.slider(
    "Minimum rating",
    0.0,
    10.0,
    6.0,
    0.5,
)

mask = (
    df["release_year"].between(year_range[0], year_range[1], inclusive="both")
    & df["vote_average"].ge(min_rating)
)

if selected_genres:
    mask &= df["primary_genre"].isin(selected_genres)

filtered = df[mask].copy()

# ---------------- KPI cards ----------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Movies analyzed", f"{len(filtered):,}")

with k2:
    avg_rating = filtered["vote_average"].mean() if not filtered.empty else 0
    st.metric("Average rating", f"{avg_rating:.2f}/10")

with k3:
    valid_budget = filtered.loc[filtered["budget"] > 0, "budget"]
    avg_budget = valid_budget.mean() / 1e6 if not valid_budget.empty else 0
    st.metric("Average budget", f"${avg_budget:.1f}M")

with k4:
    valid_revenue = filtered.loc[filtered["revenue"] > 0, "revenue"]
    avg_revenue = valid_revenue.mean() / 1e6 if not valid_revenue.empty else 0
    st.metric("Average revenue", f"${avg_revenue:.1f}M")

if filtered.empty:
    st.warning("No movies match the selected filters.")
    st.stop()

tab_eda, tab_recommend, tab_evaluation, tab_explorer = st.tabs(
    ["📊 EDA Dashboard", "🤖 Recommendation", "🧪 Evaluation", "🔍 Data Explorer"]
)

# ---------------- EDA ----------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    c1, c2 = st.columns(2)

    with c1:
        yearly = (
            filtered.groupby("release_year")
            .size()
            .reset_index(name="movie_count")
        )
        fig = px.area(
            yearly,
            x="release_year",
            y="movie_count",
            title="Movie Releases Over Time",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        financial = filtered[
            (filtered["budget"] > 0) & (filtered["revenue"] > 0)
        ]
        fig = px.scatter(
            financial,
            x="budget",
            y="revenue",
            color="vote_average",
            hover_name="title",
            title="Budget vs Revenue",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        genre_counts = (
            filtered["primary_genre"]
            .value_counts()
            .reset_index()
        )
        genre_counts.columns = ["Genre", "Count"]
        fig = px.pie(
            genre_counts,
            names="Genre",
            values="Count",
            hole=0.4,
            title="Genre Composition",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        top_rated = (
            filtered.sort_values(
                ["vote_average", "vote_count"],
                ascending=[False, False],
            )
            .head(10)
            .sort_values("vote_average")
        )
        fig = px.bar(
            top_rated,
            x="vote_average",
            y="title",
            orientation="h",
            color="vote_count",
            title="Top Rated Movies",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        popular = (
            filtered.sort_values("popularity", ascending=False)
            .head(10)
            .sort_values("popularity")
        )
        fig = px.bar(
            popular,
            x="popularity",
            y="title",
            orientation="h",
            title="Most Popular Movies",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        runtime_df = filtered[filtered["runtime"] > 0]
        fig = px.scatter(
            runtime_df,
            x="runtime",
            y="vote_average",
            color="primary_genre",
            hover_name="title",
            title="Runtime vs Rating",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    c7, c8 = st.columns(2)

    with c7:
        runtime_df = filtered[filtered["runtime"] > 0]
        fig = px.histogram(
            runtime_df,
            x="runtime",
            nbins=50,
            title="Runtime Distribution",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        lang = (
            filtered["original_language"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        lang.columns = ["Language", "Count"]
        fig = px.bar(
            lang,
            x="Language",
            y="Count",
            title="Top Original Languages",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    c9, c10 = st.columns(2)

    with c9:
        top_genres = filtered["primary_genre"].value_counts().head(5).index
        box_df = filtered[
            filtered["primary_genre"].isin(top_genres)
            & (filtered["revenue"] > 0)
        ]
        fig = px.box(
            box_df,
            x="primary_genre",
            y="revenue",
            color="primary_genre",
            title="Revenue Distribution by Genre",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c10:
        fin_yearly = (
            filtered[
                (filtered["budget"] > 0) | (filtered["revenue"] > 0)
            ]
            .groupby("release_year")[["budget", "revenue"]]
            .mean()
            .reset_index()
        )
        fig = px.line(
            fin_yearly,
            x="release_year",
            y=["budget", "revenue"],
            title="Average Budget and Revenue by Year",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------- Recommendation ----------------
with tab_recommend:
    st.subheader("🤖 Content-Based Movie Recommendation")

    st.write(
        "Select a movie. The system compares its text-based movie features "
        "using TF-IDF and cosine similarity."
    )

    movie_titles = sorted(bundle["titles"])

    selected_movie = st.selectbox(
        "Choose a movie",
        movie_titles,
        index=movie_titles.index("Toy Story") if "Toy Story" in movie_titles else 0,
    )

    n_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=15,
        value=10,
    )

    if st.button("🎯 Recommend Movies", type="primary"):
        recommendations = recommend_movies(
            bundle,
            selected_movie,
            n=n_recommendations,
        )

        if recommendations:
            result_df = pd.DataFrame(recommendations)
            result_df["similarity"] = (
                result_df["similarity"] * 100
            ).round(2)

            st.success(
                f"Top recommendations similar to **{selected_movie}**"
            )

            for i, row in result_df.iterrows():
                st.markdown(
                    f"### {i + 1}. {row['title']}"
                )
                st.write(
                    f"**Similarity:** {row['similarity']:.2f}%  |  "
                    f"**Genre:** {row['primary_genre']}  |  "
                    f"**Rating:** {row['vote_average']:.1f}/10"
                )
                if row.get("overview"):
                    st.caption(str(row["overview"])[:350])
                st.divider()
        else:
            st.warning("No recommendations were generated.")

# ---------------- Evaluation ----------------
with tab_evaluation:
    st.subheader("🧪 Offline Evaluation")

    st.info(
        "This project uses a transparent genre-overlap proxy because the supplied "
        "movie metadata does not contain user-level preference/rating histories "
        "for a standard collaborative-filtering ground truth."
    )

    if st.button("Run Evaluation", type="secondary"):
        with st.spinner("Evaluating sampled movies..."):
            results = genre_precision_at_k(
                bundle,
                sample_size=100,
                k_values=(5, 10),
            )

        e1, e2 = st.columns(2)
        with e1:
            st.metric("Genre Precision@5", f"{results['Genre Precision@5']:.2%}")
        with e2:
            st.metric("Genre Precision@10", f"{results['Genre Precision@10']:.2%}")

        st.caption(
            "A recommendation is counted as relevant when it shares at least "
            "one genre with the query movie. This is a proxy consistency metric, "
            "not a user-level accuracy score."
        )

# ---------------- Explorer ----------------
with tab_explorer:
    st.subheader("🔍 Detailed Data Explorer")

    columns = [
        "title",
        "primary_genre",
        "release_year",
        "vote_average",
        "vote_count",
        "budget",
        "revenue",
        "popularity",
        "runtime",
        "original_language",
    ]

    display_df = filtered[columns].rename(
        columns={
            "title": "Title",
            "primary_genre": "Genre",
            "release_year": "Year",
            "vote_average": "Rating",
            "vote_count": "Votes",
            "budget": "Budget ($)",
            "revenue": "Revenue ($)",
            "popularity": "Popularity",
            "runtime": "Runtime (min)",
            "original_language": "Language",
        }
    )

    st.dataframe(
        display_df.head(500),
        use_container_width=True,
        height=450,
    )
