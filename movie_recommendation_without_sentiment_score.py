import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# Page title
st.title("🎬 Movie Recommendation System without Sentiment Filtering")

# Load datasets
try:
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    st.success("✅ CSVs loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load CSVs: {e}")
    st.stop()

# Merge datasets
try:
    movie_data = pd.merge(ratings, movies, on='movieId')
    st.success("✅ Merge success")
except Exception as e:
    st.error(f"❌ Merge failed: {e}")
    st.stop()

# Drop any rows with missing values
movie_data.dropna(inplace=True)

# Create user-item matrix
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)

st.success("✅ Pivot table created")
st.write("Matrix shape:", user_movie_matrix.shape)
st.write("Any NaNs left in matrix?", user_movie_matrix.isnull().values.any())

# Compute collaborative filtering similarity
try:
    movie_similarity = cosine_similarity(user_movie_matrix.T)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
except Exception as e:
    st.error(f"❌ Similarity computation error (collaborative): {e}")
    st.stop()

# Process genres
movies['genres'] = movies['genres'].fillna('')  # Fill any missing genre entries
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# Compute content-based similarity (genres)
try:
    cv = CountVectorizer()
    genre_matrix = cv.fit_transform(movies['genres'])
    genre_similarity = cosine_similarity(genre_matrix)
    genre_similarity_df = pd.DataFrame(genre_similarity, index=movies['title'], columns=movies['title'])
except Exception as e:
    st.error(f"❌ Similarity computation error (genre): {e}")
    st.stop()

# Recommender logic
def recommend(movie_title):
    if movie_title not in movie_similarity_df or movie_title not in genre_similarity_df:
        return ["Movie not found"]

    collab_scores = movie_similarity_df[movie_title]
    genre_scores = genre_similarity_df[movie_title]

    # Ensure no NaNs
    final_scores = (collab_scores.fillna(0) + genre_scores.fillna(0)) / 2

    recommendations = final_scores.sort_values(ascending=False)[1:6]
    return list(recommendations.index)

# Streamlit UI
movie_list = movies['title'].dropna().unique()
selected_movie = st.selectbox("Select a movie you like:", ["🔽 Select a movie"] + list(movie_list))

if selected_movie != "🔽 Select a movie" and st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.markdown(f"🎬 You liked the movie **{selected_movie}**.")
    st.markdown("📽️ **Top 5 recommendations for you:**")
    for i, rec in enumerate(recommendations):
        st.write(f"{i+1}. {rec}")
