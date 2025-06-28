import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
movies.head()
# Merge datasets
movie_data = pd.merge(ratings, movies, on='movieId')
movie_data.dropna(inplace=True)

# Create user-item matrix
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN with 0 for collaborative filtering
user_movie_matrix.fillna(0, inplace=True)

# Compute similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Process genres
movies['genres'] = movies['genres'].str.replace('|', ' ')
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])
genre_similarity = cosine_similarity(genre_matrix)

# Genre similarity dataframe
genre_similarity_df = pd.DataFrame(genre_similarity, index=movies['title'], columns=movies['title'])

def recommend(movie_title):
    # Average of collaborative and genre similarity
    if movie_title not in movie_similarity_df or movie_title not in genre_similarity_df:
        return ["Movie not found"]

    collab_scores = movie_similarity_df[movie_title]
    genre_scores = genre_similarity_df[movie_title]
    final_scores = (collab_scores + genre_scores) / 2

    recommendations = final_scores.sort_values(ascending=False)[1:6]
    return list(recommendations.index)


# To run: `streamlit run movie_recommendation_without_sentiment_score.py`

st.title("🎬 Movie Recommendation System without sentiment filtering")
movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie you like:", ["🔽 Select a movie"] + list(movie_list))

if selected_movie != "🔽 Select a movie" and st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.markdown(f"🎬 You liked the movie **{selected_movie}**.")
    st.markdown("📽️ **Top 5 recommendations for you:**")
    for i, rec in enumerate(recommendations):
        st.write(f"{i+1}. {rec}")
