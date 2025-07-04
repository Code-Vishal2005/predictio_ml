import streamlit as st
import pandas as pd
import difflib
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Page Config
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Movie Recommendation System")
st.markdown("Get movie recommendations based on your favorite movie!")

uploaded_file = st.file_uploader("Upload your movies.csv file")
st.write("Current working directory:", os.getcwd())

if uploaded_file is not None:
    movies_data = pd.read_csv(uploaded_file)
    st.write(movies_data)
else:
    st.warning("Please upload the 'movies.csv' file.")
# Load pre-trained model for movie recommendations
with open("E:\Machine Learning Projects\similarity_model.pkl", "rb") as file:
    model = joblib.load(file)
# Load movie dataset
@st.cache_data
def load_data():
    return pd.read_csv("E:\Datasets\movies.csv")

movies_data = load_data()

# Select features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorize
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity
similarity = cosine_similarity(feature_vectors)

# User input
movie_list = movies_data['title'].tolist()
movie_name = st.selectbox("🎞️ Choose a movie you like:", sorted(movie_list))

if st.button("🔍 Show Recommendations"):
    find_close_match = difflib.get_close_matches(movie_name, movie_list)
    if find_close_match:
        close_match = find_close_match[0]
        index_of_movie = movies_data[movies_data.title == close_match].index[0]
        similarity_scores = list(enumerate(similarity[index_of_movie]))
        sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:15]

        st.success(f"📌 Movies similar to **{close_match}**:")
        for i, (index, score) in enumerate(sorted_similar_movies):
            st.write(f"{i+1}. 🎬 {movies_data.iloc[index]['title']}")
    else:
        st.error("Sorry, no close match found. Try another title.")
