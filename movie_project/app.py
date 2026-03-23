import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------------------------
# Ocean blue theme + modern cards
st.markdown("""
<style>
body {
    background-color: #001f3f !important;  /* Dark navy */
    color: #B0E0E6 !important;
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background-color: #001f3f !important;
}
.row-title {
    font-size: 24px;
    font-weight: bold;
    color: #00CED1;
    margin-top: 20px;
    margin-bottom: 10px;
}
.cards-container {
    display: flex;
    overflow-x: auto;
    gap: 20px;
    padding-bottom: 20px;
}
.card {
    background: linear-gradient(145deg, #004080, #0066cc);
    padding: 25px;
    border-radius: 15px;
    min-width: 180px;
    flex: 0 0 auto;
    text-align: center;
    color: #E0FFFF;
    box-shadow: 0 8px 15px rgba(0,0,0,0.5);
    transition: transform 0.3s, background 0.3s;
    font-size: 16px;
}
.card:hover {
    transform: scale(1.08);
    background: linear-gradient(145deg, #0066cc, #3399ff);
    color: #ffffff;
}
.icon {
    font-size: 40px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#00CED1;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)


# ---------------------------
# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[['title', 'genres']]
movies['genres'] = movies['genres'].fillna('')

# ---------------------------
# Vectorization
cv = CountVectorizer()
matrix = cv.fit_transform(movies['genres'])
similarity = cosine_similarity(matrix)

# ---------------------------
# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower().strip()
    titles = movies['title'].str.lower()
    if movie_name not in titles.values:
        return ["Movie not found"]
    index = titles[titles == movie_name].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:11]
    recommended_titles = [movies.iloc[i[0]].title for i in movies_list]
    return recommended_titles

# ---------------------------
# Input
st.markdown("<h3 style='color:#B0E0E6;'>Type a movie name:</h3>", unsafe_allow_html=True)
movie_name = st.text_input("")
if st.button("Recommend"):
    titles = recommend(movie_name)
    if titles[0] == "Movie not found":
        st.markdown("<p style='color:red;'>❌ Movie not found. Try another name.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='row-title'>Recommendations</div>", unsafe_allow_html=True)
        st.markdown("<div class='cards-container'>", unsafe_allow_html=True)
        for t in titles:
            # Use emoji as icon placeholder
            st.markdown(f"<div class='card'><div class='icon'>🎬</div>{t}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Optional multiple rows: example categories
categories = ["Trending", "Action", "Romance"]
for cat in categories:
    st.markdown(f"<div class='row-title'>{cat}</div>", unsafe_allow_html=True)
    st.markdown("<div class='cards-container'>", unsafe_allow_html=True)
    for t in movies['title'].sample(10, random_state=42):  # pick 10 random movies for demo
        st.markdown(f"<div class='card'><div class='icon'>🎬</div>{t}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)