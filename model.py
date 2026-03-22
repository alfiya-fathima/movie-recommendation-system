import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Select important columns
movies = movies[['original_title','genres','keywords','overview']]
movies = movies.dropna()

# Convert JSON-like data to text
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return " ".join(L)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Combine all features
movies['combined'] = movies['genres'] + " " + movies['keywords'] + " " + movies['overview']

# Convert text to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
matrix = cv.fit_transform(movies['combined'])

# Calculate similarity
similarity = cosine_similarity(matrix)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    
    if movie not in movies['original_title'].str.lower().values:
        return ["Movie not found"]
    
    index = movies[movies['original_title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommended = []
    for i in movies_list:
        recommended.append(movies.iloc[i[0]].original_title)
    
    return recommended