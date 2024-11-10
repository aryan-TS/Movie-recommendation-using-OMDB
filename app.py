from flask import Flask, render_template, request
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
OMDB_API_KEY = os.getenv('OMDB_API_KEY')

def fetch_movie_details(title):
    """Fetch movie details from OMDB API"""
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}&type=movie"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get('Response') == 'True':
            return {
                'Title': data.get('Title', ''),
                'Year': data.get('Year', ''),
                'Genre': data.get('Genre', ''),
                'Plot': data.get('Plot', ''),
                'Poster': data.get('Poster', ''),
                'imdbRating': data.get('imdbRating', 'N/A')
            }
    return None

def search_movies(query):
    """Search movies from OMDB API"""
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={query}&type=movie"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get('Response') == 'True':
            return data.get('Search', [])
    return []

def get_recommendations(query):
    """Get movie recommendations based on search query"""
    initial_results = search_movies(query)
    if not initial_results:
        return []
    
    movies_data = []
    
    for movie in initial_results[:10]:  # Limit to first 10 results
        details = fetch_movie_details(movie['Title'])
        if details:
            movies_data.append(details)
    
    if not movies_data:
        return []
    
    # Create a DataFrame for content-based filtering
    df = pd.DataFrame(movies_data)
    
    # Create feature matrix from genres
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(df['Genre'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Get recommendations based on genre similarity
    query_genres = set(df[df['Title'].str.lower() == query.lower()]['Genre'].str.split(', ').explode().tolist())
    sim_scores = []
    for i, genres in enumerate(df['Genre']):
        genres_set = set(genres.split(', '))
        sim_score = len(query_genres.intersection(genres_set)) / len(query_genres.union(genres_set))
        sim_scores.append(sim_score)
    
    # Sort movies by genre similarity and get the top 3 unique recommendations
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    unique_recommendations = []
    for i, score in sim_scores:
        if len(unique_recommendations) < 3 and df.iloc[i]['Title'] not in [m['Title'] for m in unique_recommendations]:
            unique_recommendations.append(movies_data[i])
        if len(unique_recommendations) >= 3:
            break
    
    return unique_recommendations

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    error = None
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            recommendations = get_recommendations(query)
            if not recommendations:
                error = "No recommendations found. Please try another movie title!"
    
    return render_template('index.html', recommendations=recommendations, error=error)

if __name__ == '__main__':
    app.run(debug=True)