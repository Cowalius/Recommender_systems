import streamlit as st
import pandas as pd
import os
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
import requests
import csv
from datetime import datetime

#TMDB API key
API_KEY = '75c2e91b432df43287a4b58ef4337193'

# Function to fetch the movie poster URL and homepage URL from TMDB by TMDB ID
def get_movie_details_by_id(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        homepage = data.get('homepage', None)
        if not homepage:
            homepage = f"https://www.themoviedb.org/movie/{tmdb_id}"
        full_path = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        return full_path, homepage
    return None, f"https://www.themoviedb.org/movie/{tmdb_id}"

st.title('Movie Recommender System')

# Load data and cache it
@st.cache_data
def load_data():
    df = pd.read_csv('ratings.csv')  
    df['movieId'] = df['movieId'].astype(int)
    return df

df = load_data()

# Create survey results file if it doesn't exist
survey_file = "survey_results.csv"
if not os.path.exists(survey_file):
    with open(survey_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'selected_algorithm', 'survey_completed'])

# Survey in the sidebar: asking which algorithm gave the best recommendations
with st.sidebar:
    st.subheader("Survey: Which Algorithm Gave You the Best Recommendations?")
    best_algorithm = st.radio("Choose the best algorithm", options=['SVD', 'KNN', 'NMF'])
    if st.button('Submit'):
        # Record the result with a timestamp
        with open(survey_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), best_algorithm, True])
        st.success("Thank you for your feedback!")

# Sidebar explanations and info
with st.sidebar.expander("ℹ️ How does Collaborative Filtering work?"):
    st.write("""
    **Collaborative Filtering** is a widely-used technique in recommendation systems that leverages the preferences and interactions of users to make predictions. 
    It identifies patterns in user behavior by analyzing similarities between users (user-based filtering) or between items (item-based filtering). 
    The system then recommends items that users with similar preferences have liked or interacted with, even if the items themselves are not directly related.
    """)

with st.sidebar.expander("ℹ️ How does Content-Based Filtering work?"):
    st.write("""
    **Content-Based Filtering** is a recommendation approach that focuses on the attributes or features of items. 
    It suggests items based on their similarities to other items that the user has previously interacted with. 
    The system creates a profile of user preferences by analyzing item characteristics (such as genre, keywords, or other metadata) and recommends new items with similar traits.
    """)

with st.sidebar.expander("ℹ️ How does SVD work?"):
    st.write("""
    **SVD (Singular Value Decomposition)**: This algorithm reduces the dimensionality of the data and discovers latent factors. 
    It works by decomposing the user-item matrix into user and item feature matrices, learning the relationships between users and items.
    """)

with st.sidebar.expander("ℹ️ How does KNN work?"):
    st.write("""
    **KNN (K-Nearest Neighbors)**: This algorithm looks for similarities between movies based on user ratings. 
    It finds the k most similar movies to a given movie and predicts ratings based on the user's preferences for those similar movies.
    """)

with st.sidebar.expander("ℹ️ How does NMF work?"):
    st.write("""
    **NMF (Non-negative Matrix Factorization)**: This algorithm decomposes the user-item matrix into two lower-dimensional matrices, but unlike SVD, it only uses non-negative values.
    """)

# Explain likeliness
with st.sidebar.expander("ℹ️ How Percentage Likeliness Works"):
    st.write("""
    The percentage likeliness represents the predicted probability that you will enjoy a movie based on the rating the algorithm predicts for you.
    In other words 100% means the algorithm predicts you will give the movie a rating of 5, and 0% means the algorithm predicts you will give the movie a rating of 1.
    """)

# Preview data
with st.sidebar.expander("ℹ️ How the data that powers the app looks like"):
    st.write(df.head())

# Helpfull resources
with st.sidebar.expander("📚 Resources that will help you develop your understanding of Recommendation Systems"):
    st.write("[good in-depth article about Recommendation Systems](https://medium.com/@eli.hatcher/how-to-build-a-recommendation-system-e72fe9efb086).")
    st.write("[Google's Recommendation Systems tutorial](https://developers.google.com/machine-learning/recommendation/overview).")

# Video link
with st.sidebar.expander("🎬 How Recommendation Systems Works"):
    st.write("[A very good video explaining the logic behind Recommendation Systems](https://www.youtube.com/watch?v=n3RKsY2H-NE).")

# Movie selection and rating input
st.write('Rate some movies to get recommendations!')

movie_choice = st.selectbox('Pick the movie you want to rate:', df['title'].unique())
selected_movie_id = df[df['title'] == movie_choice]['movieId'].values[0]
selected_tmdb_id = df[df['title'] == movie_choice]['tmdbId'].values[0]

rating = st.selectbox('Your rating:', [1, 2, 3, 4, 5])

if st.button('Add Rating'):
    if "user_ratings" not in st.session_state:
        st.session_state["user_ratings"] = []
    
    st.session_state["user_ratings"].append({
        "user_id": 200000,
        "movieId": selected_movie_id,
        "rating": rating,
        "title": movie_choice,
        "tmdbId": selected_tmdb_id
    })
    st.success(f"Rating added: {rating} for the movie: {movie_choice}!")

# Display user ratings and recommendations
if st.session_state.get("user_ratings"):
    st.subheader('Your ratings:')
    user_ratings_df = pd.DataFrame(st.session_state["user_ratings"])
    
    # Display rated movies with posters
    for index, row in user_ratings_df.iterrows():
        poster_url, homepage = get_movie_details_by_id(row['tmdbId'])
        st.image(poster_url if poster_url else "https://via.placeholder.com/100x150?text=No+Image", width=100)
        st.markdown(f"[**{row['title']}** - Rating: {row['rating']}]({homepage})")

    # Combine user ratings with training data
    temp_ratings = pd.concat([df[['user_id', 'movieId', 'rating']], user_ratings_df])

    # Convert to Surprise dataset format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(temp_ratings[['user_id', 'movieId', 'rating']], reader)

    # Train SVD, KNN, and NMF models
    trainset = data.build_full_trainset()
    
    algo_svd = SVD()
    algo_svd.fit(trainset)

    algo_knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
    algo_knn.fit(trainset)

    algo_nmf = NMF()
    algo_nmf.fit(trainset)

    # Generate recommendations
    st.subheader('Recommendations for You:')
    movie_ids = df['movieId'].unique()
    user_id = 200000

    # Predict ratings with each algorithm
    predicted_ratings_svd = [(movie_id, algo_svd.predict(user_id, movie_id).est) for movie_id in movie_ids]
    predicted_ratings_knn = [(movie_id, algo_knn.predict(user_id, movie_id).est) for movie_id in movie_ids]
    predicted_ratings_nmf = [(movie_id, algo_nmf.predict(user_id, movie_id).est) for movie_id in movie_ids]

    # Sort and get top N recommendations for each algorithm
    predicted_ratings_svd.sort(key=lambda x: x[1], reverse=True)
    predicted_ratings_knn.sort(key=lambda x: x[1], reverse=True)
    predicted_ratings_nmf.sort(key=lambda x: x[1], reverse=True)

    # Adding pagination mechanism for displaying more recommendations
    items_per_page = 10

    if 'svd_offset' not in st.session_state:
        st.session_state['svd_offset'] = 0
    if 'knn_offset' not in st.session_state:
        st.session_state['knn_offset'] = 0
    if 'nmf_offset' not in st.session_state:
        st.session_state['nmf_offset'] = 0
    if 'worst_svd_offset' not in st.session_state:
        st.session_state['worst_svd_offset'] = 0

    def display_recommendations(recommendations, offset, limit):
        for movie_id, predicted_rating in recommendations[offset:offset + limit]:
            movie_data = df[df['movieId'] == movie_id].iloc[0]
            movie_title = movie_data['title']
            tmdb_id = movie_data['tmdbId']
            poster_url, homepage = get_movie_details_by_id(tmdb_id)
            likelihood_percentage = (predicted_rating / 5) * 100
            st.image(poster_url if poster_url else "https://via.placeholder.com/100x150?text=No+Image", width=100)
            st.markdown(f"[**{movie_title}** - {likelihood_percentage:.2f}% chance you will like it!]({homepage})")

    # Create separate tabs for each algorithm and the worst SVD recommendations
    tab1, tab2, tab3, tab4 = st.tabs(["SVD Recommendations", "KNN Recommendations", "NMF Recommendations", "Worst SVD Recommendations"])

    # SVD Recommendations Tab with pagination
    with tab1:
        st.write("**SVD Algorithm**: Recommendations based on your ratings.")
        display_recommendations(predicted_ratings_svd, st.session_state['svd_offset'], items_per_page)
        if st.button('Display more SVD recommendations'):
            st.session_state['svd_offset'] += items_per_page
            display_recommendations(predicted_ratings_svd, st.session_state['svd_offset'], items_per_page)

    # KNN Recommendations Tab with pagination
    with tab2:
        st.write("**KNN Algorithm**: Recommendations based on movie similarities.")
        display_recommendations(predicted_ratings_knn, st.session_state['knn_offset'], items_per_page)
        if st.button('Display more KNN recommendations'):
            st.session_state['knn_offset'] += items_per_page
            display_recommendations(predicted_ratings_knn, st.session_state['knn_offset'], items_per_page)

    # NMF Recommendations Tab with pagination
    with tab3:
        st.write("**NMF Algorithm**: Recommendations based on latent features.")
        display_recommendations(predicted_ratings_nmf, st.session_state['nmf_offset'], items_per_page)
        if st.button('Display more NMF recommendations'):
            st.session_state['nmf_offset'] += items_per_page
            display_recommendations(predicted_ratings_nmf, st.session_state['nmf_offset'], items_per_page)

    # Worst SVD Recommendations Tab with pagination
    with tab4:
        st.write("**Worst Recommendations (SVD)**: Movies you are least likely to enjoy.")
        predicted_ratings_svd.sort(key=lambda x: x[1])  # Sort by ascending rating to get the worst
        display_recommendations(predicted_ratings_svd, st.session_state['worst_svd_offset'], items_per_page)
        if st.button('Display more Worst SVD recommendations'):
            st.session_state['worst_svd_offset'] += items_per_page
            display_recommendations(predicted_ratings_svd, st.session_state['worst_svd_offset'], items_per_page)

else:
    st.write('Rate some movies to get recommendations!')
