import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- 1. Synthetic Dataset Creation ---
# We create a small dataset of movies and user ratings to make this app self-contained.

@st.cache_data
def load_data():
    # Movies Data: Content (Title, Genres, Overview)
    movies_data = {
        'MovieID': range(1, 21),
        'Title': [
            'The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Avengers: Endgame',
            'The Godfather', 'Pulp Fiction', 'Fight Club', 'Forrest Gump', 'The Shawshank Redemption',
            'Titanic', 'The Notebook', 'La La Land', 'Beauty and the Beast', 'Frozen',
            'Get Out', 'The Conjuring', 'It', 'A Quiet Place', 'Hereditary'
        ],
        'Genres': [
            'Sci-Fi Action', 'Sci-Fi Thriller', 'Sci-Fi Adventure', 'Action Crime', 'Action Adventure',
            'Crime Drama', 'Crime Drama', 'Drama Thriller', 'Drama Romance', 'Drama',
            'Romance Drama', 'Romance Drama', 'Romance Musical', 'Fantasy Musical', 'Animation Adventure',
            'Horror Thriller', 'Horror Supernatural', 'Horror Thriller', 'Horror Sci-Fi', 'Horror Drama'
        ],
        'Overview': [
            'A computer hacker learns about the true nature of his reality and his role in the war against its controllers.',
            'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea.',
            'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological tests.',
            'After the devastating events of Infinity War, the universe is in ruins.',
            'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine.',
            'An insomniac office worker and a devil-may-care soap maker form an underground fight club.',
            'The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate and other historical events unfold from the perspective of an Alabama man.',
            'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
            'A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom.',
            'While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.',
            'A selfish Prince is cursed to become a monster for the rest of his life, unless he learns to fall in love with a beautiful young woman.',
            'When the newly crowned Queen Elsa accidentally uses her power to turn things into ice to curse her home in infinite winter.',
            'A young African-American visits his white girlfriend\'s parents for the weekend, where his simmering uneasiness about their reception reaches a boiling point.',
            'Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse.',
            'In the summer of 1989, a group of bullied kids band together to destroy a shape-shifting monster, which disguises itself as a clown.',
            'In a post-apocalyptic world, a family is forced to live in silence while hiding from monsters with ultra-sensitive hearing.',
            'A grieving family is haunted by tragic and disturbing occurrences.'
        ]
    }
    movies_df = pd.DataFrame(movies_data)

    # Ratings Data: User-Item Interactions (UserID, MovieID, Rating 1-5)
    # Creating a pattern: User 1 likes Sci-Fi, User 2 likes Romance, User 3 likes Horror
    ratings_data = {
        'UserID': [
            1, 1, 1, 1, 1,  # Sci-Fi/Action Fan
            2, 2, 2, 2, 2,  # Romance/Drama Fan
            3, 3, 3, 3, 3,  # Horror Fan
            4, 4, 4, 4, 4   # Mixed/Popularity focus
        ],
        'MovieID': [
            1, 2, 3, 4, 5,   # Matrix, Inception, Interstellar...
            11, 12, 13, 14, 10, # Titanic, Notebook, La La Land...
            16, 17, 18, 19, 20, # Get Out, Conjuring...
            1, 11, 16, 6, 8    # Hits from various genres
        ],
        'Rating': [
            5, 5, 4, 5, 4,
            5, 5, 4, 3, 5,
            5, 4, 5, 4, 5,
            5, 5, 4, 5, 5
        ]
    }
    ratings_df = pd.DataFrame(ratings_data)
    
    return movies_df, ratings_df

movies, ratings = load_data()

# --- 2. Recommendation Engines ---

def get_popularity_recommendations(n=5):
    """
    Approach 1: Popularity Based
    Returns movies with the highest average rating and count.
    """
    # Calculate average rating and count of ratings
    movie_stats = ratings.groupby('MovieID').agg({'Rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'rating_count']
    
    # Merge with movie titles
    popular_movies = pd.merge(movie_stats, movies, on='MovieID')
    
    # Sort by average rating then count
    popular_movies = popular_movies.sort_values(['avg_rating', 'rating_count'], ascending=False)
    
    return popular_movies.head(n)[['Title', 'Genres', 'avg_rating', 'Overview']]

def get_content_based_recommendations(movie_title, n=5):
    """
    Approach 2: Content-Based Filtering
    Uses TF-IDF on Movie Overviews/Genres to find similar movies.
    """
    # Combine genres and overview for richer features
    movies['combined_features'] = movies['Genres'] + " " + movies['Overview']
    
    # Create TF-IDF Matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    
    # Calculate Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get index of the movie that matches the title
    try:
        idx = movies[movies['Title'] == movie_title].index[0]
    except IndexError:
        return None

    # Get similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top n similar movies (ignoring the movie itself)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies.iloc[movie_indices][['Title', 'Genres', 'Overview']]

def get_collaborative_recommendations(user_id, n=5):
    """
    Approach 3: Collaborative Filtering (User-Based)
    Finds users similar to the target user and recommends what they liked.
    """
    # Create User-Item Matrix
    user_movie_matrix = ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)
    
    # Calculate User Similarity (Cosine Similarity)
    user_similarity = cosine_similarity(user_movie_matrix)
    user_sim_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    
    # Get similar users (excluding the user themselves)
    if user_id not in user_sim_df.index:
        return pd.DataFrame(columns=['Title', 'Genres']) # User has no data
        
    similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1:]
    
    # Find movies highly rated by similar users that the target user hasn't seen
    user_seen_movies = ratings[ratings['UserID'] == user_id]['MovieID'].tolist()
    recommendations = []
    
    for similar_user in similar_users:
        # Get movies rated 4 or 5 by the similar user
        sim_user_ratings = ratings[(ratings['UserID'] == similar_user) & (ratings['Rating'] >= 4)]
        
        for _, row in sim_user_ratings.iterrows():
            if row['MovieID'] not in user_seen_movies and row['MovieID'] not in [r['MovieID'] for r in recommendations]:
                movie_info = movies[movies['MovieID'] == row['MovieID']].iloc[0]
                recommendations.append({
                    'Title': movie_info['Title'],
                    'Genres': movie_info['Genres'],
                    'Overview': movie_info['Overview'],
                    'MovieID': row['MovieID'],
                    'Recommended By User': similar_user
                })
                
                if len(recommendations) >= n:
                    break
        if len(recommendations) >= n:
            break
            
    return pd.DataFrame(recommendations)

# --- 3. Streamlit UI Layout ---

st.title("🎬 Movie Recommendation System")
st.markdown("""
This app demonstrates three standard approaches to Recommendation Systems using a synthetic movie dataset.
""")

# Sidebar for Controls
st.sidebar.header("User Settings")
selected_user_id = st.sidebar.selectbox("Select User ID (for Collaborative Filtering)", ratings['UserID'].unique())

st.sidebar.header("Content Settings")
selected_movie_title = st.sidebar.selectbox("Select a Movie (for Content-Based)", movies['Title'].values)

# Tabs for the Three Approaches
tab1, tab2, tab3 = st.tabs(["🔥 Popularity-Based", "📚 Content-Based", "🤝 Collaborative Filtering"])

# --- Tab 1: Popularity Based ---
with tab1:
    st.header("Top Rated Movies")
    st.write("These movies are recommended based on highest average ratings across all users. Good for new users (Cold Start problem).")
    
    pop_recs = get_popularity_recommendations()
    
    for index, row in pop_recs.iterrows():
        with st.expander(f"#{index+1} {row['Title']} (⭐ {row['avg_rating']:.1f})"):
            st.write(f"**Genre:** {row['Genres']}")
            st.write(f"**Overview:** {row['Overview']}")

# --- Tab 2: Content Based ---
with tab2:
    st.header(f"Because you liked '{selected_movie_title}'...")
    st.write("These recommendations are based on matching **Genres** and **Plot Descriptions** (TF-IDF & Cosine Similarity).")
    
    content_recs = get_content_based_recommendations(selected_movie_title)
    
    if content_recs is not None and not content_recs.empty:
        cols = st.columns(3)
        for idx, (index, row) in enumerate(content_recs.iterrows()):
            with cols[idx % 3]:
                st.info(f"**{row['Title']}**")
                st.caption(f"_{row['Genres']}_")
                st.write(row['Overview'][:100] + "...")
    else:
        st.error("Could not find recommendations.")

# --- Tab 3: Collaborative Filtering ---
with tab3:
    st.header(f"Recommendations for User {selected_user_id}")
    st.write("These recommendations are based on **similar users**. We look at who votes similarly to you and suggest movies they liked that you haven't seen yet.")
    
    collab_recs = get_collaborative_recommendations(selected_user_id)
    
    # Display "User Profile" first to explain why
    st.subheader("Your Taste (Movies you rated high):")
    user_history = ratings[(ratings['UserID'] == selected_user_id) & (ratings['Rating'] >= 4)].merge(movies, on='MovieID')
    st.write(", ".join(user_history['Title'].tolist()))
    st.divider()
    
    st.subheader("Recommended for you:")
    if not collab_recs.empty:
        for index, row in collab_recs.iterrows():
            st.success(f"**{row['Title']}**")
            st.write(f"**Genre:** {row['Genres']}")
            st.caption(f"Recommended because similar User {row['Recommended By User']} liked it.")
    else:
        st.warning("No new recommendations found. You might have seen everything similar users liked, or there isn't enough data yet!")

# --- Footer ---
st.markdown("---")
st.markdown("Created with Streamlit • Pandas • Scikit-Learn")
