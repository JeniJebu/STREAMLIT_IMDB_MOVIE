import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')


st.title("Movie Success Prediction")
st.write("Enter the details of the movie:")

# Group 1: Review and Popularity
st.header("Review and Popularity")
col1, col2 = st.columns(2)
with col1:
    num_critic_for_reviews = st.number_input('Number of Critic Reviews', min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
    num_user_for_reviews = st.number_input('Number of User Reviews', min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
with col2:
    num_voted_users = st.number_input('Number of Voted Users', min_value=0.0, max_value=1e7, value=50000.0, step=1000.0)

# Group 2: Social Media Likes
st.header("Social Media Likes")
col1, col2 = st.columns(2)
with col1:
    director_facebook_likes = st.number_input('Director Facebook Likes', min_value=0.0, max_value=1e6, value=5000.0, step=100.0)
    actor_1_facebook_likes = st.number_input('Actor 1 Facebook Likes', min_value=0.0, max_value=1e6, value=5000.0, step=100.0)
    actor_3_facebook_likes = st.number_input('Actor 3 Facebook Likes', min_value=0.0, max_value=1e6, value=5000.0, step=100.0)
with col2:
    actor_2_facebook_likes = st.number_input('Actor 2 Facebook Likes', min_value=0.0, max_value=1e6, value=5000.0, step=100.0)
    movie_facebook_likes = st.number_input('Movie Facebook Likes', min_value=0.0, max_value=1e6, value=5000.0, step=100.0)

# Group 3: Movie Details
st.header("Movie Details")
col1, col2 = st.columns(2)
with col1:
    duration = st.number_input('Duration (in minutes)', min_value=0.0, max_value=500.0, value=120.0, step=1.0)
    facenumber_in_poster = st.number_input('Number of Faces in Poster', min_value=0.0, max_value=10.0, value=1.0, step=1.0)
with col2:
    title_year = st.number_input('Title Year', min_value=1900.0, max_value=2100.0, value=2020.0, step=1.0)

# Group 4: Financials
st.header("Financials")
col1, col2 = st.columns(2)
with col1:
    gross = st.number_input('Gross Earnings', min_value=0.0, max_value=1e10, value=1e6, step=1e4)
with col2:
    budget = st.number_input('Budget', min_value=0.0, max_value=1e10, value=1e6, step=1e4)

# Center the Predict button using CSS
st.markdown("""
    <style>
    .stButton>button {
        display: block;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

if st.button('Predict'):
    # Prepare input data for prediction
    input_array = np.array([[num_critic_for_reviews, num_user_for_reviews, num_voted_users,
                             director_facebook_likes, actor_1_facebook_likes,actor_3_facebook_likes,
                             actor_2_facebook_likes, movie_facebook_likes, duration,  facenumber_in_poster,
                             title_year, gross,  budget,  ]])
  

                            
    # Make prediction
    prediction = model.predict(input_array)

    # Debug information
    st.write(f"Prediction output: {prediction}")

    # Map prediction to success or not
    if 'Poor' in prediction:  # If 'Poor' is predicted
        st.write("The movie is predicted not to be a success!")
    else:  # If only 'Average' or 'Good' is predicted
        st.success("The movie is predicted to be a success!")