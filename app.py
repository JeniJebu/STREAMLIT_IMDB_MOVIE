import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('best_random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

st.title("Movie Success Prediction")
st.write("Enter the details of the movie:")

col1, col2 = st.columns(2)

# Input fields for user to enter the movie details
with col1:
    num_critic_for_reviews = st.number_input("Number of Critic Reviews", min_value=2.0, max_value=813.0, value=138.0)
    duration = st.number_input("Duration (minutes)", min_value=37.0, max_value=330.0, value=106.0)
    director_facebook_likes = st.number_input("Director Facebook Likes", min_value=0.0, max_value=23000.0, value=63.0)
    actor_3_facebook_likes = st.number_input("Actor 3 Facebook Likes", min_value=0.0, max_value=23000.0, value=436.0)
    actor_1_facebook_likes = st.number_input("Actor 1 Facebook Likes", min_value=0.0, max_value=640000.0, value=1000.0)
    gross = st.number_input("Gross Revenue", min_value=162.0, max_value=760505800.0, value=30054710.5)
    num_voted_users = st.number_input("Number of Voted Users", min_value=91.0, max_value=1689764.0, value=53993.5)

with col2:
    facenumber_in_poster = st.number_input("Face Number in Poster", min_value=0.0, max_value=43.0, value=1.0)
    num_user_for_reviews = st.number_input("Number of User Reviews", min_value=4.0, max_value=5060.0, value=209.5)
    budget = st.number_input("Budget", min_value=218.0, max_value=12215500000.0, value=25000000.0)
    title_year = st.number_input("Title Year", min_value=1927.0, max_value=2016.0, value=2004.0)
    actor_2_facebook_likes = st.number_input("Actor 2 Facebook Likes", min_value=0.0, max_value=137000.0, value=683.0)
    movie_facebook_likes = st.number_input("Movie Facebook Likes", min_value=0.0, max_value=349000.0, value=225.5)
    
    
# Creating empty columns around the button for centering
col3, col4, col5 = st.columns([2, 1, 2])

with col4:
    submitted = st.button('Predict')

if submitted:
    user_features = np.array([
        num_critic_for_reviews, duration, director_facebook_likes,
        actor_3_facebook_likes, actor_1_facebook_likes, gross,
        num_voted_users, facenumber_in_poster, num_user_for_reviews, 
        budget, title_year, actor_2_facebook_likes, movie_facebook_likes,
    ]).reshape(1, -1)
    
    predicted_category = loaded_model.predict(user_features)[0]
    labels = ['Poor', 'Average', 'Good']
    
    # Convert predicted_category to integer or use directly if already numeric
    if isinstance(predicted_category, str):
        predicted_category = labels.index(predicted_category)
    predicted_label = labels[predicted_category]

    st.write(f"Predicted Category: {predicted_label}")

    if predicted_label == 'Poor':
        st.error("The movie is predicted not to be a success!")
    else:
        st.success("The movie is predicted to be a success!")