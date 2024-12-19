import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load Index for words from IMDB dataset
word_index = imdb.get_word_index()

# Load the RNN model
model = load_model("models/simplernn.keras")

# Function to preprocess user reviews
def preprocess_review(review):
    review_words = review.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in review_words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen = 500)
    return padded_review

# Function to predict the sentiment
def predict_sentiment(review):
    preprocessed_review = preprocess_review(review)
    output = model.predict(preprocessed_review)
    rounded_output = float(f"{output[0][0]:.2f}")
    sentiment = "Positive" if rounded_output > 0.5 else "Negative"
    return sentiment, rounded_output

## streamlit app
st.title("Sentiment Analysis on IMDB Movie Reviews ")
st.write(" Enter a movie review to determine if it's **Positive** or **Negative**.")

# user input
input = st.text_area("Movie Review")

# prediction
if st.button("Submit"):
    if input.strip():
        sentiment, output = predict_sentiment(input)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {output}')
    else:
        st.write("Please enter a valid movie review.")


