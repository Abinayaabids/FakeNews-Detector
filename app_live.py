# app_live.py
import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

st.title("🌐 Live Fake News Detector")

# Step 1: Load and combine datasets
@st.cache_data
def load_dataset():
    fake = pd.read_csv("fake.csv")
    true = pd.read_csv("true.csv")
    fake['label'] = 'FAKE'
    true['label'] = 'REAL'
    df = pd.concat([fake, true])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

df = load_dataset()

# Step 2: Train the model
@st.cache_resource
def train_model(df):
    X = df['text']  # or 'title' for headlines only
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(tfidf_train, y_train)
    
    return model, tfidf_vectorizer

model, tfidf_vectorizer = train_model(df)

# Step 3: NewsAPI input
api_key = st.text_input("Enter your NewsAPI.org API key:")

country_code = st.selectbox("Select Country for live news:", ['ae','us','ir','gb'])

if api_key:
    url = f"https://newsapi.org/v2/top-headlines?country={country_code}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    headlines = [article['title'] for article in data.get('articles', [])]

    if len(headlines) == 0:
        st.warning("No headlines found. Try another country or check your API key.")
    else:
        st.write(f"Found {len(headlines)} latest headlines:")

        for headline in headlines:
            prediction = model.predict(tfidf_vectorizer.transform([headline]))[0]
            st.write(f"📰 {headline}\n👉 Prediction: {prediction}\n")