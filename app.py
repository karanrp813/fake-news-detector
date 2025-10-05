import streamlit as st
import pandas as pd
from newsapi import NewsApiClient
import re
import spacy
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from collections import Counter

# --- App Configuration ---
st.set_page_config(
    page_title="Health Trend Spotter",
    page_icon="⚕️",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False) # To hide a common warning with pyplot

st.title("⚕️ Health Trend Spotter")
st.write("An application to analyze public health trends from news articles using NLP and time-series forecasting.")

# --- Caching for Performance ---
# Cache loading of models to avoid reloading on every interaction
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

nlp = load_spacy_model()
sentiment_pipeline = load_sentiment_pipeline()

# --- Data Fetching and Processing ---
@st.cache_data # Cache the data fetching and processing function
def fetch_and_process_data(api_key):
    # 1. Fetch Data from NewsAPI
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(
        q='flu OR vaccine OR pandemic',
        language='en',
        sort_by='relevancy',
        page_size=100
    )
    articles_list = all_articles['articles']

    processed_articles = []
    for article in articles_list:
        text_content = f"{article['title']}. {article['description']}"
        processed_articles.append({
            'date': article['publishedAt'],
            'text': text_content
        })

    df = pd.DataFrame(processed_articles)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.drop_duplicates(subset=['text'], keep='first').dropna(subset=['text'])

    # 2. Clean Text for NLP
    def process_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        doc = nlp(text)
        clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip()]
        return " ".join(clean_tokens)

    df['cleaned_text'] = df['text'].apply(process_text)
    return df

# --- Main App Logic ---
try:
    # Use the secret key from Hugging Face settings
    api_key = st.secrets["NEWS_API_KEY"]
    
    with st.spinner("Fetching and analyzing the latest news articles... This may take a moment."):
        df = fetch_and_process_data(api_key)

    st.success(f"Successfully fetched and processed {len(df)} unique articles.")

    # 3. Sentiment Analysis
    st.header("Sentiment Analysis of News Articles")
    text_list = df['cleaned_text'].tolist()
    sentiment_results = sentiment_pipeline(text_list)
    df['sentiment_label'] = [result['label'] for result in sentiment_results]

    sentiment_counts = df['sentiment_label'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Sentiment Distribution:")
        st.dataframe(sentiment_counts)
    with col2:
        st.write("Sentiment Visualization:")
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
        ax.set_title('Distribution of Sentiment')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Number of Articles')
        st.pyplot(fig)

    # 4. Named Entity Recognition (NER)
    st.header("Key Entity Recognition")
    st.write("Identifying the most mentioned organizations (ORG) and locations (GPE).")
    def extract_key_entities(text):
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'GPE']]
        return entities

    df['entities'] = df['text'].apply(extract_key_entities)
    all_entities = [entity for sublist in df['entities'] for entity in sublist]
    entity_counts = Counter(all_entities)
    most_common_entities = entity_counts.most_common(15)
    
    df_entities = pd.DataFrame(most_common_entities, columns=['Entity', 'Count'])

    fig_entities, ax_entities = plt.subplots(figsize=(10, 8))
    df_entities.sort_values(by='Count').plot.barh(x='Entity', y='Count', ax=ax_entities, color='teal')
    ax_entities.set_title('Top 15 Most Mentioned Organizations and Locations')
    st.pyplot(fig_entities)

    # 5. Time-Series Forecasting with Prophet
    st.header("Forecasting Negative Sentiment Trends")
    st.write("Using Prophet to forecast the daily count of negative-sentiment articles for the next 30 days.")

    df_negative = df[df['sentiment_label'] == 'negative'].copy()
    daily_counts = df_negative.groupby('date').size().reset_index(name='count')
    prophet_df = daily_counts.rename(columns={'date': 'ds', 'count': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    if len(prophet_df) > 2: # Prophet needs at least 2 data points
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        plt.title('Forecast of Daily Negative-Sentiment Articles')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        st.pyplot(fig1)
        
        st.write("Forecast Components:")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
    else:
        st.warning("Not enough historical data with negative sentiment to generate a forecast.")


except KeyError:
    st.error("NEWS_API_KEY not found. Please add it to your Hugging Face Space secrets in the 'Settings' tab.")
except Exception as e:
    st.error(f"An error occurred: {e}")