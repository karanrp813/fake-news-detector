# app.py
import gradio as gr
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# --- Load Models and Preprocessing Objects ---

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the saved vectorizer and model
vectorizer = joblib.load('model/vectorizer.pkl')
model = joblib.load('model/model.pkl')

# Initialize Stemmer and Stopwords
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Preprocessing Function ---

def preprocess_text(content):
    """Cleans and preprocesses the input text."""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# --- Prediction Function ---

def predict_news(text):
    """Predicts if a news article is real or fake."""
    if not text.strip():
        return "Please enter some text."

    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    
    if prediction[0] == 1:
        return {"Fake News": 0.99, "Real News": 0.01} # Example confidence
    else:
        return {"Real News": 0.99, "Fake News": 0.01}

# --- Gradio Interface ---

interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=10, placeholder="Paste the full news article text here..."),
    outputs=gr.Label(label="Prediction", num_top_classes=2),
    title="📰 Fake News Detector",
    description="An AI-powered tool to distinguish between real and fake news. It uses a Passive-Aggressive Classifier trained on over 44,000 news articles. Paste an article's text to check its credibility.",
    examples=[
        ["The Centers for Disease Control and Prevention is advising Americans to prepare for the possibility of a COVID-19 outbreak in the United States, officials said Tuesday."],
        ["BREAKING: Mueller Deputy Walks Out Of Trump Hearing In Disgust (VIDEO)"]
    ]
)

# --- Launch the App ---

if __name__ == "__main__":
    interface.launch()
