# 📰 Fake News Detector

An AI-powered web application to classify news articles as "Real" or "Fake". This project uses Natural Language Processing (NLP) techniques and a Passive-Aggressive Classifier model.


## 📚 Project Overview

In today's digital age, misinformation is rampant. This project tackles the challenge of fake news detection by leveraging machine learning. The model is trained on a dataset of over 44,000 news articles and can predict the credibility of a given text with high accuracy. The interactive web interface is built with Gradio and deployed on Hugging Face Spaces.

## 🛠️ Technologies Used
- **Python:** The core programming language.
- **Scikit-learn:** For building and training the machine learning model.
- **Pandas:** For data manipulation and analysis.
- **NLTK:** For natural language processing tasks like stemming and stopword removal.
- **Gradio:** For creating the interactive web interface.
- **Hugging Face Spaces:** For deploying the application.
- **Jupyter/Google Colab:** For exploratory data analysis and model development.

## ⚙️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fake-news-detector.git](https://github.com/your-username/fake-news-detector.git)
    cd fake-news-detector
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:7860`.

## 📂 Project Structure
├── model/                # Contains the saved model and vectorizer
│   ├── model.pkl
│   └── vectorizer.pkl
├── notebooks/            # Contains the Jupyter notebook for analysis
│   └── fake_news_analysis.ipynb
├── .gitignore            # Specifies files for Git to ignore
├── app.py                # The main Gradio application script
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies

