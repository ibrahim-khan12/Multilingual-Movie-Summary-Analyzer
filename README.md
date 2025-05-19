# Movie Genre Prediction and Audio Conversion System
## 📌 Overview
This project is a comprehensive system that:
- Predicts movie genres from textual summaries using machine learning.
- Translates the summaries into Arabic, Urdu, and Korean.
- Converts translated summaries into audio using Text-to-Speech (TTS).
- Provides an interactive command-line interface for user interaction.

# ✨ Features
- ✅ Preprocessing of movie summaries (cleaning, tokenization, lemmatization)
- 🎯 Multi-label genre prediction using machine learning
- 🌐 Multilingual translation: Arabic, Urdu, Korean
- 🔊 Audio conversion with natural pronunciation

# 🧠 Models used:

- models.pkl: Trained Random Forest wrapped in OneVsRestClassifier
- multilabel_binarizer.pkl: For encoding/decoding genre labels
- vectorizer.pkl: TF-IDF vectorizer for feature extraction

# 🧭 Menu-based CLI interface

- 📊 Visualizations: genre distributions, confusion matrices, word clouds
- 🚫 Robust error handling for API and user input


## Clone the repository

- git clone https://github.com/yourusername/movie-genre-audio-system.git
- Create a virtual environment (optional but recommended)
- python -m venv venv
-  source venv/bin/activate  # On Windows: venv\Scripts\activate
## Install dependencies
- pip install -r requirements.txt
## 🚀 Usage
- Run the system using:
python src/main.py
# Menu Options:

===== MOVIE GENRE PREDICTION AND AUDIO CONVERSION SYSTEM =====
1. Enter a movie summary
2. Convert summary to audio
3. Predict movie genre
4. Exit
# 🧠 Model Details
- Model File: models/models.pkl
- Type: RandomForestClassifier wrapped with OneVsRestClassifier
- Training Features: TF-IDF vectorized summaries
- Multi-Label Encoding: multilabel_binarizer.pkl

# 🌐 Language & Audio Support
- Translations via googletrans API
- Audio via gTTS (Google Text-to-Speech)

## Supported Languages:

- Arabic ('ar')
- Urdu ('ur')
- Korean ('ko')

## 🧪 Performance Metrics
- Accuracy: 76.8%
- Macro F1-Score: 77.4%
- Micro F1-Score: 79.2%
## Top Genre Performance:
- Action: 85.3%
- Horror: 88.2%
- Sci-Fi: 84.1%

## 📊Visualizations
- 📈 Genre Distribution Charts
- 🎭 Genre-specific Word Clouds
- 🔍 Confusion Matrices for each genre
## 💡Future Work
- Web-based GUI for broader accessibility
- More language support and TTS engines
- Improved genre classification with larger datasets
- Real-time data integration from IMDB, TMDb APIs

## 📄 License
MIT License © M. ibrahim

## 🤝 Acknowledgements
- CMU Movie Summary Corpus
- Google Translate API
- Google Text-to-Speech (gTTS)
- Scikit-learn, NLTK, Matplotlib, Seaborn
