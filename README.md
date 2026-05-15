# Email Spam Classifier

A machine learning-based web application that classifies email/spam messages in real-time.

## What This Project Does

- Takes any message as input
- Preprocesses the text (lowercasing, tokenization, removing stopwords and punctuation, stemming)
- Converts text to numerical features using TF-IDF vectorization
- Predicts whether the message is Spam or Not-Spam using a trained ML model

## Problem Solves

Automatically identifies unwanted/spam emails and messages, helping users filter out malicious or irrelevant content before reading.

## Tools & Technologies Used

- **Streamlit** — Web UI framework for building the interactive interface
- **scikit-learn** — ML library for the classification model (TF-IDF + classifier)
- **NLTK** — Natural Language Toolkit for text preprocessing (tokenization, stopwords, stemming)
- **Python** — Core programming language

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```
