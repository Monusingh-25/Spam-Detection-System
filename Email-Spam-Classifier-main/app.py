import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# These lines ensure the necessary NLTK data is available for deployment
import nltk
import streamlit as st
import os

@st.cache_resource
def download_nltk_data():
    nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
    nltk.download('punkt', download_dir='nltk_data')
    nltk.download('punkt_tab', download_dir='nltk_data')
    nltk.download('stopwords', download_dir='nltk_data')

download_nltk_data()



nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    y.clear()
    
    y = [ps.stem(i) for i in text]

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app interface
st.title("Spam Message Classifier")
st.write("Enter a message to check if it's spam or not.")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("This is a Spam message.")
        else:
            st.header("This is Not a Spam message.")
    else:
        st.warning("Please enter a message to predict.")
        