import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained model and TfidfVectorizer
with open(r'C:\Users\Ranjan kumar pradhan\.vscode\emotion_prediction\best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open(r'C:\Users\Ranjan kumar pradhan\.vscode\emotion_prediction\vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Initialize the Porter Stemmer
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Lowercase the content
    stemmed_content = stemmed_content.lower()
    # Split the content into words
    stemmed_content = stemmed_content.split()
    # Remove stopwords and stem the words
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    # Join the words back into a single string
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Streamlit app
def main():
    st.title("Emotion Prediction API")
    st.write("Enter a text to predict the emotion.")
    
    user_input = st.text_area("Enter your text:")
    
    if st.button("Predict"):
        if user_input:
            # Preprocess the text input
            cleaned_text = preprocess_text(user_input)
            processed_text = stemming(cleaned_text)
            
            # Transform the input text using TfidfVectorizer
            text_vectorized = tfidf_vectorizer.transform([processed_text])
            
            # Predict the emotion
            prediction = model.predict(text_vectorized)[0]
            
            # Mapping of prediction to emotion
            emotion_mapping = {0: 'Happiness', 1: 'Sadness', 2: 'Angriness'}
            
            # Display the result
            st.write(f"Predicted Emotion: {emotion_mapping[prediction]}")
        else:
            st.write("Please enter some text to predict the emotion.")

if __name__ == '__main__':
    main()
