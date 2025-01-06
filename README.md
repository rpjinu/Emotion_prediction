# Emotion_prediction
"Emotional detection prediction use NLP in the project use streamlit to deploy model"
<img src="https://github.com/rpjinu/Emotion_prediction/blob/main/project_image.png" width='800'>
# Emotion Prediction NLP Project

This project is an NLP (Natural Language Processing) application designed to predict the emotion conveyed in a given text input. The emotions classified by the model include happiness, sadness, and angriness. The project leverages machine learning techniques, including text preprocessing, TF-IDF vectorization, and a pre-trained classification model.

## Features

- **Text Input**: Users can input any text to analyze the underlying emotion.
- **Text Preprocessing**: Includes HTML tag removal, special character removal, conversion to lowercase, stopword removal, and stemming.
- **TF-IDF Vectorization**: Transforms the preprocessed text into a numerical representation suitable for model prediction.
- **Emotion Prediction**: Predicts the emotion using a pre-trained classification model.
- **Streamlit Integration**: A simple and interactive web interface for users to input text and view predictions.

## How It Works

1. **Text Preprocessing**:
    - HTML tags and special characters are removed.
    - Text is converted to lowercase.
    - Stopwords are removed.
    - Words are stemmed to their root forms using the Porter Stemmer.

2. **TF-IDF Vectorization**:
    - The cleaned and preprocessed text is transformed into a TF-IDF vector.

3. **Model Prediction**:
    - The TF-IDF vector is fed into a pre-trained machine learning model to predict the emotion.

4. **Result Display**:
    - The predicted emotion is displayed on the Streamlit web interface.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emotion_prediction.git
   cd emotion_prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Navigate to the local URL provided by Streamlit after running the app.
2. Enter the text you want to analyze in the text area.
3. Click on the "Predict" button to see the predicted emotion.

## Model Training

- The model was trained on a labeled dataset with text samples corresponding to different emotions.
- Preprocessing steps included stemming and stopword removal.
- TF-IDF vectorization was used to convert text into numerical features.
- A classification algorithm was trained to predict emotions based on these features.

## Files

- `app.py`: Main Streamlit application file.
- `emotion_model.pkl`: Pre-trained machine learning model.
- `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer.
- `requirements.txt`: List of dependencies required to run the project.

##Deployment in Streamlit image:-
<img src="https://github.com/rpjinu/Emotion_prediction/blob/main/Deploy_image.jpg" width="600">

## Future Enhancements

- Expand the emotion categories to include more nuanced emotions.
- Improve the preprocessing pipeline with advanced NLP techniques.
- Integrate more sophisticated machine learning models or neural networks.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

- **Ranjan**

Feel free to reach out for collaboration or feedback on the project!
