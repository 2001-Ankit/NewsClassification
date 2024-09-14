import nltk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import re
import tensorflow as tf
# from keras.api.models import load_model
from keras.api.models import load_model
from keras.api.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


nltk.download('punkt')
app = FastAPI()

# Load the models and label encoder
model_lstm = load_model('model_lstm_v2.keras')
model_rnn = load_model('model_rnn_v2.keras')
model_gru = load_model('model_gru_v2.keras')
with open('label_encoder_v2.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)
word2vec_model = Word2Vec.load('word2vec_model_v2.model')

nepali_stopwords = [
    'यो', 'छ', 'र', 'थियो', 'हुने', 'मा', 'लाई', 'पनि', 'भएको', 'तथा', 'यस', 'तर', 'कि', 'उनको', 'भन्ने',
    'को', 'वा', 'रूपमा', 'को', 'नै', 'हो', 'गरेको', 'गर्ने', 'उनले', 'हुन्छ', 'गरे', 'गर्न', 'के', 'संग', 'गरेका',
    'अझै', 'अथवा', 'अर्थात', 'अर्को', 'अगाडी', 'अझ', 'आज', 'अनुसार', 'अन्य', 'अभि', 'अवस्था',
    'आधा', 'आदि', 'आजकल', 'आपको', 'आफ्नै', 'आफ्नो', 'आफू', 'आफैं', 'आवश्यक', 'इ', 'इन',
    'उनी', 'उनीहरू', 'उनको', 'उनकै', 'उहाँ', 'ए', 'एक', 'एउटा', 'कति', 'कसैले', 'कहाँ', 'का',
    'कि', 'कुनै', 'कुनैपनि', 'के', 'कसैलाई', 'कसले', 'को', 'कुन', 'किन', 'कि', 'कृपया', 'खास',
    'खासगरी', 'गरे', 'गरेको', 'गर्न', 'गरेका', 'गर्ने', 'गरेपछि', 'गर्नु', 'गर्छ', 'गर्छन्', 'गर्नेछन्',
    'गर्छु', 'गर्दा', 'गर्दछ', 'गर्‍यो', 'गइरहेको', 'गैर', 'घटना', 'चलिरहेको', 'चलाउँछ', 'छ',
    'छैन', 'चाहन्छ', 'जस्तो', 'जब', 'जसले', 'जसको', 'जहाँ', 'जस्तै', 'जसरी', 'जस', 'जसलाई',
    'जस्तोसुकै', 'जति', 'जसमा', 'जसले', 'जससँग', 'जुन', 'जुनसुकै', 'जुनसुकै', 'तत्काल', 'तपाईं',
    'तपाईँ', 'तपाई', 'तपाईंको', 'तपाईँको', 'तपाईको', 'तिमी', 'तिम्रो', 'तिमीले', 'तपाईँले', 'तर',
    'तथा', 'त्यस', 'त्यसको', 'त्यसैले', 'त्यसो', 'त्यस्तै', 'त्यसपछि', 'तिनका', 'तिनी', 'तिनीहरु',
    'तिनीहरूको', 'तिनको', 'तिमीहरु', 'त्यो', 'तिनीलाई', 'तपाईंहरू', 'तपाईंले', 'तिमील', 'तिम्रा',
    'तिनै', 'त्यति', 'थियो', 'थिए', 'थिईन', 'थिएन', 'थियो', 'दुई', 'देखि', 'देखि', 'दिए', 'दिने',
    'दिएको', 'दिएको', 'देखा', 'दुबै', 'दोश्रो', 'न', 'नभएको', 'नभएपछि', 'नभन्ने', 'नजर',
    'नजिकै', 'नत्र', 'नयाँ', 'न', 'नहुने', 'नहि', 'निश्चित', 'नया', 'निको', 'निको', 'नियम',
    'पनि', 'पहिलो', 'परेको', 'पर्याप्त', 'पर्दछ', 'पर्छ', 'परेका', 'पहिले', 'प्राय', 'परेर',
    'पटक', 'पनि', 'फेरि', 'बनी', 'बनाइ', 'बनाइएको', 'बनाई', 'बनाएको', 'बने', 'बन्न', 'बलियो',
    'बनाउने', 'बन्दै', 'बस', 'बीच', 'बारे', 'भरि', 'भर', 'भए', 'भएर', 'भएको', 'भन्छ', 'भन्ने',
    'भएकोले', 'भने', 'भनिन्छ', 'भनेपछि', 'भयो', 'म', 'माथि', 'मात्रै', 'मात्र', 'मेरो', 'मैले',
    'माझ', 'मात्र', 'मध्ये', 'माथिको', 'मात्र', 'माफ', 'मेरा', 'मै', 'मसँग', 'यति', 'यदि', 'यद्यपि',
    'यहाँ', 'यही', 'यहीँ', 'यस', 'यसबारे', 'यसको', 'यसले', 'यसमा', 'यस्तो', 'यसै', 'या', 'यो',
    'र', 'रख', 'रहेका', 'रहेछ', 'रह्यो', 'रहने', 'राखेको', 'रहेको', 'रखिएको', 'राख्ने', 'रहने',
    'रखिएको', 'राख्न', 'लगायत', 'लिएर', 'लिए', 'लिएपछि', 'लिएर', 'लिन्छ', 'ले', 'लेख',
    'लेख्ने', 'लागि', 'लगायतका', 'लिएर', 'लाई', 'लिएर', 'लिएर', 'वा', 'शायद', 'सक्छ', 'सक्ने',
    'सबै', 'सबैका', 'सुरु', 'समेत', 'सधै', 'सँग', 'साथै', 'सक्छन्', 'समय', 'सकिन्छ', 'सक्ने',
    'सबैभन्दा', 'सम्बन्धित', 'सम्भव', 'सो', 'सोही', 'सक्नु', 'सम्म', 'सय', 'सयौं', 'सवै',
    'सीधा', 'सम्बन्धित', 'सक्ने', 'सम्पूर्ण', 'सरोकार', 'हाल', 'हालै', 'हालसम्म', 'हिजो',
    'हुने', 'हुनु', 'हुनसक्छ', 'हुन', 'हुँदै', 'हुनुहुन्छ', 'हुन्छ', 'हुनसक्ने', 'हुने',
    'हुनेछ', 'हुनुहुन्छ', 'हुन्', 'हेर्न', 'होस्', 'हो', 'होला', 'होइन', 'हुँदा', 'हुँदैन',
    'हुन्छ', 'हुनु', 'हुनेछन्', 'हुन्छ', 'हुँदैछ', 'हुनुहुन्छ', 'हुँदा', 'हुने', 'हुँदैन'
]

# Define the input data model
class NewsText(BaseModel):
  text: str
from snowballstemmer import stemmer

def clean_and_stem_nepali_text(text):
  """
  Cleans and stems Nepali text.

  Args:
    text: The input Nepali text.

  Returns:
    The cleaned and stemmed text.
  """
  # Remove special characters
  text = str(text)
  text = re.sub(r'[^\w\s।?!]', '', text)
  # Tokenize the text
  tokens = word_tokenize(text)
  # Stem the tokens
  nepali_stemmer = stemmer("nepali")
  stemmed_tokens = [nepali_stemmer.stemWord(token) for token in tokens]
  # Join the stemmed tokens back into a string
  cleaned_text = " ".join(stemmed_tokens)
  return cleaned_text



def remove_digits_and_symbols(text):
  """
  Removes digits and the symbol '।' from the given text.

  Args:
    text: The input text.

  Returns:
    The text with digits and '।' removed.
  """
  text = re.sub(r'\d+', '', text)  # Remove digits
  text = text.replace('।', '')  # Remove '।'
  return text

# Remove stop words
def remove_stopwords(text):
  tokens = word_tokenize(text)
  filtered_tokens = [token for token in tokens if token not in nepali_stopwords]
  return " ".join(filtered_tokens)

# Define a function to preprocess the input text
def preprocess_new_text(text, word2vec_model, max_len):
  """
  Preprocesses new text for prediction.

  Args:
    text: The new text to preprocess.
    word2vec_model: The trained Word2Vec model.
    max_len: The maximum length of the padded sequence.

  Returns:
    A padded sequence of indices.
  """
  # Clean and stem the text (assuming clean_and_stem_nepali_text is defined elsewhere)
  cleaned_text = clean_and_stem_nepali_text(text)

  # Remove stop words (assuming remove_stopwords is defined elsewhere)
  cleaned_text = remove_stopwords(cleaned_text)

  # Remove digits and symbols (assuming remove_digits_and_symbols is defined elsewhere)
  cleaned_text = remove_digits_and_symbols(cleaned_text)


  # Tokenize the text
  tokens = word_tokenize(cleaned_text)

  # Convert tokens to indices
  indices = [word2vec_model.wv.key_to_index.get(word, 0) for word in tokens]

  # Pad the sequence
  padded_sequence = pad_sequences([indices], maxlen=max_len)

  return padded_sequence


import logging

@app.post("/predict_all")
async def predict_all(news_text: NewsText):
    text = news_text.text
    try:
        padded_sequence = preprocess_new_text(text, word2vec_model, max_len=200)

        # Predict with LSTM model
        lstm_probabilities = model_lstm.predict(padded_sequence)
        lstm_predicted_class_index = np.argmax(lstm_probabilities)
        lstm_predicted_class = label_encoder.classes_[lstm_predicted_class_index]
        lstm_class_probabilities = dict(zip(label_encoder.classes_, lstm_probabilities[0].tolist()))

        # Predict with RNN model
        rnn_probabilities = model_rnn.predict(padded_sequence)
        rnn_predicted_class_index = np.argmax(rnn_probabilities)
        rnn_predicted_class = label_encoder.classes_[rnn_predicted_class_index]
        rnn_class_probabilities = dict(zip(label_encoder.classes_, rnn_probabilities[0].tolist()))

        # Predict with GRU model
        gru_probabilities = model_gru.predict(padded_sequence)
        gru_predicted_class_index = np.argmax(gru_probabilities)
        gru_predicted_class = label_encoder.classes_[gru_predicted_class_index]
        gru_class_probabilities = dict(zip(label_encoder.classes_, gru_probabilities[0].tolist()))

        return {
            "lstm": {
                "predicted_class": lstm_predicted_class,
                "class_probabilities": lstm_class_probabilities
            },
            "rnn": {
                "predicted_class": rnn_predicted_class,
                "class_probabilities": rnn_class_probabilities
            },
            "gru": {
                "predicted_class": gru_predicted_class,
                "class_probabilities": gru_class_probabilities
            }
        }
    except Exception as e:
        logging.error(f"Error in predict_all: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")



# Run the application
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
