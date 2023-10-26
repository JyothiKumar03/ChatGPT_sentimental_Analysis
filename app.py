# from flask import Flask, render_template, request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from custom_optimizer import CustomAdam
# from keras.utils import get_custom_objects
from flask import Flask, render_template, request
# import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing import sequence, text
# import os

app = Flask(__name__)

# class CustomAdam(Adam):
#     pass

# Load the model with custom_objects argument
# custom_objects = {'CustomAdam': CustomAdam}
lstm_model = load_model('lstm_model.h5', custom_objects={'CustomAdam': CustomAdam},compile=False)

# Register the custom optimizer
# get_custom_objects().update({'CustomAdam': CustomAdam})

# Load the model with custom_objects argument
# lstm_model = load_model('lstm_model.h5')
# lstm_model = load_model('lstm_model.h5', custom_objects={'CustomAdam': CustomAdam})


# Initialize NLTK lemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Assuming you have a list of text data
texts = [
    "High quality pants. Very comfortable and great for sport activities. Good price for nice quality! I recommend to all fans of sports",
    "And this is the second text.",
    "Another example text."
]

# Create and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Save the tokenizer to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower()  # Convert to lowercase for consistency

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']

        # Clean, tokenize, and lemmatize the input tweet
        processed_tweet = clean_text(tweet)
        processed_tweet = word_tokenize(processed_tweet)
        processed_tweet = lemmatize_tokens(processed_tweet)

        # Convert the processed tweet to sequences and pad
        sequences = tokenizer.texts_to_sequences([processed_tweet])
        data = pad_sequences(sequences, maxlen=100)  # Assuming max_len is 100

        # Make predictions using the LSTM model
        output = lstm_model.predict(data)
        label_id = tf.argmax(output, axis=1).numpy()[0]
        sentiment_label = sentiment_category(label_id)

        return render_template('result.html', tweet=tweet, sentiment=sentiment_label)

def sentiment_category(label_id):
    if label_id == 0:
        return 'Negative'
    elif label_id == 1:
        return 'Neutral'
    elif label_id == 2:
        return 'Positive'
    else:
        return 'Unknown'

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=5000)
