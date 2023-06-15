from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import logging
import pandas as pd
import os

app = Flask(__name__)

# Update logging configuration
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', 
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

# Load from environment variables
tokenizer_path = os.getenv('') # change to your tokenizer path
model_path = os.getenv('') # change to your model path
unique_labels_path = os.getenv('') # change to your unique_labels.csv path

max_seq_length = 100

# Load the tokenizer, model and unique_labels at start
logging.info('Loading tokenizer...')
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

logging.info('Loading model...')
model = tf.keras.models.load_model(model_path)

# Load dataset and extract unique labels (positions)
logging.info('Loading dataset...')
df = pd.read_csv(unique_labels_path)
unique_labels = df['label'].unique().tolist()

def preprocess_input(skill_one, skill_two, id_disability):
    text = f'{id_disability} {skill_one} {skill_two}'
    sequences = tokenizer.texts_to_sequences([text])
    sequences = pad_sequences(sequences, maxlen=max_seq_length)
    return sequences

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info('Received prediction request...')
        data = request.get_json()

        # check if data is None
        if not data:
            logging.warning('No input data provided')
            return jsonify({'message': 'No input data provided'}), 400

        # Extract input features
        skill_one = data.get('skill_one')
        skill_two = data.get('skill_two')
        id_disability = data.get('id_disability')

        # check if fields are missing
        if any(arg is None for arg in [skill_one, skill_two, id_disability]):
            logging.warning('Data missing in JSON')
            return jsonify({'message': 'Data missing in JSON'}), 400

        logging.info('Preprocessing input...')
        sequences = preprocess_input(skill_one, skill_two, id_disability)

        logging.info('Making predictions...')
        predictions = model.predict(sequences)
        predicted_label_indices = np.argsort(predictions[0])[::-1][:10]
        ranked_labels = [unique_labels[index] for index in predicted_label_indices]

        logging.info('Preparing response...')
        response_data = {'predictions': ranked_labels}

        return jsonify(response_data)
    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'message': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)