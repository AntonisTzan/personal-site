from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

import os
import requests
import zipfile


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# URL of your model file on Dropbox (make sure it ends with `?dl=1` for direct download)
dropbox_url = 'https://www.dropbox.com/scl/fo/bnk1eg0ti7lukotkjb89p/APenlNYEKPsfBBUSBZVajoE?rlkey=qw9b1kimudobwn56ssoyhayji&st=7j35rc9r&dl=1'
local_model_dir = 'my_finetuned_bert_model_corr2'

# Check if the model already exists locally, if not download it
if not os.path.exists(local_model_dir):
    print('Downloading model from Dropbox...')
    r = requests.get(dropbox_url)
    with open('model.zip', 'wb') as f:
        f.write(r.content)

    # Unzip the model
    with zipfile.ZipFile('model.zip', 'r') as zip_ref:
        zip_ref.extractall(local_model_dir)

    print('Model downloaded and extracted.')

# Load the model
model = tf.keras.models.load_model(local_model_dir)


# Initialize the tokenizer and model here
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#model = tf.keras.models.load_model('my_finetuned_bert_model_corr2')


@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()  # Get the input data
    input_text = data['input_text']

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='tf', max_length=128, padding='max_length', truncation=True)

    # Run inference
    predictions = model.predict({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'token_type_ids':inputs['token_type_ids']})
    
    # Extract logits
    logits = predictions['logits']  # Directly use predictions as they should be logits

    # Calculate probabilities
    probabilities = tf.nn.softmax(logits)
    
    
    

    # Determine the result based on probabilities
    if probabilities[0, 0] > probabilities[0, 1] + 0.05:  # Example threshold for class 0
        result = "webpage_1"
    elif probabilities[0, 1] > probabilities[0, 0] + 0.05:  # Example threshold for class 1
        result = "webpage_2"
    else:
        result = "webpage_3"
        
        
    return jsonify({"result": result})

    # Mock response for demonstration purposes
    #response = {"result": "webpage_1"}  # Change this based on your model's output

    #return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)