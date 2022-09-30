###to run the Flask server
# FLASK_APP=Bartcus_Marius_API_092022.py flask run
###
import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import keras
import tensorflow as tf
from transformers import *


app = Flask(__name__)

model_name='bert_model.h5'
results_data_path = os.path.join("results")
model_file_path = os.path.join(results_data_path, model_name)


@app.before_first_request
def load__model():
    """
    Load model
    :return: model (global variable)
    """
    print('[INFO] Model Loading ........')
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

    global trained_model
    #model = load_model(MODEL_FOLDER + 'best_model.h5')
    trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
    trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
    trained_model.load_weights(model_file_path)
    print('[INFO] : Model loaded')


def compute_bert(text):
    input_ids=[]
    attention_masks=[]
    token_type_ids=[]
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_inp=bert_tokenizer.encode_plus(text,
                                        add_special_tokens = True,
                                        max_length = 64,
                                        pad_to_max_length = True,
                                        return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])
    token_type_ids.append(bert_inp['token_type_ids'])

    input_ids=np.asarray(input_ids)
    attention_masks=np.array(attention_masks)
    token_type_ids=np.array(token_type_ids)
    return input_ids, attention_masks, token_type_ids


def predict(my_tweet):
    val_inp, val_mask, _ = compute_bert(my_tweet)

    # Prediction:
    preds = trained_model.predict([val_inp, val_mask], batch_size=128)
    predictions = tf.math.softmax(preds.logits, axis=1).numpy()

    return predictions

# API
@app.route("/api/tweet/")
def sentiment_tweet():
    #dictionnaire = {
    #        'type': 'Prévision de température',
    #    'valeurs': [24, 24, 25, 26, 27, 28],
    #    'unite': "degrés Celcius"
    #}
    #return jsonify(dictionnaire)

    my_tweet = request.args.get("my_tweet")
    if not my_tweet:
        my_tweet = ''

    predictions = predict(my_tweet)

    pos = predictions[0,1]
    neg = predictions[0,0]


    dictionnaire = {
        'Positive': str(pos),
        'Negative': str(neg),
    }

    return jsonify(dictionnaire)
