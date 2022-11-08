###to run the Flask server
# FLASK_APP=Bartcus_Marius_API_092022.py flask run
# https://sentimentanalyseapi.herokuapp.com/api?my_tweet=I+hate+you
# http://127.0.0.1:5000/api?my_tweet=I+hate+you
###
import os
from flask import Flask, request, render_template, jsonify
import pandas as pd


app = Flask(__name__)


def get_rating_user(ratings, user_id):
    ratings_user = ratings[ratings.user_id!=user_id]
    ratings_user = ratings_user.drop_duplicates(subset=['article_id'])
    ratings_user.user_id = user_id
    ratings_user = ratings_user.reset_index(drop=True)
    return ratings_user

def pred(usr, art):
    return algo.predict(usr, art).est


@app.before_first_request
def load__model():
    """
    Load model
    :return: model and data (global variable)
    """

    # 1. load the ratings dataset, algo - model used for collaborative filtering and the embg_data - used for content based
    global algo, ratings #, embg_data

    # 2. load the 2 data used for the recommandation systems
    ratings = pd.read_parquet('results/ratings.gzip')

    # 3. load the model for colalaborative fitlering
    _, algo = dump.load('results/surprise_model.pkl')

def predict(user_id):
    # Prediction:
    ratings_user = get_rating_user(ratings, user_id)
    rtigs = pd.Series(map(pred, ratings_user.user_id, ratings_user.article_id))
    ratings_user = ratings_user.assign(rating = rtigs)
    recomandations = ratings_user.sort_values("rating", ascending=False).head(5)[['article_id', 'rating']]
    return recomandations.set_index('article_id')['rating'].to_dict()

# API
@app.route("/api")
def recommand():
    user_id = [request.args.get("user_id")]

    if not user_id:
        user_id = -1
        score = -1
        recomandations = {
            user_id: score,
        }
    else:
        recomandations = predict(user_id)


    return jsonify(recomandations)


# API TEST
@app.route("/test")
def test_api():
    dictionnaire = {
        'type': 'Prévision de température',
        'valeurs': [24, 24, 25, 26, 27, 28],
        'unite': "degrés Celcius"
    }
    return jsonify(dictionnaire)

if __name__ == "__main__":
    app.run(debug==True)
