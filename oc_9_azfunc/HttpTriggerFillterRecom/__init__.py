import logging
import azure.functions as func
import os
import json
import pandas as pd
import utils.blobdataread as blb

def get_ratings():
    # compute how many times the user clicked an article
    data = all_clicks_df.groupby(['user_id', 'click_article_id']).size().to_frame().reset_index()
    data.rename(columns = {0:'rate'}, inplace = True)

    # compute the total number of clicks per user
    user_activity = all_clicks_df.groupby('user_id').size().to_frame().reset_index()
    user_activity.rename(columns = {0:'user_clicks'}, inplace = True)

    # compute the rating
    ratings = pd.merge(data, user_activity,
             how='left', on='user_id')
    ratings['rating'] = ratings.rate / ratings.user_clicks
    return ratings[['user_id', 'click_article_id', 'rating']]

def get_rating_user(ratings, user_id):
    ratings_user = ratings[ratings.user_id!=user_id]
    ratings_user = ratings_user.drop_duplicates(subset=['click_article_id'])
    ratings_user.user_id = user_id
    ratings_user = ratings_user.reset_index(drop=True)
    return ratings_user

def pred(usr, art):
    return algo.predict(usr, art).est


def predict_fillterec(user_id): 
    # Prediction:
    ratings = get_ratings()
    ratings_user = get_rating_user(ratings, user_id)
    rtigs = pd.Series(map(pred, ratings_user.user_id, ratings_user.click_article_id))
    ratings_user = ratings_user.assign(rating = rtigs)
    recomandations = ratings_user.sort_values("rating", ascending=False).head(5)[['click_article_id', 'rating']]
    return recomandations.set_index('click_article_id')['rating'].to_dict()


def main(req: func.HttpRequest) -> func.HttpResponse:
    global algo, all_clicks_df

    connect_str = os.environ['AzureWebJobsStorage']
    container = 'oc9'

    blob_path = 'usr_clicks.gzip'
    all_clicks_df = blb.read_parquet_from_blob_to_pandas_df(connect_str, container, blob_path)
    
    blob_path='surprise_modelp4.pkl.gz'
    algo = blb.get_weights_blob(connect_str, container, blob_path)

    user_id = req.params.get("user_id")

    if not user_id:
        article_id = -1
        score = -1
        recomandations = {
            article_id: score,
        }
        result = json.dumps(recomandations) 
        return func.HttpResponse(
            body = result,
            status_code=200
        )
    else:
        recomandations = predict_fillterec(int(user_id))

    result = json.dumps(recomandations) 

    return func.HttpResponse(
        body = result,
        status_code=200
    )
    
