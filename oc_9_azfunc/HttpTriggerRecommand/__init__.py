import logging

import azure.functions as func
import json
from operator import itemgetter
import pandas as pd
from surprise import dump
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from azure.storage.blob import   BlobServiceClient, BlobClient
import pickle
import pyarrow as pa
import pyarrow.parquet as pq

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

#########################################################################################

def find_top_n_indices(data, top=5):
    indexed = enumerate(data)
    sorted_data = sorted(indexed,
                         key=itemgetter(1),
                         reverse=True)
    result = {}
    for article, score in sorted_data[:top]:
        result[article] = score
    return result

def recommendFromArticle(article_emb, embedding_articles, top=5):
    '''
    article_emb - the mean of the articles embedding the user clicked
    embedding_articles - the articles the user did not clicked
    '''
    score = cosine_similarity(embedding_articles, article_emb)
    _best_scores = find_top_n_indices(score, top)
    return _best_scores

def get_articles_user_clicked(user_id):
    articles_user_clicks = all_clicks_df[all_clicks_df.user_id == user_id].click_article_id
    usr_click = embg_data.query("article_id in @articles_user_clicks")
    usr_not_click = embg_data.query("article_id not in @articles_user_clicks")
    
    return usr_click.drop(columns=['article_id']).to_numpy(), usr_not_click.drop(columns=['article_id']).to_numpy()

def predict_collaborative(user_id):
    embedding_articles_user_clicked,  embedding_articles= get_articles_user_clicked(user_id)
    article_emb = np.mean(embedding_articles_user_clicked, axis=0)
    result = recommendFromArticle(article_emb.reshape(1, -1), embedding_articles, top=5)
    score = {}
    for article in result:
        score[article] = float(result[article][0])
    return score

##############################################################################################

def main(req: func.HttpRequest, allclicksdf:bytes, embdata:bytes, smodel:bytes) -> func.HttpResponse:
    logging.info(f'Python HTTP trigger function processed a request. clickblob: {len(allclicksdf)} bytes, embdata: {len(embdata)} bytes, surprise model: {len(smodel)} bytes')
    # 1. load the ratings dataset, algo - model used for collaborative filtering and the embg_data - used for content based
    global algo, all_clicks_df, embg_data

    #blob_to_read = BytesIO(smodel).read()
    #blob_to_read.seek(0)
    #logging.info(type(blob_to_read))
    #algo = pickle.loads(blob_to_read)
    #logging.info('algo loaded')
    #logging.info('all data loaded')


    reader = pa.BufferReader(allclicksdf)
    reader.seek(0)
    table = pq.read_table(reader)
    all_clicks_df = table.to_pandas()  # This results in a pandas.DataFrame
    reader.close()

    reader = pa.BufferReader(embdata)
    reader.seek(0)
    table = pq.read_table(reader)
    embg_data = table.to_pandas()  # This results in a pandas.DataFrame
    reader.close()

    #########################################################################################
    
    user_id = req.params.get("user_id") # it is a string but we need an int
    recomandation_type = req.params.get("recommand")

    logging.info('1. user_id: {0} and recommandation: {1} declared '.format(user_id, recomandation_type))
    
    if not user_id:
        article_id = -1
        score = -1
        recomandations = {
            article_id: score,
        }

    elif recomandation_type == 'filtering-recommandation':
        recomandations = predict_fillterec(int(user_id))
    elif recomandation_type == 'collaborative-recommandation':
        recomandations = predict_collaborative(int(user_id))

    result = json.dumps(recomandations) 

    return func.HttpResponse(
        body = result,
        status_code=200
    )
