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
import os
#import pickle
import io

def read_parquet_from_blob_to_pandas_df(connection_str, container, blob_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    blob_client = blob_service_client.get_blob_client(container = container, blob = blob_path)
    stream_downloader = blob_client.download_blob()
    stream = BytesIO()
    stream_downloader.readinto(stream)
    df = pd.read_parquet(stream, engine = 'pyarrow')
    
    return df

def get_pkl_blob(connection_str, container, blob_path):
    logging.info('run get pkl')
    #blob_client = BlobClient.from_connection_string(connection_str, container, blob_path)
    #logging.info('blob client')
    #downloader = blob_client.download_blob(0)
    #logging.info('downloader done')
    ## Load to pickle
    #b = downloader.readall()
    #logging.info('read all done')
    #pkl = pickle.loads(b)
    ##pkl = dump.load(b)
    #logging.info('pkl done')
    #logging.info(pkl)


    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    blob_client = blob_service_client.get_blob_client(container = container, blob = blob_path)
    stream_downloader = blob_client.download_blob()
    stream = BytesIO()
    stream_downloader.readinto(stream)
    logging.info('downloader done')
    pkl = dump.load(stream)
    logging.info('pkl done')
    return pkl
##############################################################################################

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
    # 2. load the 2 data used for the recommandation systems
    #all_clicks_df = pd.read_parquet('results/usr_clicks.gzip')
    #embg_data = pd.read_parquet('results/embedding_proj.gzip')
    #connect_str = 'DefaultEndpointsProtocol=https;AccountName=oc9;AccountKey=bK1SRnkyvlMRK5o9rMkFZlSAfK9ziIRoX3Kf+9EHXZ9crAg2FffDiLkAc1JVJq3sYO/dbtiAGz1u+AStdN0CLg==;EndpointSuffix=core.windows.net'

    #container = 'result'
    #embg_data_blob_path = 'embedding_proj.gzip'
    #all_clicks_blob_path = 'usr_clicks.gzip'
    
    #all_clicks_df = read_parquet_from_blob_to_pandas_df(connect_str, container, all_clicks_blob_path)
    #embg_data = read_parquet_from_blob_to_pandas_df(connect_str, container, embg_data_blob_path)
    
    #logging.info('done parquet')
    # 3. load the model for colalaborative fitlering
    #algo_data_blob_path = 'surprise_model.pkl.gz'
    #algo = get_pkl_blob(connect_str, container, algo_data_blob_path)[1]
    #logging.info('done pkl')
    #logging.info(algo)

    all_clicks_df = pd.read_parquet(allclicksdf, engine = 'pyarrow')
    logging.info('all_clicks_df loaded')
    embg_data = pd.read_parquet(embdata, engine = 'pyarrow')
    logging.info('embg_data loaded')
    _, algo = dump.load(smodel)
    logging.info('all data loaded')
    
    #_, algo = dump.load('results/surprise_model.pkl.gz')
    
    # blob_client = BlobClient.from_connection_string(connect_str, container, 'surprise_model.pkl.gz')
    # downloader = blob_client.download_blob(0)
    # b = downloader.readall()
    #_ , algo = pickle.loads(b)
    # logging.info('Data pickle loaded.')
    #########################################################################################
    
    #name = req.params.get('name')
    user_id = req.params.get("user_id") # it is a string but we need an int
    recomandation_type = req.params.get("recommand")

    logging.info('1. user_id: {0} and recommandation: {1} declared '.format(user_id, recomandation_type))
    
    # if not name:
    '''
    try:
        logging.info('try')
        req_body = req.get_json()
        logging.info('end try')
    except ValueError:
        logging.info('ValueError')
        pass
    else:
        logging.info('else')
        # name = req_body.get('name')
        user_id = req_body.get("user_id") # it is a string but we need an int
        recomandation_type = req_body.get("recommand")
        logging.info('2. user_id: {0} and recommandation: {1} declared '.format(user_id, recomandation_type))
    '''

    if not user_id:
        user_id = -1
        score = -1
        recomandations = {
            user_id: score,
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
