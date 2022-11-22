import logging

import azure.functions as func
import json
from operator import itemgetter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import utils.blobdataread as blb
#import pyarrow as pa
#import pyarrow.parquet as pq
import os

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

def main(req: func.HttpRequest) -> func.HttpResponse:
    # 1. load the ratings dataset, algo - model used for collaborative filtering and the embg_data - used for content based
    global all_clicks_df, embg_data

    connect_str = os.environ['AzureWebJobsStorage']
    container = 'oc9'

    blob_path = 'usr_clicks.gzip'
    all_clicks_df = blb.read_parquet_from_blob_to_pandas_df(connect_str, container, blob_path)
    
    blob_path = 'embedding_proj.gzip'
    embg_data = blb.read_parquet_from_blob_to_pandas_df(connect_str, container, blob_path)
    
    '''
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
    '''
    #########################################################################################
    
    user_id = req.params.get("user_id") # it is a string but we need an int

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
        recomandations = predict_collaborative(int(user_id))

    result = json.dumps(recomandations) 

    return func.HttpResponse(
        body = result,
        status_code=200
    )
