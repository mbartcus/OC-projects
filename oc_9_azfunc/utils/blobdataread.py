from azure.storage.blob import BlobClient, BlobServiceClient
import pickle
from io import BytesIO
import pandas as pd

def get_weights_blob(connection_str, container, blob_path):
    blob_client = BlobClient.from_connection_string(
        connection_str, container, blob_path
    )
    downloader = blob_client.download_blob(0)

    # Load to pickle
    b = downloader.readall()
    weights = pickle.loads(b)

    return weights

def read_parquet_from_blob_to_pandas_df(connection_str, container, blob_path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    blob_client = blob_service_client.get_blob_client(container = container, blob = blob_path)
    stream_downloader = blob_client.download_blob()
    stream = BytesIO()
    stream_downloader.readinto(stream)
    df = pd.read_parquet(stream, engine = 'pyarrow')
    
    return df