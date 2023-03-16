from pymilvus import connections
from core import config
from functools import lru_cache
import requests

settings = config.get_settings()

def send(url: str, payload, method: str = 'POST'):
    response = requests.request(
        method, 
        url, 
        headers={'Content-Type': 'application/json'},
        data=payload
    )
    return response


@lru_cache
def connect_milvus():
    # Connect to Milvus
    try:
        connections.connect(
            alias = settings.milvus_alias,
            host = settings.milvus_host,
            port = settings.milvus_port,
        )
    except Exception as e:
        print('Milvus connection failed: ', e, '!!!')

    print('Milvus connected!!!')