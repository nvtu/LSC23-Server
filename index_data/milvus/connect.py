from functools import lru_cache
from pymilvus import connections
from dotenv import load_dotenv
import os

# Environment config files 
ENVIRONMENTS = {
    'development': '.env',
    'production': '.env.deploy',
    'docker': '.env.docker',
}

CURRENT_CONFIG = 'development'

# Load environment variables
@lru_cache
def load_env(config):
    load_dotenv(ENVIRONMENTS[config])

load_env(CURRENT_CONFIG)

# Milvus config
MILVUS_ALIAS = os.getenv('MILVUS_ALIAS')
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')

def connect_milvus():
    # Connect to Milvus
    try:
        connections.connect(
            alias = MILVUS_ALIAS,
            host = MILVUS_HOST,
            port = MILVUS_PORT,
        )
    except Exception as e:
        print('Milvus connection failed: ', e, '!!!')

    print('Milvus connected!!!')