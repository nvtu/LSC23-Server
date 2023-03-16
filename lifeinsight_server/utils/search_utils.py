from utils.connection_utils import send
import numpy as np
import json
import api.deps as deps


def encode_text_query(text_query: str):
    EMBEDDING_SERVER_URL = f'{deps.settings.embedding_server_url}/encode-text'
    method = 'POST'
    payload = json.dumps({ 'query_text': text_query })
    response = send(EMBEDDING_SERVER_URL, payload, method)
    embedded_query = response.json()['embedding']
    embedded_query = np.array(embedded_query)
    return embedded_query


def milvus_search(embedding):
    LIMIT = deps.settings.milvus_embedding_search_limit # Max number of results to return

    milvus_search_params = {
        'metric_type': 'IP',
        'params': {
            'nprobe': 128
        }
    }

    ranked_list = deps.collection.search(
        data = [embedding],
        anns_field = 'embedding',
        param = milvus_search_params,
        limit = LIMIT,
        expr = None,
        consistent_level = 'strong',
    )

    if len(ranked_list) == 0:
        ranked_list = []
    else:
        ranked_list = [item.id for item in ranked_list[0]]

    return ranked_list



def do_free_text_search(text_query: str):
    # Encode the text query into an embedding
    encoded_query = encode_text_query(text_query)
    # Search Milvus for the most similar embeddings
    ranked_list = milvus_search(encoded_query)

    return ranked_list