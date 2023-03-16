from fastapi import APIRouter, status
from fastapi.encoders import jsonable_encoder
from schemas.api_v1 import FreeTextSearchRequest, FreeTextSearchResponse
from utils.search_utils import do_free_text_search


router = APIRouter(prefix = '/search', tags = ['search'])

@router.post('/free-text-search', status_code = status.HTTP_200_OK, response_model = FreeTextSearchResponse)
def free_text_search(request: FreeTextSearchRequest):
    request = jsonable_encoder(request)

    ranked_list = do_free_text_search(**request)
    response = {
        'query': request['text_query'],
        'ranked_list': ranked_list
    }
    return response

