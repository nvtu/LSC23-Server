from fastapi import APIRouter
from api.api_v1.endpoints import search as search_api


router = APIRouter()
api_v1_routers = [
    search_api.router
]
for r in api_v1_routers:
    router.include_router(r)

