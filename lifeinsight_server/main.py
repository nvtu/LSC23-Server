from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import get_settings
from api.api_v1 import api as api_v1_router


config = get_settings()
app = FastAPI(name = config.app_name,
    docs_url = f'{config.root_path}/docs',
    redoc_url = f'{config.root_path}/redoc',
    openapi_url = f'{config.root_path}/openapi.json',
)

allowed_origins = ['*']
allowed_methods = ['GET', 'POST', 'OPTIONS']
allowed_headers = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins = allowed_origins,
    allow_credentials = True,
    allow_methods = allowed_methods,
    allow_headers = allowed_headers,
)

app.include_router(api_v1_router.router, prefix=f'{config.root_path}/api/v1')

@app.get(f"{config.root_path}")
def hello():
    return {"message": "Hello World"}



