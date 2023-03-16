from pydantic import BaseSettings
from functools import lru_cache
from pymilvus import Collection


class Settings(BaseSettings):
    # App name is used in the OpenAPI docs
    app_name: str 
    root_path: str
    embedding_server_url: str
    milvus_embedding_search_limit: int
    milvus_collection_name: str
    milvus_alias: str
    milvus_host: str
    milvus_port: int

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings():
    return Settings()