from pymilvus import Collection
from core.config import get_settings
from utils.connection_utils import connect_milvus


# Server settings
settings = get_settings()


# Milvus server connection settings
connect_milvus()
collection = Collection(name = settings.milvus_collection_name, using = settings.milvus_alias)
collection.load()
