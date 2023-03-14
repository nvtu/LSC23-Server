from functools import lru_cache
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from clip_model import CLIP
import config
from schemas import EncodeTextRequestSchema, EncodeTextResponseSchema


@lru_cache
def get_settings() -> config.Settings:
    return config.Settings()


settings = get_settings()
model = CLIP(settings.model_name, settings.device)
app = FastAPI(name = settings.app_name)


@app.post("/encode-text", response_model=EncodeTextResponseSchema)
async def encode_text(request: EncodeTextRequestSchema):
    request = jsonable_encoder(request)
    text_query = request['query_text']
    print('/encode-text', text_query)

    # Encode text query
    features = model.encode_text(text_query).squeeze(0).tolist()
    response = {
        'embedding': features
    }
    return response



