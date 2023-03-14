from typing import List
from pydantic import BaseModel, Field


class EncodeTextRequestSchema(BaseModel):
    query_text: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "query_text": "A boy is in the kitchen, sitting on a chair. He gets up to look out the window."
            }
        }

    
class EncodeTextResponseSchema(BaseModel):
    embedding: List[float]

    class Config:
        schema_extra = {
            "example": {
                "embedding": [0] * 768 
            }
        }