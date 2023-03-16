from typing import List
from pydantic import BaseModel, Field


# Request Schemas
class FreeTextSearchRequest(BaseModel):
    text_query: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "text_query": "I was praying to small golden Buddha in a tunnel. There were plants and offerings around the Buddha."
            }
        }


# Response Schemas
class FreeTextSearchResponse(BaseModel):
    query: str
    ranked_list: List[str]

    class Config:
        schema_extra = {
            "example": {
                "query": "I was praying to small golden Buddha in a tunnel. There were plants and offerings around the Buddha.",
                "ranked_list": [
                    "201910/11/20191011_052810_000",
                    "201910/11/20191011_052827_000",
                    "201910/11/20191011_052844_000",
                    "201910/11/20191011_052901_000"
                ]
            }
        }