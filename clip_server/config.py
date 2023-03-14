from pydantic import BaseSettings


class Settings(BaseSettings):
    # App name is used in the OpenAPI docs
    app_name: str 
    # Name of the CLIP model to use
    model_name: str 
    # Device to use for inference (cpu or cuda)
    device: str 


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
