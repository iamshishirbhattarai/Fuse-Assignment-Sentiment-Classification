# app/config.py
import os
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_dir: str = os.getenv("APP_MODEL_DIR", "model/rotten_roberta_sentiment")
    max_seq_length: int = int(os.getenv("APP_MAX_SEQ_LENGTH", "128"))
    model_config = SettingsConfigDict(env_prefix="APP_")

    @property
    def device(self) -> torch.device:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

settings = Settings()