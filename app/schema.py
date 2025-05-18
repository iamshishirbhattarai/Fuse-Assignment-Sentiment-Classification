from pydantic import BaseModel
from typing import Dict

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    scores: Dict[str, float]
