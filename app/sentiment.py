import torch
from fastapi import APIRouter, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import settings
from app.schema import SentimentRequest, SentimentResponse

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(settings.model_dir)
model.to(settings.device)
model.eval()

@router.post("/predict", response_model=SentimentResponse)
def predict_sentiment(req: SentimentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must not be empty")
    inputs = tokenizer(
        req.text,
        truncation=True,
        padding="max_length",
        max_length=settings.max_seq_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(settings.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    label = "positive" if probs[1] > probs[0] else "negative"
    return SentimentResponse(label=label, scores={"negative": probs[0], "positive": probs[1]})