from fastapi import FastAPI
from app.sentiment import router as sentiment_router

app = FastAPI(title="Sentiment API")
app.include_router(sentiment_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment API. POST /sentiment/predict"}
