from fastapi import FastAPI, HTTPException
from predict import predict_sentiment
from pydantic import BaseModel ,Field



app=FastAPI()



class Review(BaseModel):
    text: str = Field(min_length=1)


@app.get("/")
def home():
    return {"message":"BERT API is running!"}

@app.post("/predict")
def predict(review:Review):

    if not review.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty"
        )
    sentiment,confidence=predict_sentiment(review.text)

    return{
        "sentiment":sentiment,
        "confidence":confidence
    }