from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "Erenkass/bert-imdb-sentiment"
)

# Load correct BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Erenkass/bert-imdb-sentiment"
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()


def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities=torch.softmax(outputs.logits,dim=1)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    confidence=probabilities[0][prediction].item()



    if prediction == 1:
        sentiment="Positive"
    else:
        sentiment= "Negative"

    return sentiment , confidence

    


# Test sentences

sentences = [
    "This movie was absolutely fantastic!",
    "I really enjoyed this movie.",
    "This was one of the worst movies I have ever seen.",
    "The movie was boring and disappointing."
]

for sentence in sentences:

    result = predict_sentiment(sentence)

    print("\nText:", sentence)
    print("Sentiment:", result)