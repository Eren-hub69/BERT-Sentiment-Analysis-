from transformers import pipeline
import huggingface_hub as hb
classifier=pipeline("sentiment-analysis")


text=input("Enter your Review: ")

result=classifier(text)

print(f"Prediction: {result[0]['label']}")
print(f"Confidence: {result[0]['score'] * 100:.2f}%")

