from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
import numpy as np

# =========================
# LOAD FINAL MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    "./final_model"
)

tokenizer = AutoTokenizer.from_pretrained(
    "./final_model"
)

# =========================
# LOAD DATASET
# =========================

dataset = load_dataset("stanfordnlp/imdb")

# =========================
# TOKENIZATION
# =========================

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])

test_dataset = tokenized_dataset["test"]

# =========================
# EVALUATION
# =========================

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=8
    )
)

predictions = trainer.predict(test_dataset)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)

actual_labels = np.array(test_dataset["label"])

accuracy = np.mean(
    predicted_labels == actual_labels
)

# =========================
# RESULTS
# =========================

print("\n========== RESULTS ==========")

print(f"Accuracy: {accuracy:.4f}")

print("\nFirst 20 predicted labels:")
print(predicted_labels[:20])

print("\nFirst 20 actual labels:")
print(actual_labels[:20])

print("\nPrediction distribution:")
print("Predicted 0:", np.sum(predicted_labels == 0))
print("Predicted 1:", np.sum(predicted_labels == 1))

print("\nActual distribution:")
print("Actual 0:", np.sum(actual_labels == 0))
print("Actual 1:", np.sum(actual_labels == 1))