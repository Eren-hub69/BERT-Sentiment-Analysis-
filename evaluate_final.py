from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np


# =========================
# LOAD MODEL + TOKENIZER
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    "./final_model"
)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


# =========================
# LOAD IMDB TEST DATA
# =========================

dataset = load_dataset("stanfordnlp/imdb")


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
# TRAINER
# =========================

training_args = TrainingArguments(
    output_dir="./evaluation",
    per_device_eval_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args
)


# =========================
# PREDICTIONS
# =========================

predictions = trainer.predict(test_dataset)

predicted_labels = predictions.predictions.argmax(axis=1)

actual_labels = np.array(test_dataset["label"])


# =========================
# METRICS
# =========================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision, recall, f1, _ = precision_recall_fscore_support(
    actual_labels,
    predicted_labels,
    average="binary"
)

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)


# =========================
# RESULTS
# =========================

print("\n========== FINAL RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n========== CONFUSION MATRIX ==========")
print(cm)