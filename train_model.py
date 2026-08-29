from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_dataset
import torch


# ==========================================
# 1. CHECK GPU
# ==========================================

print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. LOAD ORIGINAL BERT MODEL
# ==========================================

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


# ==========================================
# 3. LOAD MATCHING TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)


# ==========================================
# 4. LOAD IMDB
# ==========================================

dataset = load_dataset("stanfordnlp/imdb")


# ==========================================
# 5. TOKENIZATION
# ==========================================

def tokenize_function(example):

    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )


tokenized_dataset = dataset.map(tokenize_function)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])


train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]


print("Training examples:", len(train_dataset))
print("Testing examples:", len(test_dataset))


# ==========================================
# 6. TRAINING SETTINGS
# ==========================================

training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=1,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    fp16=True,

    logging_steps=100,

    eval_strategy="epoch",

    save_strategy="epoch"
)


# ==========================================
# 7. TRAINER
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset
)


# ==========================================
# 8. TRAIN
# ==========================================

trainer.train()


# ==========================================
# 9. SAVE MODEL + TOKENIZER
# ==========================================

trainer.save_model("./final_model")

tokenizer.save_pretrained("./final_model")


print("\nTraining complete!")
print("Model saved to ./final_model")