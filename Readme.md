# BERT Sentiment Analysis

A sentiment analysis project using **BERT (Bidirectional Encoder Representations from Transformers)** to classify movie reviews as **Positive** or **Negative**.

The model is fine-tuned on the **IMDb Movie Review Dataset** using Hugging Face Transformers and PyTorch.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine the emotional tone of a piece of text.

In this project, the goal is:

> Given a movie review, predict whether the review expresses a **positive** or **negative** sentiment.

For example:

```text
"This movie was absolutely fantastic!"
→ Positive

"This was one of the worst movies I have ever seen."
→ Negative

The project uses a pretrained BERT-base-uncased model and fine-tunes it for binary sentiment classification.

🎯 Objectives

The main objectives of this project are:

Understand how BERT works for text classification.
Load and preprocess the IMDb dataset.
Tokenize text using a BERT tokenizer.
Fine-tune BERT for sentiment classification.
Evaluate the trained model.
Calculate Accuracy, Precision, Recall and F1 Score.
Generate a Confusion Matrix.
Test the trained model on custom sentences.
Save the trained model for future use.
🧠 Technologies Used
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
BERT
scikit-learn
CUDA
NVIDIA GPU
📂 Dataset

The project uses the:

IMDb Movie Review Dataset

Dataset:

stanfordnlp/imdb

The dataset contains:

25,000 training examples
25,000 testing examples
50,000 examples in total

Each example contains:

text  → Movie review
label → Sentiment

Labels:

0 → Negative
1 → Positive

Example:

Review:
"I really enjoyed this movie."

Label:
1
🔄 Project Pipeline

The complete project follows this pipeline:

IMDb Dataset
      ↓
Load Dataset
      ↓
BERT Tokenizer
      ↓
Tokenization
      ↓
Input IDs + Attention Mask
      ↓
Pretrained BERT
      ↓
Fine-Tuning
      ↓
BERT Classification Head
      ↓
Predictions
      ↓
Evaluation
      ↓
Accuracy / Precision / Recall / F1
1. Loading the Dataset

The IMDb dataset is loaded using Hugging Face Datasets:

from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")

The dataset contains:

train → 25,000 examples
test  → 25,000 examples
2. BERT Tokenizer

The project uses the tokenizer associated with:

bert-base-uncased

The tokenizer converts human-readable text into numerical tokens that BERT can understand.

Example:

"I love this movie!"

is converted into token IDs.

BERT also uses special tokens:

[CLS] → Beginning of the sequence
[SEP] → End of the sequence
[PAD] → Padding
[UNK] → Unknown token
[MASK] → Masked token
3. Tokenization

The reviews are tokenized using:

def tokenize_function(example):

    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
padding="max_length"

Every input is made the same length.

Short review → PAD PAD PAD ...
Long review  → truncated if necessary
truncation=True

BERT has a maximum sequence length of 512 tokens in this setup.

Therefore, reviews longer than 512 tokens are truncated.

max_length=512

The maximum input length is:

512 tokens
4. Important Problem Encountered

During the project, an important tokenizer problem was discovered.

Initially, the checkpoint contained a tokenizer with:

vocab_size = 5

The tokenizer was producing outputs such as:

[CLS] [UNK] [UNK] [UNK] [UNK] ...

For example:

"I love this movie!"

was effectively being represented as:

[CLS] [UNK] [UNK] [UNK] [UNK] [UNK] [SEP]

This was a major problem.

❌ Why This Was Bad

BERT needs meaningful token IDs to understand language.

If almost every word becomes:

[UNK]

then the model receives almost no useful linguistic information.

As a result, the model initially produced:

Accuracy ≈ 0.50

This was essentially random/baseline performance for a balanced binary classification problem.

5. Fixing the Tokenizer Problem

The tokenizer was changed to the original BERT tokenizer:

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

Instead of loading the broken tokenizer from:

./results/checkpoint-3125

The correct BERT tokenizer was used.

This allowed the model to correctly convert English text into meaningful BERT tokens.

6. Fine-Tuning BERT

The pretrained BERT model was loaded using:

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

The model was then fine-tuned on the IMDb dataset.

The two output classes are:

0 → Negative
1 → Positive
7. Training Configuration

The model was trained using Hugging Face's Trainer.

Important configuration:

TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch"
)
Important parameters
Parameter	Value	Meaning
Epochs	1	Dataset processed once
Training batch size	8	8 examples per batch
Evaluation batch size	8	8 examples per evaluation batch
Max sequence length	512	Maximum tokens per review
FP16	True	Mixed-precision training
Evaluation	Every epoch	Evaluate after training epoch
Saving	Every epoch	Save model checkpoint
8. GPU Acceleration

The project was trained using an NVIDIA GPU.

GPU detected:

NVIDIA GeForce RTX 4050 Laptop GPU

GPU availability was checked using:

import torch

print(torch.cuda.is_available())

print(
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "No GPU"
)

Output:

GPU available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU

Using the GPU significantly speeds up BERT training and inference.

9. Training Results

After correcting the tokenizer and retraining the model, the model achieved:

Training Loss: 0.3212
Evaluation Loss: 0.2257

The model was then evaluated on the complete IMDb test set.

📊 Final Evaluation

The final model achieved:

Accuracy:  0.9341
Precision: 0.9319
Recall:    0.9366
F1 Score:  0.9342

Therefore:

The final BERT sentiment classifier achieved approximately 93.41% accuracy on the IMDb test dataset.

10. Confusion Matrix

The final confusion matrix was:

[[11644   856]
 [  792 11708]]

The matrix can be interpreted as:

                 Predicted
               Negative Positive

Actual Negative   11644     856

Actual Positive     792   11708
Confusion Matrix Explanation
True Negative (TN)
11644

11,644 negative reviews were correctly classified as negative.

False Positive (FP)
856

856 negative reviews were incorrectly classified as positive.

False Negative (FN)
792

792 positive reviews were incorrectly classified as negative.

True Positive (TP)
11708

11,708 positive reviews were correctly classified as positive.

11. Prediction Distribution

The model predicted:

Predicted 0: 12436
Predicted 1: 12564

Actual distribution:

Actual 0: 12500
Actual 1: 12500

The predicted distribution is reasonably close to the actual distribution.

This indicates that the model is not simply predicting one class for every review.

12. Custom Predictions

After training, the model was tested on custom sentences.

Example inputs:

"This movie was absolutely fantastic!"

Prediction:

Positive
"I really enjoyed this movie."

Prediction:

Positive
"This was one of the worst movies I have ever seen."

Prediction:

Negative
"The movie was boring and disappointing."

Prediction:

Negative

The model correctly classified these example sentences.

13. Model Output

BERT produces logits for the two classes.

Conceptually:

                 BERT
                  ↓
             Classification
                Head
                  ↓
          ┌───────┴───────┐
          ↓               ↓
       Negative         Positive
        Logit             Logit
          ↓               ↓
       Class 0           Class 1

The predicted class is obtained using:

predictions.predictions.argmax(axis=1)

argmax() selects the class with the highest score.

Example:

Logits:

Negative = 1.2
Positive = 4.7

Highest = 4.7

Prediction = Positive
14. Important Concepts Learned
Tokenization

Converts text into tokens/token IDs that a neural network can process.

Text
 ↓
Tokenizer
 ↓
Token IDs
Attention Mask

The attention mask tells BERT which positions contain real tokens and which positions are padding.

Example:

Input IDs:

[CLS] I love movies [SEP] [PAD] [PAD]

Attention:

  1    1   1    1     1     0     0
1 → real token
0 → padding
Fine-Tuning

Instead of training BERT from scratch, a pretrained BERT model is taken and further trained on the IMDb sentiment task.

Pretrained BERT
      ↓
IMDb Reviews
      ↓
Fine-Tuning
      ↓
Sentiment Classifier
Classification Head

The BERT encoder produces contextual representations.

A classification layer uses this information to predict:

Negative
or
Positive
15. Why BERT?

Traditional approaches such as:

Bag of Words
TF-IDF

mostly represent words based on their occurrence.

BERT is more powerful because it considers the context of words.

For example:

"The movie was not good."

The word:

good

is positive by itself.

However, BERT can use the surrounding context:

"not good"

to understand that the overall sentiment is negative.

16. Problems Encountered During the Project
Problem 1 — Accuracy stuck around 50%

Initial evaluation produced approximately:

Accuracy: 0.50004

The model predicted almost everything as the same class.

Problem 2 — Broken tokenizer

Tokenizer inspection showed:

vocab_size = 5

and text was converted mostly into:

[UNK]

This meant that the model was not receiving meaningful token information.

Problem 3 — Incorrect checkpoint/tokenizer relationship

The checkpoint contained model information, but the tokenizer associated with it was not suitable for processing the IMDb English text.

The correct BERT tokenizer was therefore loaded from:

bert-base-uncased
Problem 4 — Retraining

After fixing the tokenizer/model setup, the model had to be trained again.

The corrected training produced:

Accuracy: 93.41%

This was the major turning point of the project.

17. Project Structure

A possible project structure is:

BERT_SENTIMENT/
│
├── train.py
├── eval.py
├── predict.py
├── README.md
├── requirements.txt
│
├── results/
│   └── checkpoints/
│
└── final_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...

Large model/checkpoint files should generally not be committed directly to GitHub. Use .gitignore or Git LFS where appropriate.

18. Installation

Clone the repository:

git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project:

cd BERT_SENTIMENT

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
19. Example requirements.txt
torch
transformers
datasets
scikit-learn
accelerate
20. Running the Project
Train the model
python train.py
Evaluate the model
python eval.py
Test custom reviews
python predict.py
21. Hugging Face Authentication Warning

During execution, the following warning may appear:

Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.

This does not necessarily mean that the model is broken.

It means the Hugging Face Hub request is unauthenticated.

For this project, the model and dataset were still successfully downloaded and used.

For higher rate limits, a Hugging Face access token can be configured.

22. Limitations

Although the model achieves high accuracy, it still has limitations.

Dataset limitation

The model was trained specifically on movie reviews.

Therefore, performance may decrease on other types of text such as:

Twitter posts
Product reviews
News articles
Technical documents
Chat messages
Sarcasm

Sentiment analysis can struggle with sarcasm.

Example:

"Wow, what an amazing movie... I almost fell asleep."

The literal words may appear positive while the actual sentiment is negative.

Mixed sentiment

Reviews can contain both positive and negative opinions.

Example:

"The acting was excellent, but the story was terrible."

A simple binary classifier must still choose one class.

23. Future Improvements

Possible improvements include:

Train for multiple epochs.
Tune the learning rate.
Experiment with batch sizes.
Use validation data separately from the test set.
Perform hyperparameter tuning.
Add a web interface.
Deploy the model using FastAPI or Flask.
Create a Streamlit application.
Containerize the application using Docker.
Upload the model to Hugging Face Hub.
Build an API for sentiment prediction.
24. What This Project Demonstrates

This project demonstrates the complete workflow of a modern NLP classification system:

Dataset
   ↓
Preprocessing
   ↓
Tokenization
   ↓
Pretrained Transformer
   ↓
Fine-Tuning
   ↓
Evaluation
   ↓
Custom Predictions
   ↓
Model Saving

It also demonstrates an important real-world ML lesson:

A model achieving poor performance does not always mean that the model architecture is bad. The data pipeline, tokenizer, preprocessing, model, and checkpoint must all be compatible.

In this project, the initial ~50% accuracy was caused by a broken tokenizer setup rather than BERT itself.

🏆 Final Result

The final BERT sentiment classifier achieved:

Metric	Score
Accuracy	93.41%
Precision	93.19%
Recall	93.66%
F1 Score	93.42%

Final confusion matrix:

[[11644   856]
 [  792 11708]]

The model successfully classifies IMDb movie reviews into:

0 → Negative
1 → Positive
📚 Key Takeaways

Through this project, I learned:

How to load datasets using Hugging Face Datasets.
How BERT tokenization works.
The purpose of [CLS], [SEP], [PAD], and [UNK].
How attention masks work.
How pretrained BERT can be fine-tuned.
How Hugging Face Trainer works.
How GPU acceleration can be used for training.
How to evaluate a classification model.
How to interpret a confusion matrix.
How to calculate Accuracy, Precision, Recall and F1 Score.
How to perform predictions on custom text.
How to debug a model that produces ~50% accuracy.
Why tokenizer and model compatibility is critical in NLP.
👨‍💻 Author

Eren

NLP / Machine Learning Project

⭐ Project Status

Completed ✅

The BERT sentiment analysis model has been trained, evaluated, tested on custom inputs, and documented.

Next learning phase:

BERT Project
     ↓
Transformers + Hugging Face
     ↓
Deployment / ML Engineering

### One important GitHub tip

I **would add this `README.md` to Git**. It is exactly what someone visiting your repository should see first.

But **don't add** your whole `results/` directory or `final_model/` blindly. Those model files can be hundreds of MB and can make GitHub messy. We should set up a proper `.gitignore` next and decide whether to use **Git LFS / Hugging Face Hub** for the model.

And yes — **this project is now genuinely done**. The next phase should be learning **T