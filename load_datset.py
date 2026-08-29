from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer

dataset = load_dataset("stanfordnlp/imdb")

print("Dataset loaded!")

print("Train:")
print(len(dataset["train"]))

print("\nTest:")
print(len(dataset["test"]))

#for i in range(1,10):
 # print(dataset["train"][i])


#for item in dataset["train"]:
 # if item["label"]==1:
  #  print("POSITIVE: ")
   # print(item["text"])
    #break

#for item in dataset["train"]:
 #   if item["label"] == 0:
  #      print("NEGATIVE:")
   #     print(item["text"])
    #    break

print(dataset["train"].features)
print()

counts=Counter((dataset["train"]["label"]))
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

print(counts)
print("Negative:", counts[0])
print("Positive:", counts[1])
#for i in range(100):

#review = dataset["train"][8]["text"]
 
"""tokenized=tokenizer(review)
 if len(tokenized["input_ids"])>512:
  print("Found one!")
  print("Index: ",i)
  print("Token count: ",len(tokenized["input_ids"])) 
  break

"""



def tokenize_function(example):
    tokenized=tokenizer(example["text"],
                padding="max_length",
                 truncation=True,
                 max_length=512
                 )

    return tokenized

tokenized_dataset=dataset.map(tokenize_function)

print(tokenized_dataset)

    










