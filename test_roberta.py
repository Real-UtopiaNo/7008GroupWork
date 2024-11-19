from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("finetune_roberta")
model = AutoModelForSequenceClassification.from_pretrained("finetune_roberta", num_labels=6)
sen = "What is machine learning?"
inputs = tokenizer(sen, return_tensors="pt")
output = model(**inputs)
pred = torch.argmax(output.logits, dim=-1)
lists_of_data = os.listdir("Database")  # ["ai", "vr", ....]
label2id = {}
id2label = {}
for i, type in enumerate(lists_of_data):
    label2id[type] = i
    id2label[i] = type
print(id2label[pred.item()])