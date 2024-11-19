from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pandas as pd
tokenizer = AutoTokenizer.from_pretrained("finetune_roberta")
model = AutoModelForSequenceClassification.from_pretrained("finetune_roberta", num_labels=6)
data = pd.read_csv("Database\digit\digitalization_tools_detail.csv")
lists_of_data = os.listdir("Database")  # ["ai", "vr", ....]
label2id = {'digit': 0, 'edtech': 1, 'proptech': 2, 'ai': 3, 'vr': 4, 'artech': 5}
id2label = {0: 'digit', 1: 'edtech', 2: 'proptech', 3: 'ai', 4: 'vr', 5: 'artech'}
# for i, type in enumerate(lists_of_data):
#     label2id[type] = i
#     id2label[i] = type
# sen = "What is Zoom?"
for sen in data["Question"]:
    inputs = tokenizer(sen, return_tensors="pt")
    output = model(**inputs)
    pred = torch.argmax(output.logits, dim=-1)


    print(id2label[pred.item()])