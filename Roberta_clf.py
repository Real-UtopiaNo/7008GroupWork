from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("finetune_roberta")
model = AutoModelForSequenceClassification.from_pretrained("finetune_roberta", num_labels=6)
label2id = {'digit': 0, 'edtech': 1, 'proptech': 2, 'ai': 3, 'vr': 4, 'artech': 5}
id2label = {0: 'digit', 1: 'edtech', 2: 'proptech', 3: 'ai', 4: 'vr', 5: 'artech'}

def Roberta_classifier(question):
    inputs = tokenizer(question, return_tensors="pt")
    output = model(**inputs)
    pred = torch.argmax(output.logits, dim=-1)


    return id2label[pred.item()]