from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
tokenizer = AutoTokenizer.from_pretrained("finetune_roberta_un")
model = AutoModelForSequenceClassification.from_pretrained("finetune_roberta_un")

lists_of_data = os.listdir("Database")  # ["ai", "vr", ....]
database = []  # 存储所有数据的QA对
count = {}
correct = {}
for type in lists_of_data:
    print("Loading data type: ",type)
    count[type] = 0
    correct[type] = 0
    path = "Database/" + type
    csvfiles = os.listdir(path)
    for csvfile in csvfiles:
        data = pd.read_csv(path + "/" + csvfile)
        # data = data.dropna()
        for i in range(len(data)):
            database.append((data["Question"][i].lower(), type))
            count[type] += 1
print(count)
if torch.cuda.is_available():
    model = model.cuda()
    print("gpu available")
label2id = {'digit': 0, 'edtech': 1, 'proptech': 2, 'ai': 3, 'vr': 4, 'artech': 5}
id2label = {0: 'digit', 1: 'edtech', 2: 'proptech', 3: 'ai', 4: 'vr', 5: 'artech'}
# sen = "What is Zoom?"
for i,(sen, type) in enumerate(database):
    inputs = tokenizer(sen, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    output = model(**inputs)
    pred = torch.argmax(output.logits, dim=-1)
    if id2label[pred.item()] == type:
        correct[type] += 1
    if(i%100 == 0):
        print(i/len(database) , "%")
for type in count.keys():
    print(type , ":" ,correct[type] / count[type])

plt.figure(figsize=(10, 6))
categories, accuracies = [], []
overall_accuracy = sum(correct.values())/sum(count.values())
print("Overall accuracy: ", overall_accuracy)
for data_type in count:
    categories.append(data_type)
    accuracies.append(correct[data_type]/count[data_type])
plt.bar(categories, accuracies, color='yellowgreen')
plt.axhline(y=overall_accuracy, color='r', linestyle='--', label='Overall accuracy') 
plt.ylim(0, 1) 
plt.title('Bayesian Classifier Accuracies')
plt.xlabel('Categories')
plt.ylabel('Accuracy')
plt.legend()
os.makedirs('analysis', exist_ok=True)
plt.savefig('analysis/roberta_acc.png')