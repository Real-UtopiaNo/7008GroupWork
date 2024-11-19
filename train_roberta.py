from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
# FacebookAI/roberta-base   
lists_of_data = os.listdir("Database")  # ["ai", "vr", ....]
database = []  # 存储所有数据的QA对
count = {}
for type in lists_of_data:
    print("Loading data type: ",type)
    count[type] = 0
    path = "Database/" + type
    csvfiles = os.listdir(path)
    for csvfile in csvfiles:
        data = pd.read_csv(path + "/" + csvfile)
        # data = data.dropna()
        for i in range(len(data)):
            database.append((data["Question"][i].lower(), type))
            count[type] += 1
print(count)

class MyDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.label2id = {}
        self.id2label = {}
        self.label2id = {'digit': 0, 'edtech': 1, 'proptech': 2, 'ai': 3, 'vr': 4, 'artech': 5}
        self.id2label = {0: 'digit', 1: 'edtech', 2: 'proptech', 3: 'ai', 4: 'vr', 5: 'artech'}

    def __getitem__(self, index):
        return self.data[index][0], self.label2id[self.data[index][1]]
    
    def __len__(self):
        return len(self.data)
    
dataset = MyDataset(database)
# for i in range(5):
#     print(dataset[i])

# trainset, validset = random_split(dataset, lengths=[0.999, 0.001])
# len(trainset), len(validset)
# for i in range(5):
#     print(trainset[i])
#     print(validset[i])

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

trainloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_func)
validloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_func)

model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=6)

if torch.cuda.is_available():
    model = model.cuda()
optimizer = Adam(model.parameters(), lr=1e-4)
# class_counts = torch.tensor([i for i in count.values()])  # 每个类别的样本数

# weights = 1.0 / class_counts.float()  # 权重与样本数成反比
# weights = weights / weights.sum()  # 归一化
# if torch.cuda.is_available():
#     weights = weights.to("cuda")
# print(weights)
# criterion = nn.CrossEntropyLoss(weight=weights)

def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(dataset)

def train(epoch=5, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            # labels = batch["labels"]
            # logits = output.logits
            # loss = criterion(logits, labels)
            output.loss.backward()
            # output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
    acc = evaluate()
    print(f"ep: {epoch}, acc: {acc}")

train()
model.save_pretrained("finetune_roberta_un")
tokenizer.save_pretrained("finetune_roberta_un")



