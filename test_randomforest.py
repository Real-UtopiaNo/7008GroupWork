import os
import pandas as pd
from Randomforest_clf import *

def load_database(lists_of_data):
    database = {}  # 存储所有数据的QA对
    for type in lists_of_data:
        print("Loading data type: ",type)
        database_type = [] # QA for certain type
        path = "Database\\" + type
        csvfiles = os.listdir(path)
        for csvfile in csvfiles:
            data = pd.read_csv(path + "\\" + csvfile)
            for i in range(len(data)):
                database_type.append((data["Question"][i], data["Answer"][i]))
        database[type] = database_type
    return database

lists_of_data = os.listdir("Database")
database = load_database(lists_of_data)
clf = train_tf_classifier(database)

sum_ = {}
sum_correct = {}

for data_type in database:
    sum_[data_type] = 0
    sum_correct[data_type] = 0
    for pair in database[data_type]:
        sum_[data_type] += 1
        if Randomforest_classifier(clf,pair[0]) == data_type:
            sum_correct[data_type] += 1

print("Accuracy for each data type:")
for data_type in sum_:
    print(data_type, ":", sum_correct[data_type]/sum_[data_type])
print("Overall accuracy:", sum(sum_correct.values())/sum(sum_.values()))