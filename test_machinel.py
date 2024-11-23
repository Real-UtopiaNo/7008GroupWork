import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from machinel_chatbot import *

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

# print(database.keys())

tfidf_matrix, vectorizers = tfidf_init(database)

# for question_type in database:
#     print(f"Question type: {question_type}")
#     print(f"Number of questions: {len(database[question_type])}")
#     print(f"TF-IDF matrix shape: {tfidf_matrix[question_type].shape}")
#     print()

def test_retrieval_accuracy(database, tfidf_matrix, vectorizers, similarity_threshold=0.9):
    total_accuracy = {}
    error_retrevial = {}
    
    for question_type in database:
        print(f"\nTesting accuracy for type: {question_type}")
        correct = 0
        error = 0
        total = len(database[question_type])
        
        for i, (input_question, _) in enumerate(database[question_type]):
            _, index, similarity = tfidf_retrieve_answer(
                input_question, 
                question_type,
                database[question_type],
                tfidf_matrix[question_type],
                vectorizers
            )

            if input_question.lower() == database[question_type][index][0].lower() or similarity >= similarity_threshold:
                correct += 1
            else:
                if question_type not in error_retrevial:
                    error_retrevial[question_type] = [[input_question, database[question_type][index][0], similarity]]
                else:
                    error_retrevial[question_type].append([input_question, database[question_type][index][0], similarity])
                error += 1

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{total} questions")

        accuracy = correct / total * 100
        total_accuracy[question_type] = accuracy
        print(f"Accuracy for {question_type}: {accuracy:.2f}%")
        if accuracy == 100:
            continue

        print("\nSample error cases:")
        for errors in error_retrevial[question_type]:
            print(f"Input question: {errors[0]}")
            print(f"Retrieved question: {errors[1]}")
            print(f"Similarity: {errors[2]}")
            print("-" * 50)

    print("\nOverall Statistics:")
    print("-" * 50)
    for qtype, acc in total_accuracy.items():
        print(f"{qtype}: {acc:.2f}%")
    print(f"Average accuracy: {sum(total_accuracy.values()) / len(total_accuracy):.2f}%")

    return total_accuracy

def plot_accuracy_results(accuracy_results):
    question_types = list(accuracy_results.keys())
    accuracies = list(accuracy_results.values())

    plt.figure(figsize=(12, 7))
    
    bars = plt.bar(range(len(question_types)), accuracies)

    plt.title('Retrieval Accuracy by Machinel method', fontsize=14)
    plt.xlabel('Question Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(range(len(question_types)), question_types, rotation=45, ha='right')
    plt.ylim(0, max(accuracies) * 1.1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analysis/machinel_acc.png')
    plt.show()

accuracy_results = test_retrieval_accuracy(database, tfidf_matrix, vectorizers, 0.98) #这里调整阈值
plot_accuracy_results(accuracy_results)