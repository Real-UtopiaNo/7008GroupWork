import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from deepl_chatbot import *

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

tokenizers = RobertaTokenizer.from_pretrained('roberta-base')
preprocessed_database = preprocess_database(database, tokenizers)
model = QuestionEncoder()

def test_retrieval_accuracy(encoded_database, tokenizer, model, similarity_threshold=0.9):
    total_accuracy = {}
    error_retrevial = {}
    for question_type in database:
        print(f"\nTesting accuracy for type: {question_type}")
        correct = 0
        error = 0
        total = len(database[question_type])
        
        for i, (input_question, _) in enumerate(database[question_type]):
            index, similarity = dl_retrieve_question(
                input_question, 
                question_type,
                encoded_database,
                tokenizer,
                model
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
        for errors in error_retrevial[question_type][:5]:
            print(f"Input question: {errors[0]}")
            print(f"Retrieved question: {errors[1]}")
            print(f"Similarity: {errors[2]:.10f}")
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
    overall_accuracy = sum(accuracies) / len(accuracies)

    plt.figure(figsize=(12, 7))
    
    bars = plt.bar(range(len(question_types)), accuracies, color='green')

    plt.title('Retrieval Accuracy by Deep Learning method', fontsize=14)
    plt.xlabel('Question Type', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(range(len(question_types)), question_types)
    plt.ylim(0, max(accuracies) * 1.1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    plt.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=1.5, label=f'Overall Accuracy: {overall_accuracy:.2f}%')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analysis/deepl_acc.png')
    plt.show()

accuracy_results = test_retrieval_accuracy(preprocessed_database, tokenizers, model, 0.985) #这里调整阈值
plot_accuracy_results(accuracy_results)