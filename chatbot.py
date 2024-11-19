from tkinter import *
import pandas as pd
import nltk
import os

# custom lib
from train_classifier import *
from machinel_chatbot import *
from deepl_chatbot import *

class Chatbot:
    def __init__(self):
        # init nltk
        nltk.data.path.append('nltk_data')

        # init database
        self.lists_of_data = os.listdir("Database")  # ["ai", "vr", ....]
        print(self.lists_of_data)
        self.database = self.load_database(self.lists_of_data) #return dict: database[data_type] = [(question1, answer1),...]

        """
        in: database
        out: classifier
        # init classifier (train with all data)
        """
        self.classifier = train_classifier(self.database)
        
        
        """
        in: database
        out: tfidf_matrix, vectorizer
        # init for TF-IDF method
        # construct TF-IDF vector for each data type
        """
        self.tfidf_matrix, self.vectorizers = tfidf_init(self.database)
        
        # init for deep learning method
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.preprocessed_database = preprocess_database(self.database, self.tokenizer)
        self.bert_model = DualEncoderModel()

        #init bot 
        self.bot_mode = "dl" # "tfidf" or "dl"

        # init UI
        self.root = Tk()
        self.root.title("Tech Chatbot")
        self.txt = Text(self.root)
        self.txt.grid(row=0, column=0, columnspan=2)
        self.e = Entry(self.root, width=100)
        self.e.grid(row=1, column=0)

    def load_database(self,lists_of_data):
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


    def run(self):
        self.send = Button(self.root, text="Send", command=self.send).grid(row=1, column=1)
        self.root.mainloop()

    def send(self):
    # get user input
        question = self.e.get()
        send = "You -> "+question
        self.txt.insert(END, send + "\n")
        question_type = Beyesian_classifier(self.classifier, question)
        print(question_type)
        if self.bot_mode == "tfidf":
            answer = tfidf_retrieve_answer(question, question_type,self.database[question_type], self.tfidf_matrix[question_type], self.vectorizers)
        elif self.bot_mode == "dl":
            best_match_dl, _ = bert_retrieve_answer(question, question_type, self.preprocessed_database, self.tokenizer, self.bert_model)
            answer = self.database[question_type][best_match_dl][1]
        send = "Bot -> " + answer
        self.txt.insert(END, send + "\n")
        self.e.delete(0, END)



if __name__ == '__main__':
    bot = Chatbot()
    bot.run()
