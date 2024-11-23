from tkinter import *
import pandas as pd
import nltk
import os

# custom lib
from Bayesian_clf import *
from Roberta_clf import *
from Randomforest_clf import *
from machinel_chatbot import *
from deepl_chatbot import *

class Chatbot:
    def __init__(self):
        # init nltk
        nltk.data.path.append('nltk_data')

        #init bot 
        self.bot_mode = "dl"
        self.bot_classifier = "Roberta"

        # init database
        self.lists_of_data = os.listdir("Database")  # ["ai", "vr", ....]
        print(self.lists_of_data)
        self.database = self.load_database(self.lists_of_data) #return dict: database[data_type] = [(question1, answer1),...]


        # init classifier (train with all data)
        if self.bot_classifier == "Bayesian":
            self.classifier = train_bys_classifier(self.database)
        elif self.bot_classifier == "RandomForest":
            self.rfmodel = train_tf_classifier(self.database)
        elif self.bot_classifier == "Roberta":
            self.roberta_tokenizer, self.roberta_model = init_roberta_clf()


        # init for TF-IDF method
        if self.bot_mode == "tfidf":
            self.tfidf_matrix, self.vectorizers = tfidf_init(self.database)
        
        # init for deep learning method
        if self.bot_mode == "dl":
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.preprocessed_database = preprocess_database(self.database, self.tokenizer)
            self.bert_model = QuestionEncoder()


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
            # path = "Database\\" + type
            path = os.path.join("Database", type)
            csvfiles = os.listdir(path)
            for csvfile in csvfiles:
                data = pd.read_csv(os.path.join(path,csvfile))
                for i in range(len(data)):
                    database_type.append((data["Question"][i], data["Answer"][i]))
            database[type] = database_type
        return database


    def run(self):
        self.send = Button(self.root, text="Send", command=self.send).grid(row=1, column=1)
        self.root.mainloop()

    def send(self):
    # get user input
        question = self.robustness_input(self.e.get())
        if question != None:
            send = "You -> "+question
            question = question.lower()
            self.txt.insert(END, send + "\n")
            if self.bot_classifier == "Bayesian":
                question_type = Beyesian_classifier(self.classifier, question)
            elif self.bot_classifier == "Roberta":
                question_type = Roberta_classifier(question,self.roberta_tokenizer, self.roberta_model)
            elif self.bot_classifier == "RandomForest":
                question_type = Randomforest_classifier(self.rfmodel, question)
            print(question_type)
            if self.bot_mode == "tfidf":
                answer, _, _ = tfidf_retrieve_answer(question, question_type,self.database[question_type], self.tfidf_matrix[question_type], self.vectorizers)
            elif self.bot_mode == "dl":
                best_match_dl, _ = dl_retrieve_question(question, question_type, self.preprocessed_database, self.tokenizer, self.bert_model)
                answer = self.database[question_type][best_match_dl][1]
            send = "Bot -> " + answer
            self.txt.insert(END, send + "\n")
            self.e.delete(0, END)
        else:
            send = "Bot -> " + "Invalid input!"
            self.txt.insert(END, send + "\n")
            self.e.delete(0, END)

    def robustness_input(self, question):
        if len(question)<=1:
            return None
        else:
            return question



if __name__ == '__main__':
    bot = Chatbot()
    bot.run()
