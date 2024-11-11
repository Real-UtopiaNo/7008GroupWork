from tkinter import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import os

class Chatbot:
    def __init__(self):
        # init nltk
        nltk.data.path.append('nltk_data')

        # init database
        self.lists_of_data = os.listdir("Database")  # ["ai", "vr", ....]
        self.database = self.load_database(self.lists_of_data) #return dict: database[data_type] = [(question1, answer1),...]

        # init classifier (train with all data)
        self.classifier = self.train_classifier(self.database) #return bayesian classifier
        
        # init for TF-IDF method
        # construct TF-IDF vector for each data type
        self.corpus = {}
        self.vectorizer = TfidfVectorizer(stop_words = "english")
        self.tfidf_matrix = {}
        for data_type in self.database.keys():
            self.corpus[data_type] = [pair[0] for pair in self.database[data_type]]  # 只用问题作为语料
            self.tfidf_matrix[data_type] = self.vectorizer.fit_transform(self.corpus[data_type])
        
        # init for deep learning method
        # 
        # 
        # 

        #init bot 
        self.bot_mode = "tfidf"

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
            database_type = [] # QA for certain type
            path = "Database\\" + type
            csvfiles = os.listdir(path)
            for csvfile in csvfiles:
                data = pd.read_csv(path + "\\" + csvfile)
                for i in range(len(data)):
                    database_type.append((data["Question"][i], data["Answer"][i]))
        database[type] = database_type
        return database

    def train_classifier(self,database):
        classifier_training_set = []  # 遍历整个数据库，[(question, data_type),...]
        for data_type in database.keys():
            for pair in database[data_type]:
                classifier_training_set.append((self.extract_features(pair[0]), data_type))
        classifier = nltk.NaiveBayesClassifier.train(classifier_training_set)
        return classifier

    def extract_features(self,text):
        words = word_tokenize(text)
        return {word.lower(): True for word in words}

    def run(self):
        self.send = Button(self.root, text="Send", command=self.send).grid(row=1, column=1)
        self.root.mainloop()

    def Beyesian_classifier(self,question):
        features = self.extract_features(question)
        return self.classifier.classify(features)
        

    def tfidf_retrieve_answer(self,question, data_base, tfidf_matrix):
        question_tfidf = self.vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
        best_match_index = cosine_similarities.argmax()

        return data_base[best_match_index][1]


    def send(self):
    # get user input
        question = self.e.get()
        send = "You -> "+question
        self.txt.insert(END, send + "\n")
        question_type = self.Beyesian_classifier(question)
        print(question_type)
        if self.bot_mode == "tfidf":
            answer = self.tfidf_retrieve_answer(question, self.database[question_type], self.tfidf_matrix[question_type])
        elif self.bot_mode == "deep learning":
            pass
        send = "Bot -> " + answer
        self.txt.insert(END, send + "\n")
        self.e.delete(0, END)



if __name__ == '__main__':
    bot = Chatbot()
    bot.run()
