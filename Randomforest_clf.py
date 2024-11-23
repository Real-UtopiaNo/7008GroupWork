import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
nltk.data.path.append('nltk_data')

def train_tf_classifier(database):
    texts = []
    labels = []
    for data_type in database:
        print("Training for data type: ", data_type)
        for pair in database[data_type]:
            texts.append(pair[0].lower())
            labels.append(data_type)
    
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english')
    classifier = RandomForestClassifier(n_estimators=200, max_depth=40,random_state=42)
    rfmodel = make_pipeline(vectorizer, classifier)
    
    rfmodel.fit(texts, labels)
    return rfmodel

def Randomforest_classifier(rfmodel, question):
    return rfmodel.predict([question])[0]