from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def extract_features(text):
    words = word_tokenize(text)
    return {word.lower(): True for word in words}

def train_classifier(database):
    texts = []
    labels = []
    for data_type in database:
        print("Training for data type: ", data_type)
        for pair in database[data_type]:
            texts.append(pair[0])
            labels.append(data_type)
    
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english')
    classifier = RandomForestClassifier(n_estimators=200, max_depth=40, random_state=42)
    model = make_pipeline(vectorizer, classifier)
    
    model.fit(texts, labels)
    return model

def Beyesian_classifier(model, question):
    return model.predict([question])[0]