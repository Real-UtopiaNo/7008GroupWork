from nltk.tokenize import word_tokenize
import nltk
nltk.data.path.append('nltk_data')
def extract_features(text):
    words = word_tokenize(text)
    return {word.lower(): True for word in words}

def train_bys_classifier(database):
    classifier_training_set = []  # 遍历整个数据库，[(question, data_type),...]
    for data_type in database:
        print("Training for data type: ",data_type)
        for pair in database[data_type]:
            classifier_training_set.append((extract_features(pair[0]), data_type))
    classifier = nltk.NaiveBayesClassifier.train(classifier_training_set)
    return classifier

def Beyesian_classifier(classifier, question):
    features = extract_features(question)
    return classifier.classify(features)