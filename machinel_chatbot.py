from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def tfidf_init(database):
    corpus = {}
    vectorizers = {}
    tfidf_matrix = {}
    for data_type in database:
        vectorizer = TfidfVectorizer(stop_words = "english")
        corpus[data_type] = [pair[0] for pair in database[data_type]]  # 只用问题作为语料
        tfidf_matrix[data_type] = vectorizer.fit_transform(corpus[data_type])
        vectorizers[data_type] = vectorizer
    return tfidf_matrix, vectorizers

def tfidf_retrieve_answer(question, question_type, data_base, tfidf_matrix, vectorizer):
    question_tfidf = vectorizer[question_type].transform([question])
    cosine_similarities = cosine_similarity(tfidf_matrix, question_tfidf).flatten()
    best_match_index = cosine_similarities.argmax()

    return data_base[best_match_index][1]