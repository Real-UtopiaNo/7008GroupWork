from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备问题-答案对
knowledge_base = [
    ("What is AI?", "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines."),
    ("What is VR?", "Virtual Reality (VR) is a simulated experience that can be similar to or different from the real world."),
    ("What is blockchain?", "Blockchain is a decentralized digital ledger that records transactions across many computers.")
]

# 将句子转换为 BERT 向量
def sentence_to_bert_vector(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    # 取 [CLS] 标记的向量表示
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# 构建知识库中每个问题的 BERT 向量
knowledge_base_vectors = [sentence_to_bert_vector(pair[0]) for pair in knowledge_base]

# 检索答案
def retrieve_answer(question, knowledge_base, knowledge_base_vectors):
    question_vector = sentence_to_bert_vector(question)
    # 计算语义相似度
    cosine_similarities = np.array([cosine_similarity(question_vector, knowledge_base_vector).flatten() for knowledge_base_vector in knowledge_base_vectors])
    
    # 找到最相似的问题
    best_match_index = cosine_similarities.argmax()
    
    # 返回对应的答案
    return knowledge_base[best_match_index][1]

# 测试
user_question = "I want to know about AI tools."
answer = retrieve_answer(user_question, knowledge_base, knowledge_base_vectors)
print(answer)
