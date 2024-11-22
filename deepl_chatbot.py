import torch
from transformers import RobertaModel, RobertaTokenizer
import os
import json
import hashlib

class QuestionEncoder(torch.nn.Module):
    def __init__(self):
        super(QuestionEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')
    
    def forward(self, inputs):
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

# judgement of the database
def generate_database_hash(database):
    database_str = json.dumps(database, sort_keys=True)
    return hashlib.md5(database_str.encode()).hexdigest()

def preprocess_database(database, tokenizer, cache_dir="./cache"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    database_hash = generate_database_hash(database)
    cache_file = os.path.join(cache_dir, f"encoded_questions_{database_hash}.pt")
    
    if os.path.exists(cache_file):
        print("loading existing cache file...")
        encoded_questions = torch.load(cache_file)
        for question_type in encoded_questions:
            encoded_questions[question_type] = encoded_questions[question_type].to(device)
        return encoded_questions
    
    print("cache file not found, encoding questions...")
    model = QuestionEncoder().to(device)
    model.eval()
    
    encoded_questions = {}
    
    for question_type, qa_pairs in database.items():
        encoded_list = []
        
        for (q, _) in qa_pairs:
            with torch.no_grad():
                q_inputs = tokenizer(q, return_tensors='pt', padding=True, truncation=True).to(device)
                q_vector = model(q_inputs)
                encoded_list.append(q_vector)
                
        encoded_questions[question_type] = torch.cat(encoded_list, dim=0)
    
    encoded_questions_cpu = {
        k: v.cpu() for k, v in encoded_questions.items()
    }
    torch.save(encoded_questions_cpu, cache_file)
    print(f"encode result saved to: {cache_file}")
    
    return encoded_questions

def dl_retrieve_question(query, question_type, encoded_database, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    query_inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        query_vector = model(query_inputs)

        similarities = torch.cosine_similarity(
            query_vector.unsqueeze(1), 
            encoded_database[question_type].unsqueeze(0), 
            dim=2
        )
        
        best_match_index = torch.argmax(similarities).item()
        best_score = similarities[0][best_match_index].item()

    return best_match_index, best_score