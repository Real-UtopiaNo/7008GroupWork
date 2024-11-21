import torch
from transformers import RobertaModel, RobertaTokenizer

class DualEncoderModel(torch.nn.Module):
    def __init__(self):
        super(DualEncoderModel, self).__init__()
        self.question_encoder = RobertaModel.from_pretrained('roberta-base')
        self.answer_encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, question_inputs, answer_inputs):
        question_outputs = self.question_encoder(**question_inputs)
        answer_outputs = self.answer_encoder(**answer_inputs)
        return question_outputs, answer_outputs

def preprocess_database(database, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessed_database = {}
    for question_type, _ in database.items():
        preprocessed_database[question_type] = []
        for (q, _) in database[question_type]:
            answer_inputs = tokenizer(q, return_tensors='pt', padding=True, truncation=True).to(device)
            preprocessed_database[question_type].append(answer_inputs)
    return preprocessed_database

def bert_retrieve_answer(question, question_type, preprocessed_database, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    best_match_index = -1
    best_score = float('-inf')

    question_inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True).to(device)

    for idx, answer_inputs_db in enumerate(preprocessed_database[question_type]):
        with torch.no_grad():
            question_outputs, answer_outputs = model(question_inputs, answer_inputs_db)
            score = torch.cosine_similarity(question_outputs.last_hidden_state.mean(dim=1),
                                            answer_outputs.last_hidden_state.mean(dim=1)).item()

        if score > best_score:
            best_score = score
            best_match_index = idx

    return best_match_index, best_score