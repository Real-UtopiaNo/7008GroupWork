import torch
from transformers import BertModel, BertTokenizer

class DualEncoderModel(torch.nn.Module):
    def __init__(self):
        super(DualEncoderModel, self).__init__()
        self.question_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.answer_encoder = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, question_inputs, answer_inputs):
        question_outputs = self.question_encoder(**question_inputs)
        answer_outputs = self.answer_encoder(**answer_inputs)
        return question_outputs, answer_outputs

def bert_retrieve_answer(question, question_type, database, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    best_match_index = -1
    best_score = float('-inf')

    question_inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True).to(device)

    for idx, (q, a) in enumerate(database[question_type]):
        answer_inputs = tokenizer(q, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            question_outputs, answer_outputs = model(question_inputs, answer_inputs)
            score = torch.cosine_similarity(question_outputs.last_hidden_state.mean(dim=1),
                                            answer_outputs.last_hidden_state.mean(dim=1)).item()

        if score > best_score:
            best_score = score
            best_match_index = idx

    return database[question_type][best_match_index][1]