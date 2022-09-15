from transformers import BertTokenizer, BertModel
from torch import nn
import torch

def get_projection_module_simple(device, hidden_size):
    model = nn.Sequential(
        nn.Linear(768, hidden_size),
        nn.ELU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ELU(),
        nn.Linear(hidden_size,200),
    ).to(device).train()
    
    return model

def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class EncoderBERT(nn.Module):
    def __init__(self, device):
        super(EncoderBERT,self).__init__()
        model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
        self.encoder =  model
        
    def forward(self,questions_ids):
        q_ids = torch.tensor(questions_ids)
        last_hidden_states = self.encoder(q_ids)[0]
        q_emb = last_hidden_states.mean(1)
        return q_emb