import random
import torch
import numpy as np

from tqdm import tqdm

from models import get_tokenizer

MAX_LEN_Q = 32

def expand_rubq(questions, answers, entities, relations):
    expanded_questions = []
    expanded_answers = []
    expanded_entities = []
    expanded_relations = []
    
    for i, q in enumerate(questions):
        for a in answers[i]:
            expanded_questions.append(q)
            expanded_answers.append([a,])
            expanded_entities.append(entities[i])
            expanded_relations.append(relations[i])
    
    return (expanded_questions, expanded_answers, expanded_relations, expanded_entities)

def load_simple_questions(seed, graph_embeddings_Q, graph_embeddings_P):
    simple_questions = np.load("../new_data/simple_questions_train.npy")
    
    simple_questions_filtered = []
    print(len(simple_questions))
    for e, p, a, q in tqdm(simple_questions):
        if e in graph_embeddings_Q and a in graph_embeddings_Q:
            simple_questions_filtered.append((e, p, a, q))
    print(len(simple_questions_filtered))

    simple_questions = simple_questions_filtered
    
    np.random.seed(seed)

    val_ids = np.random.randint(0, len(simple_questions), size=1000)
    train_ids = list(set(list(range(0, len(simple_questions)))) - set(val_ids))

    simple_questions_train = np.array(simple_questions)[train_ids]
    simple_questions_val = np.array(simple_questions)[val_ids]
    
    return simple_questions_train, simple_questions_val

def load_rubq(seed, graph_embeddings_Q, graph_embeddings_P):
    questions = list(np.load("../new_data/all_EN_rubq_val_questions_1_hop_uri.npy"))
    relations = list(np.load("../new_data/all_rubq_val_relations_1_hop_uri.npy"))
    entities = list(np.load("../new_data/all_rubq_val_entities_1_hop_uri.npy"))
    answers = list(np.load("../new_data/all_rubq_val_answers_1_hop_uri.npy",allow_pickle=True))

    questions_test = list(np.load("../new_data/all_EN_rubq_test_questions_1_hop_uri.npy"))
    answers_test = list(np.load("../new_data/all_rubq_test_answers_1_hop_uri.npy", allow_pickle=True))

    yes = []
    no = []
    no_ids = []
    print(len(questions))

    for i, answer in enumerate(answers):
        flag = True
        for a in answer:
            if not a in graph_embeddings_Q:
                flag = False
        if flag and relations[i] in graph_embeddings_P and entities[i] in graph_embeddings_Q:
            yes.append(answer)
        else:
            no.append(answer)
            no_ids.append(i)

    for i in no_ids[::-1]:
        del answers[i]
        del questions[i]
        del relations[i]
        del entities[i]


    print(len(questions))
    print(len(questions_test))

    np.random.seed(seed)

    val_ids = np.random.randint(0, len(questions), size=40)
    train_ids = list(set(list(range(0, len(questions)))) - set(val_ids))

    questions_train = np.array(questions)[train_ids]
    relations_train = np.array(relations)[train_ids]
    entities_train = np.array(entities)[train_ids]
    answers_train = np.array(answers)[train_ids]

    questions_val = np.array(questions)[val_ids]
    relations_val = np.array(relations)[val_ids]
    entities_val = np.array(entities)[val_ids]
    answers_val = np.array(answers)[val_ids]
    
    return questions_train, relations_train, entities_train, answers_train, questions_val, relations_val, entities_val, answers_val,questions_test, answers_test
    
class combined_dataset(torch.utils.data.Dataset):
    def __init__(self, questions, answers, entities, relations, graph_Q, graph_P, simple_questions, device):
        self.simple_questions = simple_questions
        self.questions = questions
        self.answers = answers
        self.entities = entities
        self.relations = relations
        self.graph_Q = graph_Q
        self.graph_P = graph_P
        self.tokenizer = get_tokenizer()
        self.device = device
        
    def __len__(self):
        return len(self.simple_questions)
    
    def __getitem__ (self,i):
        if i % 5 > 0:
            id_e, id_p, id_a, q = self.simple_questions[i]
            input_ids = torch.tensor([self.tokenizer.encode(q, max_length=MAX_LEN_Q, add_special_tokens=True, pad_to_max_length=True)]).to(self.device)[0]
            answer = id_a
            if id_p[0] == "P":
                relation = id_p
            else:
                relation = "P" + id_p[1:]
            graph_E_embedding = torch.FloatTensor(self.graph_Q[id_e])
            graph_Q_embedding = torch.FloatTensor(self.graph_Q[answer])
            graph_P_embedding = torch.FloatTensor(self.graph_P[relation])
            return (input_ids.to(self.device), graph_E_embedding.to(self.device), graph_Q_embedding.to(self.device), graph_P_embedding.to(self.device))
        else:
            i = random.randint(0,len(self.questions) - 1)
            input_ids = torch.tensor([self.tokenizer.encode(self.questions[i], max_length=MAX_LEN_Q, add_special_tokens=True, pad_to_max_length=True)]).to(self.device)[0]
            entity = self.entities[i]
            answer = self.answers[i]
            ind = random.randint(0,len(answer) - 1)
            answer = answer[ind]
            relation = self.relations[i]
            graph_E_embedding = torch.FloatTensor(self.graph_Q[entity])
            graph_Q_embedding = torch.FloatTensor(self.graph_Q[answer])
            graph_P_embedding = torch.FloatTensor(self.graph_P[relation])
            return (input_ids.to(self.device), graph_E_embedding.to(self.device), graph_Q_embedding.to(self.device), graph_P_embedding.to(self.device))
        
class combined_dataset_non_stochastic(torch.utils.data.Dataset):
    def __init__(self, questions, answers, entities, relations, graph_Q, graph_P, simple_questions, device):
        self.simple_questions = simple_questions
        self.questions = questions
        self.answers = answers
        self.entities = entities
        self.relations = relations
        self.graph_Q = graph_Q
        self.graph_P = graph_P
        self.tokenizer = get_tokenizer()
        self.device = device
        
    def __len__(self):
        return len(self.simple_questions) + len(self.questions)
    
    def __getitem__ (self,i):
        if i < len(self.simple_questions):
            id_e, id_p, id_a, q = self.simple_questions[i]
            input_ids = torch.tensor([self.tokenizer.encode(q, max_length=MAX_LEN_Q, add_special_tokens=True, pad_to_max_length=True)]).to(self.device)[0]
            answer = id_a
            if id_p[0] == "P":
                relation = id_p
            else:
                relation = "P" + id_p[1:]
            graph_E_embedding = torch.FloatTensor(self.graph_Q[id_e])
            graph_Q_embedding = torch.FloatTensor(self.graph_Q[answer])
            graph_P_embedding = torch.FloatTensor(self.graph_P[relation])
            return (input_ids.to(self.device), graph_E_embedding.to(self.device), graph_Q_embedding.to(self.device), graph_P_embedding.to(self.device))
        else:
            i = i - len(self.simple_questions)
            input_ids = torch.tensor([self.tokenizer.encode(self.questions[i], max_length=MAX_LEN_Q, add_special_tokens=True, pad_to_max_length=True)]).to(self.device)[0]
            entity = self.entities[i]
            answer = self.answers[i]
            ind = random.randint(0,len(answer) - 1)
            answer = answer[ind]
            relation = self.relations[i]
            graph_E_embedding = torch.FloatTensor(self.graph_Q[entity])
            graph_Q_embedding = torch.FloatTensor(self.graph_Q[answer])
            graph_P_embedding = torch.FloatTensor(self.graph_P[relation])
            return (input_ids.to(self.device), graph_E_embedding.to(self.device), graph_Q_embedding.to(self.device), graph_P_embedding.to(self.device))
        
class rubq_dataset(torch.utils.data.Dataset):
    def __init__(self, questions, answers, entities, relations, graph_Q, graph_P, device):
        self.questions = questions
        self.answers = answers
        self.entities = entities
        self.relations = relations
        self.graph_Q = graph_Q
        self.graph_P = graph_P
        self.tokenizer = get_tokenizer()
        self.device = device
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__ (self,i):
        input_ids = torch.tensor([self.tokenizer.encode(self.questions[i], max_length=MAX_LEN_Q, add_special_tokens=True, pad_to_max_length=True)]).to(self.device)[0]
        entity = self.entities[i]
        # answer = self.answers[i][0]
        answer = self.answers[i]
        ind = random.randint(0,len(answer) - 1)
        answer = answer[ind]
        relation = self.relations[i]
        graph_E_embedding = torch.FloatTensor(self.graph_Q[entity])
        graph_Q_embedding = torch.FloatTensor(self.graph_Q[answer])
        graph_P_embedding = torch.FloatTensor(self.graph_P[relation])
        return (input_ids.to(self.device), graph_E_embedding.to(self.device), graph_Q_embedding.to(self.device), graph_P_embedding.to(self.device))
    
class sq_dataset(torch.utils.data.Dataset):
    def __init__(self, graph_Q, graph_P, simple_questions, device):
        self.simple_questions = simple_questions
        self.graph_Q = graph_Q
        self.graph_P = graph_P
        self.tokenizer = get_tokenizer()
        self.device = device
        
    def __len__(self):
        return len(self.simple_questions)
    
    def __getitem__ (self,i):
        id_e, id_p, id_a, q = self.simple_questions[i]
        input_ids = torch.tensor([self.tokenizer.encode(q, max_length=MAX_LEN_Q, add_special_tokens=True, pad_to_max_length=True)]).to(self.device)[0]
        answer = id_a
        if id_p[0] == "P":
            relation = id_p
        else:
            relation = "P" + id_p[1:]
        graph_E_embedding = torch.FloatTensor(self.graph_Q[id_e])
        graph_Q_embedding = torch.FloatTensor(self.graph_Q[answer])
        graph_P_embedding = torch.FloatTensor(self.graph_P[relation])
        return (input_ids.to(self.device), graph_E_embedding.to(self.device), graph_Q_embedding.to(self.device), graph_P_embedding.to(self.device))