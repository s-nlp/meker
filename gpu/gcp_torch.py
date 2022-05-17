import numpy as np
import sys
import datetime
import math
import argparse

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import pickle
import time
from ipypb import track
import argparse
from timeit import default_timer as timer
from sklearn.preprocessing import normalize

from t_alg import mttcrp, mttcrp1, get_elem_deriv_tensor, factors_to_tensor, gcp_grad, multi_ind_to_indices, indices_to_multi_ind
from samplings import give_ns, generate_data
from elementwise_grads import bernoulli_logit_loss, bernoulli_logit_loss_grad, bernoulli_loss, bernoulli_loss_grad
from general_functions1 import sqrt_err_relative, check_coo_tensor, gen_coo_tensor, create_filter, hr

from decimal import Decimal
from timeit import default_timer as timer

from experiments import data_storage, Trainer, run_epoch

from model import MEKER

import os
from os.path import dirname, abspath
path_parent = dirname(dirname(abspath(__file__))) 
print ("path parent ", path_parent)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(fail_count=0)
def check_early_stop(target_score, previous_best, margin=0, max_attempts=1000):
    
    if (previous_best > target_score):
        previous_best = target_score
    if (margin >= 0) and (target_score > previous_best + margin):
        print ("fail_count ", check_early_stop.fail_count)
        check_early_stop.fail_count += 1
        print ("fail_count ", check_early_stop.fail_count)
    elif (math.isnan(target_score) or math.isinf(target_score)):
        print ("is_nun ", check_early_stop.fail_count)
        check_early_stop.fail_count += 1
        print ("is_nun ", check_early_stop.fail_count)
    else:
        check_early_stop.fail_count = 0
    if check_early_stop.fail_count >= max_attempts:
        print('Interrupted due to early stopping condition.', check_early_stop.fail_count, flush = True)
        raise StopIteration

@static_vars(fail_count_score=0)        
def check_early_stop_score(target_score, previous_best, margin=0, max_attempts=3000):
    if (previous_best < target_score):
        previous_best = target_score
    if (margin >= 0) and (target_score + margin < previous_best):
        print ("fail_count ", check_early_stop_score.fail_count_score)
        check_early_stop_score.fail_count_score += 1
        print ("fail_count ", check_early_stop_score.fail_count_score)
    else:
        check_early_stop_score.fail_count_score = 0
    if check_early_stop_score.fail_count_score >= max_attempts:
        print('Interrupted due to early stopping scoring condition.', check_early_stop_score.fail_count_score, flush = True)
        raise StopIteration



def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, required=True, default=200)
    parser.add_argument("--lr", type=float, required=True) # depends on choice of data pack
    parser.add_argument("--path_data", type=str, default="/notebook/Relations_Learning/Link_Prediction_Data/FB15K237/")
    parser.add_argument("--path_filters", type=str, default="/notebook/Relations_Learning/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--opt_type", type=str, default='adam')

    parser.add_argument('--dim', type = int, default = 100)
    parser.add_argument('--how_many', type = int, default = 200)
    parser.add_argument('--l2', type = float, default = 0.0)
    parser.add_argument('--scheduler_step', type=int, default=2, help="Scheduler step size")
    parser.add_argument("--scheduler_gamma", type=float, default = 0.5, help="scheduler_gamma")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum in Sgd')
    parser.add_argument('--nesterov', type=bool, default=False, help='nesterov momentum in Sgd')
    parser.add_argument('--out_file', type=str, default='/notebook/Relations_Learning/grid_search/output_files/out.txt', help='path tot output_file')

    parser.add_argument("--seed", type=int, default=55, nargs="?",
                        help="random seed")

    args = parser.parse_args()
    seed = args.seed
    num_epoch = args.n_epoch
    rank = args.dim 
    lr = args.lr
    batch_size = args.batch_size
    how_many = args.how_many
    l2 = args.l2

    print ("path_parent + 'test_filter.pkl'", path_parent + '/test_filter.pkl')
    path_data = path_parent + "/Link_Prediction_Data/FB15K237/"
    
    with open(path_data + '/test_filter.pkl', 'rb') as f:
        test_filter = pickle.load(f)
    
    with open(path_data + '/valid_filter.pkl', 'rb') as f:
        valid_filter = pickle.load(f)
    
    entity_list = pickle.load(open(path_data + 'entity_list', 'rb'))
    relation_list = pickle.load(open(path_data + 'relation_list', 'rb'))

    train_triples = pickle.load(open(path_data + 'train_triples', 'rb'))
    valid_triples = pickle.load(open(path_data + 'valid_triples', 'rb'))
    test_triples = pickle.load(open(path_data + 'test_triples', 'rb'))
    train_valid_triples = pickle.load(open(path_data + 'train_valid_triples', 'rb'))

    entity_map = pickle.load(open(path_data + 'entity_map', 'rb'))
    relation_map = pickle.load(open(path_data + 'relation_map', 'rb'))

    all_triples = train_valid_triples + test_triples

    hm = 1000
    
    values = [1] * len(train_triples)
    values = np.array(values, dtype=np.int64)
    coords = np.array(train_triples, dtype=np.int64)
    nnz = len(train_triples)
    data_shape = (len(entity_list), len(relation_list), len(entity_list))

    print (data_shape, flush = True)

    coo_tensor = coords
    vals = values
    shape = data_shape

    num_epoch = args.n_epoch

    random_state = np.random.seed(seed)

    # specify property of data
    init_mind_set = set(indices_to_multi_ind(coo_tensor, shape))
    coo_ns = np.empty((how_many * len(init_mind_set) + vals.size, 3), dtype=np.int64)
    vals_ns = np.empty((how_many * len(init_mind_set) + vals.size,), dtype=np.float64)

    data_s = data_storage(sparse_coords = coords, sparse_vals =values, mind_set = init_mind_set, shape=data_shape, how_many=args.how_many, valid_filters = valid_filter, valid_triples = valid_triples, test_filters = test_filter, test_triples = test_triples)

    # specify property of training
    err_arr = []
    error = 0.0
    it = 0
    previous_best_loss = 100000.0
    best_tuple = (0.0, 0.0, 0.0, 0.0)
    best_hit_10 = 0.0
    # specify training class
    trainer = Trainer(best_hit_10, previous_best_loss, err_arr, it)

    
    # create model object
    model = MEKER(rank=rank, shape=data_shape, given_loss=bernoulli_logit_loss, given_loss_grad=bernoulli_logit_loss_grad, device=device)
    model.init()

    optimizer = optim.AdamW([model.a_torch, model.b_torch], lr = args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    show_iter = True
    for epoch in range(num_epoch):
        try:
            run_epoch(data_s, epoch, device, model, optimizer, scheduler, batch_size, trainer, show_iter = True)
        except StopIteration: # early stopping condition met
            break
            print ("early_stoping loss", flush = True)
            raise StopIteration
            

        hit3, hit5, hit10, mrr = model.evaluate(data_s)
        print (hit3, hit5, hit10, mrr, flush = True)
        
        # early stopping by hit@10
        try:
            check_early_stop_score(hit10, best_hit_10, margin=0.01, max_attempts=1000)
        except StopIteration: # early stopping condition met
                break
                print ("early_stoping score", flush = True)
        
        # if hit@10 grows update checkpoint
        if (hit10 > best_hit_10):
            best_hit_10 = hit10
            #np.save(path_parent + '/gpu/gpu_a.npz', model.a_torch.cpu().data.numpy())
            #np.save(path_parent + '/gpu/gpu_b.npz', model.b_torch.cpu().data.numpy())
        
if __name__ == "__main__":
    main()

