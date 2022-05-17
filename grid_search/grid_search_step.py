import os
import numpy as np

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/notebook/Relations_Learning/gpu/')

import math

import argparse
import wandb

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import normalize

import pickle
from ipypb import track
import argparse

from t_alg import mttcrp, mttcrp1, get_elem_deriv_tensor, factors_to_tensor, gcp_grad, multi_ind_to_indices, indices_to_multi_ind

from samplings import give_ns, generate_data

from elementwise_grads import bernoulli_logit_loss, bernoulli_logit_loss_grad, bernoulli_loss, bernoulli_loss_grad

from general_functions1 import sqrt_err_relative, check_coo_tensor, gen_coo_tensor
from general_functions1 import create_filter, hr

from decimal import Decimal
from timeit import default_timer as timer

from experiments import data_storage, Trainer, run_epoch

from model import MEKER



# in yaml
# num_epoch = args.n_epoch
# rank = args.dim 
# lr = args.lr
# batch_size = args.batch_size
# step_size=args.scheduler_step
# gamma=args.scheduler_gamma
# momentum = args.momentum
# opt_type = args.opt_type (SGD, ADAM, AdamW)
# output_file = args.out_file

import wandb

from util import import_source_as_module

import_source_as_module('/notebook/Relations_Learning/grid_search/configs/adam_grid.py')
cur_dir = os.path.dirname(os.path.abspath(__file__))

import adam_grid  # python file with default hyperparameters
# Set up your default hyperparameters
hyperparameters = adam_grid

# Pass them wandb.init
wandb.init(project = 'FOxIE2', entity = 'foxie')
# Access all hyperparameter values through wandb.config
#config = wandb.config

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
        check_early_stop.fail_count += 1
    elif (math.isnan(target_score) or math.isinf(target_score)):
        check_early_stop.fail_count += 1
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


parser = argparse.ArgumentParser()
parser.add_argument("--n_epoch", type=int, required=True, default=200)
parser.add_argument("--lr", type=float, required=True) # depends on choice of data pack
parser.add_argument("--path_data", type=str, default="/notebook/Relations_Learning/Link_Prediction_Data/FB15K237/")
parser.add_argument("--path_filters", type=str, default="/notebook/Relations_Learning/")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--opt_type", type=str, default='adam')

parser.add_argument('--dim', type = int, default = 200)
parser.add_argument('--how_many', type = int, default = 200)
parser.add_argument('--l2', type = float, default = 0.0)
parser.add_argument('--scheduler_step', type=int, default=2, help="Scheduler step size")
parser.add_argument("--scheduler_gamma", type=float, default = 0.5, help="scheduler_gamma")
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in Sgd')
parser.add_argument('--nesterov', type=bool, default=False, help='nesterov momentum in Sgd')
parser.add_argument('--out_file', type=str, default='/notebook/Relations_Learning/grid_search/output_files/out.txt', help='path tot output_file')

args = parser.parse_args()

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    
output_file = cur_dir + "/output_files/" + str(datetime.datetime.now())+".txt" # files for text output
    
dim = 200
    
file_out = open(output_file, "w")

path_data = args.path_data
path_filters = args.path_filters

with open(path_filters + 'test_filter.pkl', 'rb') as f:
    test_filter = pickle.load(f)
    
with open(path_filters + 'valid_filter.pkl', 'rb') as f:
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

file_out.write("loaded1_\n")
num_epoch = args.n_epoch
rank = args.dim 
lr = args.lr
batch_size = args.batch_size
    
    
seed = 13 
how_many = args.how_many
l2 = args.l2
    
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
    
data_s = data_storage(sparse_coords = coords, sparse_vals =values, mind_set = init_mind_set, shape=data_shape, how_many=args.how_many, valid_filters = valid_filter, valid_triples = valid_triples)

# specify property of training
err_arr = []
error = 0.0
it = 0
previous_best_loss = 100000.0
best_tuple = (0.0, 0.0, 0.0, 0.0)
best_hit_10 = 0.0
# specify training class
trainer = Trainer(best_hit_10, previous_best_loss, err_arr, it)
    
model = MEKER(rank=rank, shape=data_shape, given_loss=bernoulli_logit_loss, given_loss_grad=bernoulli_logit_loss_grad, device=device)
model.init()

score_margin_ = 0.0
score_attempts_ = 4
optimizer = optim.Adam([model.a_torch, model.b_torch], lr = args.lr)

if (args.opt_type == 'sdg'):
    score_margin_ = 0.0
    score_attempts_ = 5
    optimizer = optim.SGD([model.a_torch, model.b_torch], lr = args.lr, momentum = args.momentum)
          
elif (args.opt_type == 'adamw'):
    score_margin_ = 0.0
    score_attempts_ = 4
    optimizer = optim.AdamW([model.a_torch, model.b_torch], lr = args.lr)
          
scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

show_iter = True
for epoch in range(num_epoch):
    try:
        d = 6
        run_epoch(data_s, epoch, device, model, optimizer, scheduler, batch_size, trainer, show_iter = True, fout = file_out)
    except StopIteration: # early stopping condition met
        break
        print ("early_stoping loss", flush = True)
        raise StopIteration

    if (epoch%5 == 0):
        hit3, hit5, hit10 = model.evaluate(data_s)
        mrr = 0

        metrics = {"hit3": hit3,
                    "hit5": hit5,
                    "hit10": hit10,
                    "mrr":mrr
                  }
        wandb.log(metrics)
        print (hit3, hit5, hit10, mrr, flush = True)
        
        try:
            check_early_stop_score(hit10, best_hit_10, margin=score_margin_, max_attempts=score_attempts_)
        except StopIteration: # early stopping condition met
            end = timer()
            time = end - start
            file_out.write("\n")
            file_out.write("In %s epoch; time %s \n" % (epoch, time))
            file_out.write("early_stoping score")
            file_out.write("Best scores %s %s %s %s \n" % (best_tuple[0], best_tuple[1], best_tuple[2], best_tuple[3]))
            file_out.flush()
            print ("early_stoping score", flush = True)
            wandb.summary.update({'epoch': epoch, **metrics})
            break
    
    loss_metric = {"loss":np.mean(trainer.err_arr[len(trainer.err_arr) - 1])}
    wandb.log(loss_metric)

    file_out.write('%s %s %s %s \n' % (hit3, hit5, hit10, mrr))
    file_out.flush()
        # early stopping by hit@10

    # if hit@10 grows update checkpoint
    if (hit10 > best_hit_10):
        best_hit_10 = hit10
        best_tuple = (hit3, hit5, hit10, mrr)
        #np.save('/notebook/Relations_Learning/gpu/gpu_a.npz', a_torch.cpu().data.numpy())
        #np.save('/notebook/Relations_Learning/gpu/gpu_b.npz', b_torch.cpu().data.numpy())
        #np.save('/notebook/Relations_Learning/gpu/gpu_c.npz', a_torch.cpu().data.numpy())
    
file_out.write("In %s epoch; time %s \n" % (epoch, time))
file_out.write("Best scores %s %s %s %s \n" % (best_tuple[0],best_tuple[1],best_tuple[2],best_tuple[3]))
file_out.write("\n")
file_out.flush()
file_out.close()
                   