import numpy as np
import pandas as pd
import pickle
import argparse
import sys

import torch
from torch.optim.lr_scheduler import StepLR

from t_alg import mttcrp, mttcrp1, get_elem_deriv_tensor, factors_to_tensor, gcp_grad, multi_ind_to_indices, indices_to_multi_ind
from samplings import give_ns, generate_data
from elementwise_grads import bernoulli_logit_loss, bernoulli_logit_loss_grad, bernoulli_loss, bernoulli_loss_grad
from general_functions1 import sqrt_err_relative, check_coo_tensor, gen_coo_tensor
from general_functions1 import create_filter, hr
#from evaluation_functions import hr

import os
from os.path import dirname, abspath
path_parent = dirname(dirname(abspath(__file__))) 
print ("path parent ", path_parent)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(fail_count=0)
def check_early_stop(target_score, previous_best, margin=0, max_attempts=10):
    if (previous_best > target_score):
        previous_best = target_score
    if (margin >= 0) and (target_score < previous_best + margin):
        check_early_stop.fail_count += 1
    else:
        check_early_stop.fail_count = 0
    if check_early_stop.fail_count >= max_attempts:
        print('Interrupted due to early stopping condition.')
        raise StopIteration
    return previous_best

@static_vars(fail_count_score=0)   
def check_early_stop_score(target_score, previous_best, margin=0, max_attempts=10):
    if (previous_best > target_score):
        previous_best = target_score
    if (margin >= 0) and (target_score < previous_best + margin):
        check_early_stop.fail_count_score += 1
    else:
        check_early_stop.fail_count_score = 0
    if check_early_stop.fail_count_score >= max_attempts:
        print('Interrupted due to early stopping condition.')
        raise StopIteration

def gcp_grad(coo, val, shape, a, b, l2, loss_function, loss_function_grad, device):
    """
        GCP loss function and gradient calculation.
        All the tensors have the same coordinate set: coo_tensor.
    """

    # Construct sparse kruskal tensor
    kruskal_val = torch.sum((a[coo[:,0], :] * b[coo[:,1], :] * a[coo[:,2], :]),1)
    
    # Calculate mean loss on known entries
    loss = loss_function(val, kruskal_val)
    # Compute the elementwise derivative tensor
    deriv_tensor_val = loss_function_grad(val, kruskal_val)

    # Calculate gradients w.r.t. a, b, c factor matrices
    g_a = mttcrp1(coo, deriv_tensor_val, shape, 0, b, a, device)
    g_b = mttcrp1(coo, deriv_tensor_val, shape, 1, a, a, device)
    g_c = mttcrp1(coo, deriv_tensor_val, shape, 2, a, b, device)
    
    if l2 != 0:
        g_a += l2 * a[coo[0], :]
        g_b += l2 * b[coo[1], :]
        g_c += l2 * c[coo[2], :]
    
    return loss, g_a, g_b, g_c
    


def main():
    print ("loaded 0", flush = True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=200, nargs="?",
                    help="set desored emebdding dimention")

    args = parser.parse_args()
    dim = args.dim

        
    device=torch.device("cuda:2")
    print (device)

    print ("path_parent + 'test_filter.pkl'", path_parent + '/test_filter.pkl')
    path_data = path_parent + "/Link_Prediction_Data/Wiki4M/"
    
    with open(path_data + '/test_filter.pkl', 'rb') as f:
        test_filter = pickle.load(f)
    
    train_triples = pickle.load(open(path_data + 'train', 'rb'))
    valid_triples = pickle.load(open(path_data + 'valid', 'rb'))
    test_triples = pickle.load(open(path_data + 'test', 'rb'))
    train_valid_triples = pickle.load(open(path_data + 'all_triples_all', 'rb'))

    entity_map = pickle.load(open(path_data + 'ents_map', 'rb'))
    relation_map = pickle.load(open(path_data + 'relation_map', 'rb'))

    all_triples = train_valid_triples + test_triples

    print ("loaded 1", flush = True)
    num_epoch = 15
    rank = dim 
    lr = 1e-2
    seed = 13 
    hm = 1000
    how_many = 5
    l2 = 0
    
    values = [1] * len(train_triples)
    values = np.array(values, dtype=np.int64)

    coords = np.array(train_triples, dtype=np.int64)
    nnz = len(train_triples)
    data_shape = (len(entity_map), len(relation_map), len(entity_map))
    
    print ("data_shape", data_shape, flush = True)
    
    
    coo_tensor = coords
    vals = values
    shape = data_shape
    loss_function = bernoulli_logit_loss
    loss_function_grad = bernoulli_logit_loss_grad

    from torch.nn.init import xavier_normal_
    from torch import optim

    device=torch.device("cuda:3")


    random_state = np.random.seed(seed)

    batch_size = 400
    init_mind_set = set(indices_to_multi_ind(coo_tensor, shape))
   
    error = 0.0
    it = 0
    
    a_torch = torch.empty((shape[0], rank), requires_grad = True, device = device)
    xavier_normal_(a_torch)
    a_torch.grad = torch.zeros(a_torch.shape, device = device)

    b_torch = torch.empty((shape[1], rank), requires_grad = True, device = device)
    xavier_normal_(b_torch)
    b_torch.grad = torch.zeros(b_torch.shape, device = device)
    
    a = a_torch.cpu().data.numpy()
    b = b_torch.cpu().data.numpy()
    hit1, hit3, hit10, mrr = hr(test_filter[:300], test_triples[:300], a, b, a, [1, 3, 10], iter_show=True, freq=300)
    print ("hr: ", hit1, hit3, hit10, mrr)
    
    optimizer = optim.AdamW([a_torch, b_torch], lr=1e-3, eps=1e-08, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.8)
    
    del a
    del b
    
    best_hit_10 = 0.0
    best_mrr = 0.0
    for epoch in range(30):      
                
        init_mind_set = set(indices_to_multi_ind(coo_tensor, shape))
        coo_ns = np.empty((how_many * len(init_mind_set) + vals.size, 3), dtype=np.int64)
        vals_ns = np.empty((how_many * len(init_mind_set) + vals.size,), dtype=np.float64)

        err_arr = np.empty((num_epoch*vals_ns.shape[0]//batch_size + 1, ), dtype=np.float64)
        error = 0.0
        it = 0
        show_iter = True
        coo_ns, vals_ns = generate_data(coo_tensor, vals, init_mind_set, shape, how_many, epoch)
        #coo_ns = torch.tensor(coo_ns, device = device)
        #vals_ns = torch.tensor(vals_ns, device = device)
        shuffler = np.random.permutation(vals_ns.shape[0])
        coo_ns = coo_ns[shuffler]
        vals_ns = vals_ns[shuffler]
        print (vals_ns.shape[0], batch_size, vals_ns.shape[0]//batch_size)
        err_list = []
        
        
        for i in range(vals_ns.shape[0]//batch_size):
            end = min(vals_ns.shape[0] - 1, (i+1)*batch_size)
            loss, g_a, g_b, g_c = gcp_grad(
                torch.tensor(coo_ns[i*batch_size : end], requires_grad = False, device = device), torch.tensor(vals_ns[i*batch_size : end], requires_grad = False, device = device), shape, a_torch, b_torch, l2, loss_function, loss_function_grad, device)
            # print ("loss", loss.cpu().detach().numpy().mean())
            if (torch.isnan(loss).any() or torch.isinf(loss).any()):
                print ("loss isnan loss is inf")
                print ("loss", loss.cpu().detach().numpy().mean())
                sys.exit()

                
            err_list.append(loss.cpu().detach().numpy().mean())

            a_elems = coo_ns[i*batch_size : end, 0]
            b_elems = coo_ns[i*batch_size : end, 1]
            c_elems = coo_ns[i*batch_size : end, 2]

            a_torch.grad[a_elems, :] = g_a
            b_torch.grad[b_elems, :] = g_b
            a_torch.grad[c_elems, :] = g_c
            
            optimizer.step()            
            
            a_torch.grad = torch.zeros(a_torch.shape, device = device)
            b_torch.grad = torch.zeros(b_torch.shape, device = device)

            err_arr[it] = np.mean(err_list)
            if show_iter and i%100 == 0:
                print("Iter: ", it, "; Error: ", np.mean(np.array(err_list)), flush = True)
            it += 1  
        
        scheduler.step()
        
        if (epoch % 1 == 0):
            print ("count hr", flush = True)
            a = a_torch.cpu().data.numpy()
            b = b_torch.cpu().data.numpy()
            hit1, hit3, hit10, mrr = hr(test_filter[:3000], test_triples[:3000], a, b, a, [1, 3, 10], iter_show=True, freq=300)
            print (hit1, hit3, hit10, mrr, flush = True)
            del a
            del b
        
        if (hit1 > best_hit_10 or mrr > best_mrr):
            best_hit_10 = hit10
            best_mrr = mrr
            np.save(path_parent + '/gpu/gpu_a.npz', a_torch.cpu().data.numpy())
            np.save(path_parent + '/gpu/gpu_b.npz', b_torch.cpu().data.numpy())
        
        del coo_ns
        del vals_ns
        
        
        with torch.cuda.device('cuda:2'):
            torch.cuda.empty_cache()

    
    
if __name__ == "__main__":
    main()

