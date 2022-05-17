import numpy as np
import pandas as pd

from t_alg import multi_ind_to_indices, indices_to_multi_ind

def give_ns(multi_inx_set, tensor_shape, how_many=1, seed=13, show_iter=False):
    random_state = np.random.seed(seed)
    ns_size = how_many * len(multi_inx_set)
    mixs = multi_inx_set.copy()
    ns = np.zeros(ns_size, dtype=np.int64)
    all_ind = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    for i in range(ns_size):
        check = True
        while check:
            cand = np.random.choice(all_ind)
            if cand not in mixs: 
                mixs.add(cand)
                ns[i] = cand
                check = False
        if show_iter:        
            if i % 10000 == 0:
                print("Iter: ", i)
    return  multi_ind_to_indices(ns, tensor_shape)         

def generate_data(coo_tensor, vals, multi_inx_set, shape, how_many, seed):
    ns = give_ns(multi_inx_set, shape, how_many, seed, show_iter=False)
    all_coords = np.concatenate((coo_tensor, ns), axis=0)
    all_vals = np.zeros((how_many * len(multi_inx_set) + vals.size, ))
    all_vals[:vals.size] = vals
    return all_coords, all_vals