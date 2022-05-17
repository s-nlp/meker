import numpy as np
from collections import defaultdict

def create_filter(all_triples):
    """ Filter all true objects for a particular test triple """
    ft = defaultdict(list)
    sz = len(all_triples)
    for i in range(sz):
        ft[all_triples[i][:2]].append(all_triples[i][2]) 
    return ft



def hr(ftr, test_triples, a, b, c,
       how_many=[1, 3, 10], iter_show=False, freq=1000):
    """ Calculate HR@[how_many] """
    
    total = len(test_triples)
    hit = np.zeros(len(how_many), dtype=np.float64)
    
    p_all = [test_triples[m][0] for m in range(total)]
    q_all = [test_triples[m][1] for m in range(total)]
    
    temp = (a[p_all, :] * b[q_all, :])
    candidate_values = np.dot(temp, c.T)
    
    iteration = 0
    mrr = 0.0
    for j, entity in enumerate(test_triples):
        p, q, r = entity
        
        tmp = candidate_values[j][r]
    
        candidate_values[j][ftr[j]] = 0.0
        #candidate_values[j][ftr[p][q]] = 0.0
        candidate_values[j][r] = tmp
        
        top = np.argpartition(candidate_values[j], -how_many[-1])[-how_many[-1]:]
        top = top[np.argsort(candidate_values[j][top])][::-1]
        
        for i, h in enumerate(how_many):
            if r in top[:h]:
                hit[i] += 1.0  
        
        iteration += 1
        if iter_show:
            if iteration % freq == 0:
                print(f"Iter: {iteration}\n")
    hit = hit / total  
    return hit