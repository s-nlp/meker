import numpy as np

def create_filter(test_triple, all_triples, show=False, shift=100):
    """ Filter all true objects for a particular test triple """
    
    filt = []
    it = 0
    for i in test_triple:
        filt_set = []
        for j in all_triples:
            if (i[0] == j[0]) and (i[1] == j[1]) and (i[2] != j[2]):
                filt_set.append(j[2]) 
           
        filt.append(filt_set)
        
        it += 1
        if show:
            if it % shift == 0:
                print(it)
                
    
    return filt   

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hr(test_filter, test_triples, a, b, c,
       how_many=[1, 3, 10], iter_show=False, freq=3000):
    """ Calculate HR@[how_many] and MRR using filter """
    
    total = len(test_triples)
    hit = [0, 0, 0, 0]
    iteration = 0
    for entity, filt in zip(test_triples, test_filter):
        p = entity[0]
        q = entity[1]
        r = entity[2]

        candidate_values = np.sum(a[p, :] * b[q, :] * c, axis=1)
        candidate_values = sigmoid(candidate_values)
        
        top = (np.argsort(candidate_values)[::-1]).tolist()   
        
        for obj in filt:
            top.remove(obj)
        
        ind = top.index(r)
        for i, h in enumerate(how_many):
            if ind < h:
                hit[i] += 1
        hit[3] += 1 / (1 + ind)    
        
        iteration += 1
        if iter_show:
            if iteration % freq == 0:
                print(hit[2] / iteration, hit[2], iteration)
            
    return hit[0] / total, hit[1] / total, hit[2] / total, hit[3] / total

def gen_coo_tensor(shape, density=0.02):
    nnz = int(density * shape[0] * shape[1] * shape[2])
    m = np.random.choice(shape[0], nnz)
    n = np.random.choice(shape[1], nnz)
    k = np.random.choice(shape[2], nnz)
    vals = np.random.rand(nnz)
    return np.vstack((m, n, k)).T, vals

def check_coo_tensor(coo):
    count = 0
    for i in range(coo.shape[0]):
        for j in range(coo.shape[0]):
            if (coo[i]==coo[j]).sum() == 3:
                count += 1
                if count > 1:
                    return "Bad"
        count = 0  

def gen_hilbert_tensor(shape):
    coo = []
    vals = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                coo.append((i, j, k))
                vals.append(1 / (i + j + k + 3))
    
    coo = np.array(coo)
    vals = np.array(vals)
    return coo, vals     


#@jit(nopython=True) 
def sqrt_err(coo_tensor, vals, shape, a, b, c):
    result = 0.0
    for item in range(coo_tensor.shape[0]):
        coord = coo_tensor[item]
        result += (vals[item] - np.sum(
            a[coord[0], :] * b[coord[1], :] * c[coord[2], :]))**2        
    return np.sqrt(result)


#@jit(nopython=True) 
def sqrt_err_relative(coo_tensor, vals, shape, a, b, c):
    result = sqrt_err(coo_tensor, vals, shape, a, b, c)        
    return result / np.sqrt((vals**2).sum())
