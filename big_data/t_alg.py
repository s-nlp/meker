import numpy as np
import pandas as pd
import torch

def mttcrp1(coo, val, shape, mode, a, b, device):
    """
        coo is a list of triples
        Calculate matricized-tensor times Khatri-Rao product. 
    """
    temp = torch.zeros((len(coo), a.shape[1]), device = device)
    # print (temp.shape)
    
    if mode == 0:
        mode_a = 1 
        mode_b = 2
        
    elif mode == 1:
        mode_a = 0
        mode_b = 2
        
    else:
        mode_a = 0
        mode_b = 1
    
    # print (a[coo[0,mode_a], :].shape, b[coo[0,mode_b], :].shape, val[0].shape)
    # print (temp[0].shape)
    for i in range(len(val)):
        # d = a[coo[i,mode_a], :] * b[coo[i,mode_b], :] * val[i]
        # print ("d shape ", d.shape)
        temp[i, :] += a[coo[i,mode_a], :] * b[coo[i,mode_b], :] * val[i]
    # print ("temp.shape", temp.shape)
    return temp

def mttcrp(coo, val, shape, mode, a, b):
    """
        Calculate matricized-tensor times Khatri-Rao product. 
    """
    temp = np.zeros((a.shape[1],))
    
    if mode == 0:
        mode_a = 1 
        mode_b = 2
        
    elif mode == 1:
        mode_a = 0
        mode_b = 2
        
    else:
        mode_a = 0
        mode_b = 1
        
    temp += a[coo[mode_a], :] * b[coo[mode_b], :] * val 
    
    return temp

def get_elem_deriv_tensor(vals, kruskal_vals, loss_function_grad):
    """
        Calculate the elementwise derivative tensor Y.
    """
    #deriv_tensor_vals = loss_function_grad(vals, kruskal_vals) / vals.size
    deriv_tensor_vals = loss_function_grad(vals, kruskal_vals)
    return deriv_tensor_vals    

def factors_to_tensor(coo_tensor, vals, a, b, c):
    """
        Calculate Kruskal tensor values with
        the same coordinates as initial tensor has.
    """
    
    krus_vals = np.zeros_like(vals)
    for item in range(coo_tensor.shape[0]):
        coord = coo_tensor[item]
        krus_vals[item] = np.sum(
            a[coord[0], :] * b[coord[1], :] * c[coord[2], :]
        )
    return krus_vals    

def gcp_grad(coo, val, shape, a, b, c, l2, loss_function, loss_function_grad):
    """
        GCP loss function and gradient calculation.
        All the tensors have the same coordinate set: coo_tensor.
    """
    
    # Construct sparse kruskal tensor
    kruskal_val = np.sum(
        a[coo[0], :] * b[coo[1], :] * c[coo[2], :]
    )#factors_to_tensor(coo_tensor, vals, a, b, c)
    
    #if (val >=0):
        #loss_warp, grad_warp = count_warp(triplets)
    
    # Calculate mean loss on known entries
    loss = loss_function(val, kruskal_val)
    
    # Compute the elementwise derivative tensor
    deriv_tensor_val = loss_function_grad(val, kruskal_val)
    
    # Calculate gradients w.r.t. a, b, c factor matrices
    g_a = mttcrp(coo, deriv_tensor_val, shape, 0, b, c)
    g_b = mttcrp(coo, deriv_tensor_val, shape, 1, a, c)
    g_c = mttcrp(coo, deriv_tensor_val, shape, 2, a, b)
    
    # Add L2 regularization
    if l2 != 0:
        g_a += l2 * a[coo[0], :]
        g_b += l2 * b[coo[1], :]
        g_c += l2 * c[coo[2], :]
    
    return loss, g_a, g_b, g_c

def multi_ind_to_indices(multi_indices, shape):
    coords = np.zeros(shape=(multi_indices.shape[0], 3), dtype=np.int64)
    coords[:, 2] = multi_indices % shape[2]
    i1 = multi_indices // shape[2]
    coords[:, 0] = i1 // shape[1]
    coords[:, 1] = i1 % shape[1]
    return coords

def indices_to_multi_ind(coords, shape):
    multi_indices = (coords[:, 2] + (shape[2] * coords[:, 1])
        + (shape[1] * shape[2] * coords[:, 0]))
    return multi_indices