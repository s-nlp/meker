import numpy as np
import pandas as pd

import torch

from t_alg import mttcrp, mttcrp1, get_elem_deriv_tensor, factors_to_tensor, gcp_grad, multi_ind_to_indices, indices_to_multi_ind


def gcp_grad(coo, val, shape, a, b, l2, loss_function, loss_function_grad, device):
    """
        GCP loss function and gradient calculation.
        All the tensors have the same coordinate set: coo_tensor.
    """

    # Construct sparse kruskal tensor
    kruskal_val = torch.sum((a[coo[:,0], :] * b[coo[:,1], :] * a[coo[:,2], :]),1)
    #factors_to_tensor(coo_tensor, vals, a, b, c)
    
    # Calculate mean loss on known entries
    loss = loss_function(val, kruskal_val)
    # Compute the elementwise derivative tensor
    deriv_tensor_val = loss_function_grad(val, kruskal_val)
    
    #print ("in qcp_grad in deriv_tensor_val ", deriv_tensor_val)
    # Calculate gradients w.r.t. a, b, c factor matrices
    g_a = mttcrp1(coo, deriv_tensor_val, shape, 0, b, a, device)
    g_b = mttcrp1(coo, deriv_tensor_val, shape, 1, a, a, device)
    g_c = mttcrp1(coo, deriv_tensor_val, shape, 2, a, b, device)
    
    #print ("\n\n")
    
    
    # Add L2 regularization
    if l2 != 0:
        g_a += l2 * a[coo[0], :]
        g_b += l2 * b[coo[1], :]
        g_c += l2 * c[coo[2], :]
    
    return loss, g_a, g_b, g_c
