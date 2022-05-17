import numpy as np
import torch
from torch.nn.init import xavier_normal_

from t_alg import mttcrp, mttcrp1, get_elem_deriv_tensor, factors_to_tensor, gcp_grad, multi_ind_to_indices, indices_to_multi_ind
from elementwise_grads import bernoulli_logit_loss, bernoulli_logit_loss_grad

#from evaluation_functions import create_filter, hr
from general_functions1 import hr

from sklearn.preprocessing import normalize


class MEKER(torch.nn.Module):
    def __init__(self, rank, shape, given_loss, given_loss_grad, device, l2 = 0, **kwargs):
        super(MEKER, self).__init__()
        self.loss_function = given_loss
        self.loss_function_grad = given_loss_grad
        self.l2 = 0
        self.a_torch = torch.empty((shape[0], rank), requires_grad = True, device = device)
        self.b_torch = torch.empty((shape[1], rank), requires_grad = True, device = device)
        self.device = device
        self.shape = shape
        self.err_list = []

    def init(self):
        print ("init")
        xavier_normal_(self.a_torch)
        self.a_torch.grad = torch.zeros(self.a_torch.shape, device = self.device)

        xavier_normal_(self.b_torch)
        self.b_torch.grad = torch.zeros(self.b_torch.shape, device = self.device)
        
    def load_from_numpy(self, path_to_load_a = '/notebook/Relations_Learning/gpu/gpu_a.npz.npy', path_to_load_b = '/notebook/Relations_Learning/gpu/gpu_b.npz.npy'):
        # load pre-trained matrix
        a = np.load(path_to_load_a)
        b = np.load(path_to_load_b)
        self.a_torch = torch.tensor(a, requires_grad = True, device = self.device)
        self.a_torch.grad = torch.zeros(self.a_torch.shape, device = self.device)
        self.b_torch = torch.tensor(b, requires_grad = True, device = self.device)
        self.b_torch.grad = torch.zeros(self.b_torch.shape, device = self.device)
        self.err_list = []

        
    def gcp_grad(self, coo, val, shape, a, b, l2, loss_function, loss_function_grad, device):
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

        #print ("\n\n")

        # Add L2 regularization
        if l2 != 0:
            g_a += l2 * a[coo[:, 0], :]
            g_b += l2 * b[coo[:, 1], :]
            g_c += l2 * a[coo[:, 2], :]

        return loss, g_a, g_b, g_c

    def forward(self, coo_ns, vals_ns, a_elems, b_elems, c_elems):
        loss, g_a, g_b, g_c = self.gcp_grad(
            coo_ns, vals_ns, self.shape, self.a_torch, self.b_torch,
            self.l2, self.loss_function, self.loss_function_grad, self.device
        )

        self.err_list.append(loss.cpu().detach().numpy().mean())

        self.a_torch.grad[a_elems, :] = g_a
        self.b_torch.grad[b_elems, :] = g_b
        self.a_torch.grad[c_elems, :] = g_c
        return 0
    
    def evaluate(self, datas):
        a = self.a_torch.cpu().data.numpy()
        b = self.b_torch.cpu().data.numpy()
        c = self.a_torch.cpu().data.numpy()
        
        a_norm = normalize(a, axis=1)
        b_norm = normalize(b, axis=1)
        c_norm = normalize(c, axis=1)
        
        print ("count hr", flush = True)
        hit1, hit3, hit10, mrr = hr(datas.valid_filters, datas.valid_triples, a_norm, b_norm, c_norm, [1, 3, 10])
        return (hit1, hit3, hit10, mrr)
    
    def get_test(self, datas):
        a = self.a_torch.cpu().data.numpy()
        b = self.b_torch.cpu().data.numpy()
        c = self.a_torch.cpu().data.numpy()
        
        a_norm = normalize(a, axis=1)
        b_norm = normalize(b, axis=1)
        c_norm = normalize(c, axis=1)
        
        print ("count hr", flush = True)
        hit1, hit3, hit10, mrr = hr(datas.test_filters, datas.test_triples, a_norm, b_norm, c_norm, [1, 3, 10])
        return (hit1, hit3, hit10, mrr)