import numpy as np
import pandas as pd
import torch


#@jit(nopython=True)
def gaussian_loss(x_vals, m_vals):
    return (x_vals - m_vals)**2

#@jit(nopython=True)
def gaussian_loss_grad(x_vals, m_vals):
    return -2 * (x_vals - m_vals)

#@jit(nopython=True)
def bernoulli_loss(x_vals, m_vals):
    eps = 1e-8
    return np.log(1 + m_vals) - x_vals*np.log(eps + m_vals)

#@jit(nopython=True)
def bernoulli_loss_grad(x_vals, m_vals):
    eps = 1e-8
    return (1 / (1 + m_vals)) - (x_vals / (eps + m_vals))

#@jit(nopython=True)
def bernoulli_logit_loss(x_vals, m_vals):
    return torch.log(1 + torch.exp(m_vals)) - (x_vals * m_vals)

#@jit(nopython=True)
def bernoulli_logit_loss_grad(x_vals, m_vals):
    exp_vals = torch.exp(m_vals)
    return (exp_vals / (1 + exp_vals)) - x_vals

#@jit(nopython=True)
def poisson_loss(x_vals, m_vals):
    eps = 1e-8
    return m_vals - x_vals * np.log(m_vals + eps)

#@jit(nopython=True)
def poisson_loss_grad(x_vals, m_vals):
    eps = 1e-8
    return 1 - (x_vals / (m_vals + eps))

#@jit(nopython=True)
def poisson_log_loss(x_vals, m_vals):
    return np.exp(m_vals) - (x_vals * m_vals)

#@jit(nopython=True)
def poisson_log_loss_grad(x_vals, m_vals):
    return np.exp(m_vals) - x_vals