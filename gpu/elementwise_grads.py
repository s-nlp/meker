import numpy as np
import pandas as pd
import torch

def log1pexp(x):
    # more stable version of log(1 + exp(x))
    return torch.where(x < 50, torch.log1p(torch.exp(x)), x)

def sigm(x):
    exp_vals = torch.exp(x)
    return (exp_vals / (1 + exp_vals))

def stab_sigm(x):
    # more stable version of log(1 + exp(x))
    return torch.where(x < 50, sigm(x), torch.ones(x.shape, device = x.get_device())+ torch.full(x.shape, 1e-8, device = x.get_device()))

def bernoulli_loss(x_vals, m_vals):
    device = x_vals.get_device()
    eps = torch.tensor([[1e-8]], device = device)
    return torch.log(1 + m_vals) - x_vals*torch.log(eps + m_vals)

def bernoulli_loss_grad(x_vals, m_vals):
    device = x_vals.get_device()
    eps = torch.tensor([[1e-8]], device = device)
    return (1 / (1 + m_vals)) - (x_vals / (eps + m_vals))

def bernoulli_logit_loss(x_vals, m_vals):
    return log1pexp(m_vals) - (x_vals * m_vals)

def bernoulli_logit_loss_grad(x_vals, m_vals):
    return stab_sigm(m_vals) - x_vals

def poisson_loss(x_vals, m_vals):
    eps = 1e-8
    return m_vals - x_vals * np.log(m_vals + eps)

def poisson_loss_grad(x_vals, m_vals):
    eps = 1e-8
    return 1 - (x_vals / (m_vals + eps))

def poisson_log_loss(x_vals, m_vals):
    return np.exp(m_vals) - (x_vals * m_vals)

def poisson_log_loss_grad(x_vals, m_vals):
    return np.exp(m_vals) - x_vals