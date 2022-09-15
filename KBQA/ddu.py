###
# The implementation is strongly based on Jishnu Mukhoti (DDU author) own code
# See https://github.com/omegafragger/DDU for the reference (MIT licencse)
###
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import sklearn
from sklearn.mixture import BayesianGaussianMixture

DOUBLE_INFO = torch.finfo(torch.double)
# JITTERS = [0, DOUBLE_INFO.tiny, 1e-32, 1e-24, 1e-20] + [10 ** exp for exp in range(-15, 0, 1)]
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-8, 0, 1)]


def entropy(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=1)
    return entropy


def logsumexp(logits):
    return torch.logsumexp(logits, dim=1, keepdim=False)


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                #out = net.module.feature
            else:
                out = net(data)
                #out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):
    if isinstance(net, nn.DataParallel):
        features_B_Z = net.module(data_B_X)
        #features_B_Z = net.module.feature
    else:
        features_B_Z = net(data_B_X)
        #features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :].float().cpu())

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)


def gmm_evaluate_embeddings(gaussians_model, loader, device, storage_device):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, 100), dtype=torch.float, device=storage_device)

    with torch.no_grad():
        start = 0
        for data in tqdm(loader):
            data = data.to(device)

            logits = []
            for emb in data:
                logits.append(gaussians_model.log_prob(emb.float().cpu()))

            logits = torch.stack(logits, dim=0)

            end = start + len(data)
            logits_N_C[start:end].copy_(logits, non_blocking=True)
            start = end

    return logits_N_C


def variational_gmm_evaluate_embeddings(gaussians_model, data):
    return gaussians_model.score_samples(data)


def gmm_fit(embeddings, labels, num_classes):
    gmm, jitter_eps = None, None
    with torch.no_grad():
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]).float()

        width = embeddings.shape[-1]
        classwise_cov_features = torch.zeros((num_classes, width, width), dtype=torch.float32)

        for c in tqdm(range(num_classes)):
            classwise_cov_features[c] = centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c])

    with torch.no_grad():
        for jitter_eps in JITTERS:
            print('jitter eps', jitter_eps)
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1], device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                print(e)
                continue
            break

    return gmm, jitter_eps


def variational_gmm_fit(embeddings, num_classes, random_state):
    with torch.no_grad():
        gmm = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior = 1e5,
            n_components=num_classes,
            reg_covar=0,
            init_params="random",
            max_iter=1500,
            mean_precision_prior=0.8,
            random_state=random_state,
        )
        gmm = gmm.fit(embeddings)
    return gmm
