import torch
import numpy as np
from models import get_tokenizer
from torch import nn
from tqdm import tqdm

from IPython.core.debugger import set_trace
    
MAX_LEN_Q = 32
    
def forward_pass_model(model, X, device):
    encoder = model['encoder']
    projection_E = model['projection_E']
    projection_P = model['projection_P']
    projection_Q = model['projection_Q']

    encoder.eval()
    projection_E.eval()
    projection_P.eval()
    projection_Q.eval()

    encoded_X = encoder(X)
    
    if 'dropout_mask' in model:
        encoded_X = nn.Dropout(p=0.6).train()(encoded_X)
    
    y_pred_e = projection_E(encoded_X).detach().cpu()
    y_pred_q = projection_Q(encoded_X).detach().cpu()
    y_pred_p = projection_P(encoded_X).detach().cpu()
    
    return y_pred_e, y_pred_q, y_pred_p


def get_top_ids_second_hop(models, device, tokenizer, text, first_hop_graph_E, second_hop_graph_Q, second_hop_graph_P, second_hop_ids_filtered_Q, second_hop_ids_filtered_P, topk, temperature=1.0, ensembling_mode='mean', weights=None):
        cosines_P = []
        cosines_E = []
        cosines_Q = []
        unnormalized_cosines = []
        embeddings_e = []
        embeddings_p = []
        embeddings_q = []
        embeddings_e_norm = []
        embeddings_p_norm = []
        embeddings_q_norm = []
        
        softmax = nn.Softmax(dim=0)
        X = torch.tensor([tokenizer.encode(text, max_length=MAX_LEN_Q, add_special_tokens=True,pad_to_max_length=True)]).to(device)[0].to(device)[None,:]

        for model in models:   
            y_pred_e, y_pred_q, y_pred_p = forward_pass_model(model, X, device)
            
            embeddings_e_norm.append(y_pred_e.numpy() / np.linalg.norm(y_pred_e.numpy()))
            embeddings_q_norm.append(y_pred_q.numpy() / np.linalg.norm(y_pred_p.numpy()))
            embeddings_p_norm.append(y_pred_p.numpy() / np.linalg.norm(y_pred_q.numpy()))
            
            embeddings_e.append(y_pred_e.numpy())
            embeddings_q.append(y_pred_q.numpy())
            embeddings_p.append(y_pred_p.numpy())

            embeddings_tensor_E = torch.FloatTensor(first_hop_graph_E)
            embeddings_tensor_Q = torch.FloatTensor(second_hop_graph_Q)
            embeddings_tensor_P = torch.FloatTensor(second_hop_graph_P)

            cosines_descr_E = torch.cosine_similarity(embeddings_tensor_E.cpu(),y_pred_e.cpu())
            cosines_E.append(cosines_descr_E.numpy())
            # norm_cosines_descr_E = nn.Softmax()(cosines_descr_E).numpy()

            cosines_descr_Q = torch.cosine_similarity(embeddings_tensor_Q.cpu(),y_pred_q.cpu())
            cosines_Q.append(cosines_descr_Q.numpy())
            # norm_cosines_descr_Q = nn.Softmax()(cosines_descr_Q).numpy()

            cosines_descr_P = torch.cosine_similarity(embeddings_tensor_P.cpu(),y_pred_p.cpu())
            cosines_P.append(cosines_descr_P.numpy())
            # norm_cosines_descr_P = nn.Softmax()(cosines_descr_P).numpy()

            cosines_aggr_all = np.array(cosines_descr_P + cosines_descr_Q + cosines_descr_E)

            unnormalized_cosines.append(cosines_aggr_all)
            
        unnormalized_cosines = np.stack(unnormalized_cosines)
            
        e_std_norm = np.array(embeddings_e_norm).squeeze(axis=1).std(axis=0).mean()
        q_std_norm = np.array(embeddings_q_norm).squeeze(axis=1).std(axis=0).mean()
        p_std_norm = np.array(embeddings_p_norm).squeeze(axis=1).std(axis=0).mean()
        
        e_std = np.array(embeddings_e).squeeze(axis=1).std(axis=0).mean()
        q_std = np.array(embeddings_q).squeeze(axis=1).std(axis=0).mean()
        p_std = np.array(embeddings_p).squeeze(axis=1).std(axis=0).mean()
        
        if ensembling_mode == 'average':
            unnormalized_cosines_mean = np.average(unnormalized_cosines, axis=0, weights=weights)
        else:
            unnormalized_cosines_mean = np.mean(unnormalized_cosines, axis=0)

        unnormalized_cosines_std = np.std(unnormalized_cosines, axis=0)
        
        cosines_P_std = np.std(np.array(cosines_P), axis=0)
        cosines_Q_std = np.std(np.array(cosines_Q), axis=0)
        cosines_E_std = np.std(np.array(cosines_E), axis=0)
        
        topk_inds = lambda cosines: torch.topk(torch.tensor(cosines),topk,sorted=True).indices.cpu().numpy()
    
        ensemble_inds = topk_inds(unnormalized_cosines_mean)
        ensemble_cosines = unnormalized_cosines_mean[ensemble_inds]
        ensemble_probas = np.array(softmax(torch.tensor(cosines_aggr_all) / temperature))
        ensemble_predicts = np.array(second_hop_ids_filtered_Q)[ensemble_inds]
        
        models_inds = [topk_inds(cosines) for cosines in unnormalized_cosines]
        models_predicts = [np.array(second_hop_ids_filtered_Q)[inds] for inds in models_inds]
        models_cosines = np.array([unnormalized_cosines[i][inds] for i, inds in enumerate(models_inds)])
        models_cosines_std = unnormalized_cosines_std[ensemble_inds].mean()
        models_probas = np.array([np.array(softmax(torch.tensor(model_cosines) / temperature)) for model_cosines in models_cosines])
        models_probas_std = np.std(models_probas, axis=0).mean()
        
        if ensembling_mode == 'majority':
            ensemble_predicts = [max(set(lst), key=list(lst).count) for lst in np.transpose(models_predicts)]
            
        return (ensemble_predicts,
                models_probas, ensemble_probas, models_probas_std, 
                models_cosines, ensemble_cosines, models_cosines_std,
                e_std, q_std, p_std, 
                e_std_norm, q_std_norm, p_std_norm,
                cosines_P_std[ensemble_inds][0], cosines_Q_std[ensemble_inds][0], cosines_E_std[ensemble_inds][0],
                models_predicts)


def eval_ensemble(questions_test, answers_test, graph_embeddings_P, graph_embeddings_Q, candidates, models, device, entropy_limit=None, temperature=1.0, ensembling_mode='mean', weights=None):
    q_list = []
    a_list = []
    a_predicts = []
    a_model_predicts = []
    inv_ranks = []
    top1_scores = []
    top2_scores = []
    e_stds = []
    q_stds = []
    p_stds = []
    e_stds_norm = []
    q_stds_norm = []
    p_stds_norm = []
    cosines_stds = []
    cosine_P_stds = []
    cosine_Q_stds = []
    cosine_E_stds = []
    entropies_of_mean = []
    mean_entropies = []
    all_cosines = []
    
    tokenizer = get_tokenizer()

    bad_question_ids = []
    for i, (q, a, second_hop_ids_QP) in tqdm(enumerate(zip(questions_test, answers_test, candidates))):
        
        if len(second_hop_ids_QP) > 0:
            first_hop_graph_E = []
            second_hop_graph_Q = []
            second_hop_ids_filtered_Q = []
            second_hop_graph_P = []
            second_hop_ids_filtered_P = []
            for key in second_hop_ids_QP.keys():
                for (idd_q, idd_p) in second_hop_ids_QP[key]:
                    if idd_q in graph_embeddings_Q and idd_p in graph_embeddings_P and key in graph_embeddings_Q:
                        first_hop_graph_E.append(graph_embeddings_Q[key])
                        second_hop_ids_filtered_Q.append(idd_q)
                        second_hop_graph_Q.append(graph_embeddings_Q[idd_q])
                        second_hop_ids_filtered_P.append(idd_p)
                        second_hop_graph_P.append(graph_embeddings_P[idd_p])

            if len(first_hop_graph_E) > 0:
                (predicts, 
                 models_probas, ensemble_probas, models_probas_std,
                 models_cosines, ensemble_cosines, models_cosines_std,
                 std_e, std_q, std_p, 
                 std_e_norm, std_q_norm, std_p_norm, 
                 cosines_P_std, cosines_Q_std, cosines_E_std, model_predicts) = get_top_ids_second_hop(models, 
                                                                                                       device, 
                                                                                                       tokenizer, 
                                                                                                       q, 
                                                                                                       first_hop_graph_E, 
                                                                                                       second_hop_graph_Q, 
                                                                                                       second_hop_graph_P, 
                                                                                                       second_hop_ids_filtered_Q, 
                                                                                                       second_hop_ids_filtered_P, 
                                                                                                       len(second_hop_graph_Q),
                                                                                                       temperature=temperature,
                                                                                                       ensembling_mode=ensembling_mode,
                                                                                                       weights=weights)

                e_stds.append(std_e)
                q_stds.append(std_q)
                p_stds.append(std_p)
                e_stds_norm.append(std_e_norm)
                q_stds_norm.append(std_q_norm)
                p_stds_norm.append(std_p_norm)
                cosines_stds.append(models_cosines_std)
                cosine_P_stds.append(cosines_P_std)
                cosine_Q_stds.append(cosines_Q_std)
                cosine_E_stds.append(cosines_E_std)
                all_cosines.append(models_cosines)
                
                if len(ensemble_probas) > 1:
                    top2 = torch.topk(torch.tensor(ensemble_probas), 2).values
                    top1_scores.append(top2[0].item())
                    top2_scores.append(top2[1].item())
                    
                    if entropy_limit is None or entropy_limit > len(ensemble_probas):
                        entropies_of_mean.append(-np.sum([p * np.log(p) for p in ensemble_probas]))
                        mean_entropies.append(np.mean([-np.sum([p * np.log(p) for p in distribution]) for distribution in models_probas]))
                    else:
                        limited_ensemble_probas = nn.Softmax(dim=0)(torch.tensor(ensemble_cosines[0:entropy_limit] / temperature))
                        limited_models_probas = [nn.Softmax(dim=0)(torch.tensor(model_cosines[0:entropy_limit] / temperature)) for model_cosines in models_cosines]

                        entropies_of_mean.append(-np.sum([p * np.log(p) for p in limited_ensemble_probas]))
                        entropies = []
                        for distribution in limited_models_probas:
                            entropies.append(-np.sum([p * np.log(p) for p in distribution]))
                        mean_entropies.append(np.mean(entropies))
                        
#                         first_preds = [pred[0] for pred in model_predicts]
#                         num_opinions = len(set(first_preds))
#                         if num_opinions > 1:
#                             set_trace()
#                             pass

                else:
                    top1_scores.append(ensemble_probas[0])
                    top2_scores.append(0)
                    
                    entropies_of_mean.append(0)
                    mean_entropies.append(0)
            else:
                bad_question_ids.append(i)
                all_cosines.append([])
                predicts = []
                model_predicts = []
                top1_scores.append(-1)
                top2_scores.append(-1)

                e_stds.append(np.inf)
                q_stds.append(np.inf)
                p_stds.append(np.inf)
                e_stds_norm.append(np.inf)
                q_stds_norm.append(np.inf)
                p_stds_norm.append(np.inf)
                cosines_stds.append(np.inf)
                cosine_P_stds.append(np.inf)
                cosine_Q_stds.append(np.inf)
                cosine_E_stds.append(np.inf)
                entropies_of_mean.append(1)
                mean_entropies.append(1)
        else:
            bad_question_ids.append(i)
            all_cosines.append([])
            predicts = []
            model_predicts = []
            top1_scores.append(-1)
            top2_scores.append(-1)
            
            e_stds.append(np.inf)
            q_stds.append(np.inf)
            p_stds.append(np.inf)
            e_stds_norm.append(np.inf)
            q_stds_norm.append(np.inf)
            p_stds_norm.append(np.inf)
            cosines_stds.append(np.inf)
            cosine_P_stds.append(np.inf)
            cosine_Q_stds.append(np.inf)
            cosine_E_stds.append(np.inf)
            entropies_of_mean.append(1)
            mean_entropies.append(1)

        a_predicts.append(predicts)
        a_model_predicts.append(model_predicts)

        inv_rs = []
        if len(predicts) > 0:
            for a_i in a:
                if a_i not in predicts:
                    inv_r = 0
                else:
                    inv_r = 1 / (list(predicts).index(a_i) + 1)
                inv_rs.append(inv_r)
            inv_ranks.append(max(inv_rs))
        else:
            inv_ranks.append(0)

    top1 = np.array(inv_ranks)[np.array(inv_ranks) == 1]
    accuracy = len(top1) / len(inv_ranks)
    print("Accuracy: ", accuracy)
        
    return q_list, a_list, a_predicts, inv_ranks, top1_scores, top2_scores, e_stds, q_stds, p_stds, e_stds_norm, q_stds_norm, p_stds_norm, cosines_stds, entropies_of_mean, mean_entropies, accuracy, cosine_P_stds, cosine_Q_stds, cosine_E_stds, bad_question_ids, all_cosines, a_model_predicts


def eval_single_model(questions_test, answers_test, graph_embeddings_P, graph_embeddings_Q, candidates, model, device, entropy_limit=None, temperature=1.0):
    q_list = []
    a_list = []
    a_predicts = []
    inv_ranks = []
    top1_scores = []
    top2_scores = []
    entropies = []
    
    tokenizer = get_tokenizer()

    bad_question_ids = []
    for i, (q, a, second_hop_ids_QP) in tqdm(enumerate(zip(questions_test, answers_test, candidates))):
        if len(second_hop_ids_QP) > 0:
            first_hop_graph_E = []
            second_hop_graph_Q = []
            second_hop_ids_filtered_Q = []
            second_hop_graph_P = []
            second_hop_ids_filtered_P = []
            for key in second_hop_ids_QP.keys():
                for (idd_q, idd_p) in second_hop_ids_QP[key]:
                    if idd_q in graph_embeddings_Q and idd_p in graph_embeddings_P and key in graph_embeddings_Q:
                        first_hop_graph_E.append(graph_embeddings_Q[key])
                        second_hop_ids_filtered_Q.append(idd_q)
                        second_hop_graph_Q.append(graph_embeddings_Q[idd_q])
                        second_hop_ids_filtered_P.append(idd_p)
                        second_hop_graph_P.append(graph_embeddings_P[idd_p])

            if len(first_hop_graph_E) > 0:
                (predicts, 
                 _, ensemble_probas, _,
                 models_cosines, _, _,
                 _, _, _, 
                 _, _, _, 
                 _, _, _, _) = get_top_ids_second_hop((model,), 
                                                       device, 
                                                       tokenizer, 
                                                       q, 
                                                       first_hop_graph_E, 
                                                       second_hop_graph_Q, 
                                                       second_hop_graph_P, 
                                                       second_hop_ids_filtered_Q, 
                                                       second_hop_ids_filtered_P,
                                                       len(second_hop_graph_Q),
                                                       temperature=temperature)

                if len(ensemble_probas) > 1:
                    top2 = torch.topk(torch.tensor(ensemble_probas), 2).values
                    
                    top1_scores.append(top2[0].item())
                    top2_scores.append(top2[1].item())
                    
                    if entropy_limit is None or entropy_limit > len(ensemble_probas):
                        normalized_scores = ensemble_probas
                    else:
                        normalized_scores = nn.Softmax()(torch.tensor(models_cosines[0][0:entropy_limit] / temperature))
                        
                    entropies.append(-np.array(normalized_scores * np.log(normalized_scores)).sum())
                else:
                    top1_scores.append(ensemble_probas[0])
                    top2_scores.append(0)
                    entropies.append(0)
            else:
                bad_question_ids.append(i)
                predicts = []
                top1_scores.append(-1)
                top2_scores.append(-1)
                entropies.append(1)
        else:
            bad_question_ids.append(i)
            predicts = []
            top1_scores.append(-1)
            top2_scores.append(-1)
            entropies.append(1)

        a_predicts.append(predicts)

        inv_rs = []
        if len(predicts) > 0:
            for a_i in a:
                if a_i not in predicts:
                    inv_r = 0
                else:
                    inv_r = 1 / (list(predicts).index(a_i) + 1)
                inv_rs.append(inv_r)
            inv_ranks.append(max(inv_rs))
        else:
            inv_ranks.append(0)

    top1 = np.array(inv_ranks)[np.array(inv_ranks) == 1]
    accuracy = len(top1) / len(inv_ranks)
    print("Accuracy: ", accuracy)
        
    return q_list, a_list, a_predicts, inv_ranks, top1_scores, top2_scores, entropies, accuracy, bad_question_ids


def get_top_ids_second_hop_simple(models_embs, device, tokenizer, text, first_hop_graph_E, second_hop_graph_Q, second_hop_graph_P, second_hop_ids_filtered_Q, second_hop_ids_filtered_P, topk, ensembling_mode='mean', weights=None):
        unnormalized_cosines = []
            
        for y_pred_e, y_pred_q, y_pred_p in models_embs:            
            embeddings_tensor_E = torch.FloatTensor(first_hop_graph_E)
            embeddings_tensor_Q = torch.FloatTensor(second_hop_graph_Q)
            embeddings_tensor_P = torch.FloatTensor(second_hop_graph_P)

            cosines_descr_E = torch.cosine_similarity(embeddings_tensor_E.cpu(),y_pred_e.cpu())

            cosines_descr_Q = torch.cosine_similarity(embeddings_tensor_Q.cpu(),y_pred_q.cpu())

            cosines_descr_P = torch.cosine_similarity(embeddings_tensor_P.cpu(),y_pred_p.cpu())

            cosines_aggr_all = np.array(cosines_descr_P + cosines_descr_Q + cosines_descr_E)

            unnormalized_cosines.append(cosines_aggr_all)
            
        unnormalized_cosines = np.stack(unnormalized_cosines)
                    
        if ensembling_mode == 'average':
            unnormalized_cosines_mean = np.average(unnormalized_cosines, axis=0, weights=weights)
        else:
            unnormalized_cosines_mean = np.mean(unnormalized_cosines, axis=0)
        
        topk_inds = lambda cosines: torch.topk(torch.tensor(cosines),topk,sorted=True).indices.cpu().numpy()
    
        ensemble_inds = topk_inds(unnormalized_cosines_mean)
        ensemble_predicts = np.array(second_hop_ids_filtered_Q)[ensemble_inds]
                            
        return ensemble_predicts
    
def eval_ensemble_simple(questions_test, answers_test, graph_embeddings_P, graph_embeddings_Q, candidates, models_embs, device, entropy_limit=None, temperature=1.0, ensembling_mode='mean', weights=None):
    a_predicts = []
    a_model_predicts = []
    inv_ranks = []
    bad_question_ids = []

    tokenizer = get_tokenizer()

    for i, (q, a, second_hop_ids_QP) in tqdm(enumerate(zip(questions_test, answers_test, candidates))):
        
        X = torch.tensor([tokenizer.encode(q, max_length=MAX_LEN_Q, add_special_tokens=True,pad_to_max_length=True)]).to(device)[0].to(device)[None,:]

        if len(second_hop_ids_QP) > 0:
            first_hop_graph_E = []
            second_hop_graph_Q = []
            second_hop_ids_filtered_Q = []
            second_hop_graph_P = []
            second_hop_ids_filtered_P = []
            for key in second_hop_ids_QP.keys():
                for (idd_q, idd_p) in second_hop_ids_QP[key]:
                    if idd_q in graph_embeddings_Q and idd_p in graph_embeddings_P and key in graph_embeddings_Q:
                        first_hop_graph_E.append(graph_embeddings_Q[key])
                        second_hop_ids_filtered_Q.append(idd_q)
                        second_hop_graph_Q.append(graph_embeddings_Q[idd_q])
                        second_hop_ids_filtered_P.append(idd_p)
                        second_hop_graph_P.append(graph_embeddings_P[idd_p])

            if len(first_hop_graph_E) > 0:
                predicts = get_top_ids_second_hop_simple(models_embs[i], 
                                                         device, 
                                                         tokenizer, 
                                                         X, 
                                                         first_hop_graph_E, 
                                                         second_hop_graph_Q, 
                                                         second_hop_graph_P, 
                                                         second_hop_ids_filtered_Q, 
                                                         second_hop_ids_filtered_P, 
                                                         len(second_hop_graph_Q),
                                                         ensembling_mode=ensembling_mode,
                                                         weights=weights)

            else:
                bad_question_ids.append(i)
                predicts = []
        else:
            bad_question_ids.append(i)
            predicts = []

        a_predicts.append(predicts)

        inv_rs = []
        if len(predicts) > 0:
            for a_i in a:
                if a_i not in predicts:
                    inv_r = 0
                else:
                    inv_r = 1 / (list(predicts).index(a_i) + 1)
                inv_rs.append(inv_r)
            inv_ranks.append(max(inv_rs))
        else:
            inv_ranks.append(0)

    top1 = np.array(inv_ranks)[np.array(inv_ranks) == 1]
    accuracy = len(top1) / len(inv_ranks)
    print("Accuracy: ", accuracy)
        
    return accuracy
