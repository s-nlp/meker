import torch
from torch import nn
import numpy as np
from pathlib import Path
from eval_models import eval_single_model

from models import EncoderBERT, get_projection_module_simple
from eval_models import eval_single_model

from IPython.core.debugger import set_trace

def train_model(proj_hidden_size, train_dataloader, val_dataloader, checkpoint_path, n_epochs, model_name, device, loss, candidates, questions_val, answers_val, graph_embeddings_P, graph_embeddings_Q, model=None):
    if model:
        encoder = model['encoder']

        projection_E = model['projection_E']
        projection_Q = model['projection_Q']
        projection_P = model['projection_P']
    else:
        encoder = EncoderBERT(device)

        projection_E = get_projection_module_simple(device, proj_hidden_size)
        projection_Q = get_projection_module_simple(device, proj_hidden_size)
        projection_P = get_projection_module_simple(device, proj_hidden_size)

    opt = torch.optim.AdamW(
        list(projection_Q.parameters()) + \
        list(projection_P.parameters()) + \
        list(projection_E.parameters()) + \
        list(encoder.parameters()), 
        lr=1e-4
    )

    min_val_loss = float('inf')
    max_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(1, n_epochs + 1):
        batch_train_losses = []
        batch_val_losses = []
        
        for X, y_e, y_q, y_p in train_dataloader:
            projection_Q.train()
            projection_E.train()
            projection_P.train()
            encoder.train()
            
            encoded_X = encoder(X)

            y_pred_e = projection_E(encoded_X)
            y_pred_q = projection_Q(encoded_X)
            y_pred_p = projection_P(encoded_X)

            if isinstance(loss, nn.CosineEmbeddingLoss):
                labels = torch.full((y_q.shape[0], ), 1).to(device)
                train_loss = loss(y_q, y_pred_q, labels) + loss(y_p, y_pred_p, labels) + loss(y_e, y_pred_e, labels)
            elif isinstance(loss, nn.MSELoss):
                train_loss = loss(y_q,y_pred_q) + loss(y_p,y_pred_p) + loss(y_e,y_pred_e)

            train_loss.backward()
            batch_train_losses.append(train_loss.item())

            opt.step()
            opt.zero_grad()

        for X, y_e, y_q, y_p in val_dataloader:
            with torch.no_grad():
                projection_Q.eval()
                projection_E.eval()
                projection_P.eval()
                encoder.eval()

                encoded_X = encoder(X)

                y_pred_e = projection_E(encoded_X)
                y_pred_q = projection_Q(encoded_X)
                y_pred_p = projection_P(encoded_X)

                if isinstance(loss, nn.CosineEmbeddingLoss):
                    labels = torch.full((y_q.shape[0], ), 1).to(device)
                    val_loss = loss(y_q, y_pred_q, labels) + loss(y_p, y_pred_p, labels) + loss(y_e, y_pred_e, labels)
                elif isinstance(loss, nn.MSELoss):
                    val_loss = loss(y_q,y_pred_q) + loss(y_p,y_pred_p) + loss(y_e,y_pred_e)
                
                batch_val_losses.append(val_loss.item())
                
        model = {'encoder': encoder, 'projection_P': projection_P, 'projection_Q': projection_Q, 'projection_E': projection_E}
        _, _, _, _, _, _, _, epoch_val_acc, _ = eval_single_model(questions_val, 
                                                                  answers_val, 
                                                                  graph_embeddings_P, 
                                                                  graph_embeddings_Q, 
                                                                  candidates, 
                                                                  model, 
                                                                  device)

        epoch_train_loss = np.mean(batch_train_losses)
        epoch_val_loss = np.mean(batch_val_losses)        

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print("EPOCH", epoch, " train loss ", epoch_train_loss, " val loss ", epoch_val_loss, " val acc ", epoch_val_acc)
        
        if epoch_val_loss < min_val_loss:
            print(f'New loss checkpoint achieved: previous {min_val_loss} new {epoch_val_loss}')

            min_val_loss = epoch_val_loss
            torch.save(encoder.state_dict(), checkpoint_path / f'encoder_{model_name}_best_loss.pt')
            torch.save(projection_E.state_dict(), checkpoint_path / f'projection_E_{model_name}_best_loss.pt')
            torch.save(projection_Q.state_dict(), checkpoint_path / f'projection_Q_{model_name}_best_loss.pt')
            torch.save(projection_P.state_dict(), checkpoint_path / f'projection_P_{model_name}_best_loss.pt')
            
        if epoch_val_acc > max_val_acc:
            print(f'New acc checkpoint achieved: previous {max_val_acc} new {epoch_val_acc}')

            max_val_acc = epoch_val_acc
            torch.save(encoder.state_dict(), checkpoint_path / f'encoder_{model_name}_best_acc.pt')
            torch.save(projection_E.state_dict(), checkpoint_path / f'projection_E_{model_name}_best_acc.pt')
            torch.save(projection_Q.state_dict(), checkpoint_path / f'projection_Q_{model_name}_best_acc.pt')
            torch.save(projection_P.state_dict(), checkpoint_path / f'projection_P_{model_name}_best_acc.pt')
        
        if epoch % 5 == 0:
            torch.save(encoder.state_dict(), checkpoint_path / f'encoder_{model_name}_{epoch}.pt')
            torch.save(projection_E.state_dict(), checkpoint_path / f'projection_E_{model_name}_{epoch}.pt')
            torch.save(projection_Q.state_dict(), checkpoint_path / f'projection_Q_{model_name}_{epoch}.pt')
            torch.save(projection_P.state_dict(), checkpoint_path / f'projection_P_{model_name}_{epoch}.pt')

    np.save(checkpoint_path / f'train_loss_{model_name}.npy', np.array(train_losses))
    np.save(checkpoint_path / f'val_loss_{model_name}.npy', np.array(val_losses))
    np.save(checkpoint_path / f'val_acc_{model_name}.npy', np.array(val_accuracies))

    
def train_ensemble(n_models, n_epochs, proj_hidden_size, train_dataset, val_dataloader, models_path, device, loss, candidates, questions_val, answers_val, graph_embeddings_P, graph_embeddings_Q, models=None):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()

    for i in range(5, n_models + 5):
        np.random.seed(i)
        torch.manual_seed(i)
        g.manual_seed(i)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64, 
            shuffle=True, 
            worker_init_fn=seed_worker,
            generator=g,
        )
        if models:
            train_model(proj_hidden_size, train_dataloader, val_dataloader, models_path, n_epochs, i, device, loss, candidates, questions_val, answers_val, graph_embeddings_P, graph_embeddings_Q, models[i])
        else:
            train_model(proj_hidden_size, train_dataloader, val_dataloader, models_path, n_epochs, i, device, loss, candidates, questions_val, answers_val, graph_embeddings_P, graph_embeddings_Q)