import copy
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from data_loader.dataset import DataSet
from utils import debug


def evaluate_loss(model: torch.nn.Module, 
                  loss_function: torch.nn.Module, 
                  num_batches: int, 
                  data_iter, 
                  cuda: bool = False):
    """Calculate loss and accuracy on a dataset split.
    
    Args:
        model (nn.Module): Devign model to evaluate
        loss_function: Loss criterion (e.g. BCELoss)
        num_batches (int): Number of batches to evaluate
        data_iter (?): Iterator function that returns (graph, targets)
        cuda (bool): Whether to use GPU acceleration. Not used
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            #targets = targets.cuda()
            predictions = model(graph, cuda=False)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            #predictions = torch.sigmoid(predictions).ge(0.5).int()
            #all_predictions.extend(predictions.detach().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), accuracy_score(all_targets, all_predictions) * 100


def evaluate_metrics(model: torch.nn.Module, 
                     loss_function: torch.nn.Module, 
                     num_batches: int, 
                     data_iter):
    """Calculate metrics on dataset split.
    
    Args:
        model (nn.Module): Devign model to evaluate
        loss_function (nn.Module): Loss criterion (e.g. BCELoss)
        num_batches (int): Number of batches to evaluate
        data_iter: Iterator function that returns (graph, targets)
        
    Returns:
        tuple: (accuracy, precision, recall, f1) percentages
        
    Note:
        All metrics are computed on binary predictions (threshold=0.5)
        and returned as percentages.
    """
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            #targets = targets.cuda()
            predictions = model(graph, cuda=False)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            print("Inital Predictions: ", predictions)
            #predictions = torch.sigmoid(predictions).ge(0.5).int()
            #print("Sigmoid Predictions: ", predictions)

            if predictions.ndim == 2:
                print("If Predictions: ", np.argmax(predictions.numpy(), axis=-1).tolist())
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                print("Else Predictions: ", predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(dtype=torch.int32).numpy().tolist())
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                    dtype=torch.int32).numpy().tolist()
                )
            #all_predictions.extend(predictions.detach().cpu().numpy().tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100

def get_all_embeddings_for_dataset(model: torch.nn.Module, dataset: DataSet, cuda=False):
    """
    Iterates through all batches in 'dataset' using 'batch_initializer'
    and 'batch_getter', returning a big tensor of GGNN embeddings.
    
    Args:
        model: your DevignModel
        dataset: your DataSet object
        batch_initializer: function to initialize batch iteration
        batch_getter: function to fetch the next batch
        cuda: whether GPU is used
        
    Returns:
        A list or tensor of embeddings for all samples in the dataset.
        The shape might be irregular if each sample has a different # of nodes.
    """
    model.eval()
    embeddings_list = []
    with torch.no_grad():
        num_batches = dataset.initialize_train_batch()  # Typically returns how many batches you can iterate
        for _ in range(num_batches):
            graph, _ = dataset.get_next_train_batch()
            h_i = model.get_ggnn_embeddings(graph, cuda=cuda)
            # h_i is shape [batch_size, #nodes, out_dim] if all graphs in that batch are same #nodes
            # or a list if they differ. Adapt how you store it:
            embeddings_list.append(h_i.cpu())
    model.train()
    
    # You may want to combine them somehow (e.g., a list of [B, N_i, out_dim]),
    # or just keep them separate in a Python list. Up to you:
    return embeddings_list

def save_all_embeddings_in_chunks(model, dataset: DataSet,
                                  base_filename='saved_data/best_ggnn_embeddings_chunk',
                                  cuda=False):
    """
    Iterates over the entire dataset in consecutive batches,
    saving the embeddings for each batch into a separate file.

    Args:
        model: Your GNN model (with get_ggnn_embeddings method).
        dataset: DataSet object.
        batch_initializer: dataset.initialize_train_batch (or similar).
        batch_getter: dataset.get_next_train_batch (or similar).
        base_filename: prefix for chunk file naming.
        cuda: whether GPU is used.

    Returns:
        Number of chunks saved, total number of samples (optional).
    """
    model.eval()
    num_batches = dataset.initialize_train_batch()  # e.g., returns how many total train batches
    chunk_idx = 0
    total_samples = 0
    with torch.no_grad():
        for _ in range(num_batches):
            graph, targets = dataset.get_next_train_batch()
            h_i = model.get_ggnn_embeddings(graph, cuda=cuda)

            chunk_data = {
                "embeddings": h_i.cpu(),
                "labels": targets.cpu() if targets.is_cuda else targets
            }

            this_chunk_filename = f"{base_filename}{chunk_idx}.pt"
            torch.save(chunk_data, this_chunk_filename)
            chunk_idx += 1

            # Optionally track how many graphs or nodes were saved
            # For example, if h_i is a list of length B:
            total_samples += len(h_i)  # if each item is one graph
    model.train()
    
    return chunk_idx, total_samples

def train(model: torch.nn.Module, 
          dataset: DataSet, 
          max_steps: int,
          dev_every: int,
          loss_function: torch.nn.Module, 
          optimizer: torch.optim.Optimizer,
          save_path: str, 
          log_every: int = 50,
          max_patience: int = 5):
    """Train Devign model with early stopping based on validation F1 score.
    
    Args:
        model: Devign model to train
        dataset (DataSet): Dataset container with train/valid splits
        max_steps (int): Maximum training iterations
        dev_every (int): Evaluate on validation set every N steps
        loss_function: Loss criterion (e.g. BCELoss)
        optimizer: Optimization algorithm
        save_path (str): Path prefix for saving model checkpoints
        log_every (int): Log training progress every N steps (default: 50)
        max_patience (int): Early stopping patience (default: 5)
        
    """
    debug('Start Training')
    train_losses = []          # Track training losses between validations
    best_model = None          # Store best model state
    patience_counter = 0       # Count validations without improvement
    best_f1 = 0              # Track best validation F1 score
    try:
        for step_count in tqdm(range(max_steps)):
            model.train()
            model.zero_grad()
            graph, targets = dataset.get_next_train_batch()
            #targets = targets.cuda()
            predictions = model(graph, cuda=False)
            batch_loss = loss_function(predictions, targets)
            if step_count % log_every == (log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
                debug('=' * 100)
            if dev_every is not None and (step_count % dev_every == dev_every - 1):
                # --- Print debug info for training set ---
                debug(f'Step {step_count}\t\tTrain Loss {batch_loss.item():.3f}')
                train_acc, train_prec, train_recall, train_f1 = evaluate_metrics(
                    model, loss_function,
                    dataset.initialize_train_batch(),
                    dataset.get_next_train_batch
                )
                debug(f"Train Metrics: Acc={train_acc:.2f}, Prec={train_prec:.2f}, "
                    f"Rec={train_recall:.2f}, F1={train_f1:.2f}")

                # --- Evaluate on validation set ---
                valid_acc, valid_prec, valid_recall, valid_f1 = evaluate_metrics(
                    model, loss_function,
                    dataset.initialize_valid_batch(),
                    dataset.get_next_valid_batch
                )
                debug(f"Valid Metrics: Acc={valid_acc:.2f}, Prec={valid_prec:.2f}, "
                    f"Rec={valid_recall:.2f}, F1={valid_f1:.2f}")

                # -- Check if we have new best F1; if so, overwrite embeddings --
                if valid_f1 > best_f1:
                    best_f1 = valid_f1

                    # Gather embeddings for entire dataset (or entire train set, etc.)
                    # Adjust these calls for whichever split(s) you want to save
                    save_all_embeddings_in_chunks(
                        model, dataset,
                        cuda=False
                    )
                    # Overwrite the same file each time
                    # debug(f"Saving Embedding Dataset")
                    # embeddings_filename = "best_ggnn_embeddings.pt"
                    # torch.save(train_embeddings, embeddings_filename)

                    # Log the new best F1 in a separate text file (append mode)
                    with open("best_scores_log.txt", "a") as f:
                        f.write(f"New best F1={best_f1:.2f} at step={step_count}\n")

                    # debug(f"Overwrote {embeddings_filename} with new best F1={best_f1:.2f}")
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
                #valid_loss, valid_f1 = evaluate_loss(model, loss_function, dataset.initialize_valid_batch(),
                #                                     dataset.get_next_valid_batch)
                #if valid_f1 > best_f1:
                #    patience_counter = 0
                #    best_f1 = valid_f1
                    #best_model = copy.deepcopy(model.state_dict())
                    #_save_file = open(save_path + '-model.bin', 'wb')
                    #torch.save(best_model.state_dict(), _save_file)
                    #_save_file.close()
                #else:
                #    patience_counter += 1
                #debug('Step %d\t\tTrain Loss %10.3f\tValid Loss%10.3f\tf1: %5.2f\tPatience %d' % (
                #    step_count, np.mean(train_losses).item(), valid_loss, valid_f1, patience_counter))
                #train_losses = []
                #if patience_counter == max_patience:
                #    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')