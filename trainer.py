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
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
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
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), accuracy_score(all_targets, all_predictions) * 100
    pass


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
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
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
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


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
    best_f1 = 0               # Track best validation F1 score
    try:
        for step_count in range(max_steps):
            model.train()
            model.zero_grad()
            graph, targets = dataset.get_next_train_batch()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            if log_every is not None and (step_count % log_every == log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
            if step_count % dev_every == (dev_every - 1):
                valid_loss, valid_f1 = evaluate_loss(model, loss_function, dataset.initialize_train_batch(),
                                                     dataset.get_next_train_batch)
                if valid_f1 > best_f1:
                    patience_counter = 0
                    best_f1 = valid_f1
                    best_model = copy.deepcopy(model.state_dict())
                    _save_file = open(save_path + '-model.bin', 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                else:
                    patience_counter += 1
                debug('Step %d\t\tTrain Loss %10.3f\tValid Loss%10.3f\tf1: %5.2f\tPatience %d' % (
                    step_count, np.mean(train_losses).item(), valid_loss, valid_f1, patience_counter))
                debug('=' * 100)
                train_losses = []
                if patience_counter == max_patience:
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(save_path + '-model.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_train_batch(),
                                       dataset.get_next_train_batch)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    debug('=' * 100)
