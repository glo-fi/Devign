import numpy as np
import torch
from data_loader import n_identifier, g_identifier, l_identifier
import inspect
from datetime import datetime


def load_default_identifiers(n: str, g: str, l: str):
    """Load default JSON key identifiers for graph data components.
    
    Args:
        n (str): Key for node features (None to use default)
        g (str): Key for graph feature (None to use default)
        l (str): Key for labels (None to use default)
        
    Returns:
        tuple: (node_id, graph_id, label_id) keys to use
    """
    if n is None:
        n = n_identifier
    if g is None:
        g = g_identifier
    if l is None:
        l = l_identifier
    return n, g, l


def initialize_batch(entries: list, batch_size: int, shuffle: bool = False):
    """Create batches by iterating through a dataset.
    
    Args:
        entries (list): List of data entries to create batches for
        batch_size (int): Maximum number of entries per batch
        shuffle (bool): Whether to randomly shuffle indices (default: False)
            - True for training data to improve model convergence
            - False for validation/test to maintain deterministic order
    
    Returns:
        list: List of index lists in reverse order, where each sublist contains
              indices for one batch.
    """
    # Get total number of entries
    total = len(entries)
    # Create sequential indices
    indices = np.arange(0, total - 1, 1)
    # Optionally shuffle indices
    if shuffle:
        np.random.shuffle(indices)
    # Create batches of indices
    batch_indices = []
    start = 0
    end = len(indices)
    curr = start
    while curr < end:
        # Calculate end index for current batch
        c_end = curr + batch_size
        if c_end > end:
            c_end = end  # Truncate last batch if needed
        # Add batch of indices
        batch_indices.append(indices[curr:c_end])
        curr = c_end
        
    # Return batches in reverse order
    return batch_indices[::-1]


def tally_param(model: torch.nn.Module):
    """Calculate total number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model to analyse
        
    Returns:
        int: Total number of parameters across all layers
    """
    total = 0
    for param in model.parameters():
        total += param.data.nelement()
    return total


def debug(*msg, sep='\t'):
    """Print debug message with timestamp and caller information.
    
    Args:
        *msg: Variable number of message components to print
        sep (str): Separator between message components (default: tab)

    """
    # Get caller information
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    
    # Format timestamp
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    
    # Print header with file info
    print('[' + str(time) + '] File \"' + file_name + '\", line ' + str(ln) + '  ', end='\t')
    
    # Print message components
    for m in msg:
        print(m, end=sep)
    print('')
