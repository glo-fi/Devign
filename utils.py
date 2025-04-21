import re
import os
import numpy as np
import torch
import json
import random
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



def split_json(input_file, output_file_1, output_file_2, split_ratio=0.8, shuffle=False):
    """
    Reads a list of dictionaries from `input_file` (JSON),
    splits it into two smaller lists at the given ratio,
    and writes each subset to a separate JSON file.

    :param input_file: Path to the input JSON file containing a list of dictionaries.
    :param output_file_1: Path to the first output JSON file.
    :param output_file_2: Path to the second output JSON file.
    :param split_ratio: Fraction of records to go to 'output_file_1' (0 < split_ratio < 1).
    :param shuffle: Whether to randomize the order of the input data before splitting.
    """

    # 1. Read the original JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # (Optional) Shuffle the list in-place
    if shuffle:
        random.shuffle(data)

    # 2. Compute the split index based on `split_ratio`
    split_index = int(len(data) * split_ratio)

    # Split the data into two lists
    data_part1 = data[:split_index]
    data_part2 = data[split_index:]

    # 3. Write the first subset to output_file_1
    with open(output_file_1, 'w') as f1:
        json.dump(data_part1, f1)

    # 4. Write the second subset to output_file_2
    with open(output_file_2, 'w') as f2:
        json.dump(data_part2, f2)


def tally_targets(input_file):
    """
    Reads a JSON file (which is expected to contain a list of dictionaries),
    counts how many times the "targets" value is 0 vs. 1,
    and prints the results along with their ratio.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_zeros = 0
    num_ones = 0

    # Count how many items have targets=0 or targets=1
    for item in data:
        if item.get("targets")[0][0] == 0:
            num_zeros += 1
        elif item.get("targets")[0][0] == 1:
            num_ones += 1

    total = num_zeros + num_ones
    if total == 0:
        print("No entries with targets=0 or targets=1 were found.")
        return

    # Compute ratio as fraction of total
    ratio_zeros = num_zeros / total
    ratio_ones = num_ones / total

    print(f"Total entries evaluated: {total}")
    print(f"Number of 0s: {num_zeros} (ratio = {ratio_zeros:.2f})")
    print(f"Number of 1s: {num_ones} (ratio = {ratio_ones:.2f})")

def combine_ggnn_json_files(input_dir, output_file):
    """
    Finds all files in `input_dir` with the format 'ggnn_{i}.json' (where {i} is an integer),
    and concatenates the lists of dictionaries within those files into a single output JSON.

    This method loads one file at a time into memory (rather than all files together),
    which can be acceptable if each individual file is not excessively large.

    :param input_dir: Path to the directory with 'ggnn_{i}.json' files.
    :param output_file: Path to the file where combined JSON will be written as a list of dicts.
    """

    # Regex to identify files like 'ggnn_123.json'
    pattern = re.compile(r'^ggnn_(\d+)\.json$')
    
    # Collect and sort files by their numeric index, e.g. {i} in 'ggnn_{i}.json'
    files = []
    for f in os.listdir(input_dir):
        match = pattern.match(f)
        if match:
            # Extract the integer part for sorting
            idx = int(match.group(1))
            files.append((idx, f))
    files.sort(key=lambda x: x[0])  # sort by the numeric index

    # Open the output file for writing
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Write the opening bracket of the JSON array
        out_f.write('[')

        first_chunk_written = False

        # Process each JSON file in sorted order
        for _, filename in files:
            filepath = os.path.join(input_dir, filename)
            # Read that file (a list of dictionaries)
            with open(filepath, 'r', encoding='utf-8') as in_f:
                data = json.load(in_f)  # This is a list of dicts.

            # If this is not the first chunk, prepend a comma to separate the arrays
            if first_chunk_written and len(data) > 0:
                out_f.write(',')

            first_chunk_written = True

            # Dump the list but strip away the outer '[' and ']' so we only get the contents
            # Example: if data is [dict1, dict2], json.dumps(data) => "[{...},{...}]"
            # We slice [1:-1] to remove the bounding brackets, then we write them out.
            list_as_str = json.dumps(data, ensure_ascii=False)
            if len(data) > 0:
                # Remove the first '[' and last ']' only if data is non-empty
                out_f.write(list_as_str[1:-1])

        # Close the JSON array
        out_f.write(']')

if __name__ == "__main__":
    #combine_ggnn_json_files("/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/fq_json/", "./all_ggnn.json")
    split_json(
        input_file="./all_ggnn.json",
        output_file_1="./train_all_ggnn.json",
        output_file_2="./valid_all_ggnn.json",
        split_ratio=0.8,  # 80% goes to split_part1, 20% to split_part2
        shuffle=True      # Randomize the list before splitting
    )
