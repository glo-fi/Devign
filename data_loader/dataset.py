import copy
import json
import os


import torch
from dgl import DGLGraph
from tqdm import tqdm

from data_loader.batch_graph import GGNNBatchGraph
from utils import load_default_identifiers, initialize_batch, debug


class DataSet:
    """Main dataset container for loading and managing graph data for Devign.
    
    Handles:
    - Loading JSON formatted graph data files
    - Creating train/validation/test splits
    - Batching graphs for GGNN model
    - Edge type management
    
    Args:
        train_src: Path to training data JSON file
        valid_src: Path to validation data JSON file (optional)
        test_src: Path to test data JSON file (optional)
        batch_size: Batch size (default: 32)
        n_ident: JSON key for node features (default: 'features')
        g_ident: JSON key for graph feature (default: 'structure')
        l_ident: JSON key for labels (default: 'label')
    """
    def __init__(self, train_src: str | os.PathLike, 
                 valid_src: str | os.PathLike = None, 
                 test_src: str | os.PathLike = None, 
                 batch_size: int = 32, 
                 n_ident: str = None, 
                 g_ident: str = None, 
                 l_ident: str = None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident = load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset(test_src, train_src, valid_src)
        self.initialize_dataset()

    def initialize_dataset(self):
        """Initialise all dataset splits with batches.
        
        Calls utils.initialize_batch() on train/valid/test examples using default batch size
        (i.e., currently no way to chaneg batch_size via this approach)
        Train batches are shuffled, while valid/test maintain deterministic order.
        """
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, test_src : str | os.PathLike,
                     train_src : str | os.PathLike, 
                     valid_src: str | os.PathLike):
        """Load and parse graph data from JSON files into DataEntry objects.
        
        Args:
            test_src: Path to test data JSON file
            train_src: Path to training data JSON file
            valid_src: Path to validation data JSON file (optional)
            
        Each JSON file should contain a list of graphs where each graph has:
            - Node features under n_ident key
            - Graph features under g_ident key
            - Target label under l_ident key
            
        Also initialises feature_size from first training example.
        """
        # Load training data first to initialise feature size
        debug('Reading Train File!')
        with open(train_src) as fp:
            train_data = json.load(fp)
            for entry in tqdm(train_data):
                example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                    edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                    debug('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
        # Then loop over validation file
        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src) as fp:
                valid_data = json.load(fp)
                for entry in tqdm(valid_data):
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.valid_examples.append(example)
        # Then test file
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = json.load(fp)
                for entry in tqdm(test_data):
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.test_examples.append(example)

    def get_edge_type_number(self, _type: str):
        """Get or create unique integer ID for an edge type string.
        
        Maintains a mapping from edge type strings to consecutive integers,
        assigning new IDs as needed.
        
        Args:
            _type: Edge type string from input graph
            
        Returns:
            int: Unique integer identifier for this edge type
        """
        if _type not in self.edge_types:
            # Assign next available ID to new edge type
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        """
        Getter for max_edge_type
        """
        return self.max_etype

    def initialize_train_batch(self, batch_size: int = -1):
        """Initialise training batches.
        
        Args:
            batch_size: Override default batch size if > 0
            
        Returns:
            int: Number of training batches created
        """
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)

    def initialize_valid_batch(self, batch_size: int = -1):
        """Initialise validation batches.
        
        Args:
            batch_size: Override default batch size if > 0
            
        Returns:
            int: Number of validation batches created
        """
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)

    def initialize_test_batch(self, batch_size: int = -1):
        """Initialise test batches.
        
        Args:
            batch_size: Override default batch size if > 0
            
        Returns:
            int: Number of test batches created
        """
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        """Create GGNN-compatible batch from selected entries.
        
        Args:
            entries (list): List of DataEntry objects to select from
            ids (list): Indices of entries to include in batch
            
        Returns:
            tuple: (GGNNBatchGraph, Tensor) containing:
                - Batched graph with node features and edge types
                - Binary vulnerability labels tensor
                
        Note:
            Creates deep copy of each graph to prevent modification of originals.
            Labels are converted to FloatTensor for BCE loss computation.
        """
        # Select entries by indices
        taken_entries = [entries[i] for i in ids]
        # Extract vulnerability labels
        labels = [e.target for e in taken_entries]
        # Create batch graph container
        batch_graph = GGNNBatchGraph()
        # Add each graph's structure and features (using deep copy to preserve originals)
        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):
        """Get next batch of training examples.
        
        Returns:
            tuple: (GGNNBatchGraph, Tensor) containing:
                - Batched graph with node features and edge types
                - Binary vulnerability labels tensor
                
        Note:
            Automatically reinitialises and reshuffles batches when exhausted.
        """
        # Reinitialise batches if exhausted
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        # Get next batch indices and create GGNN batch
        ids = self.train_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        """Get next batch of validation examples.
        
        Returns:
            tuple: (GGNNBatchGraph, Tensor) containing:
                - Batched graph with node features and edge types
                - Binary vulnerability labels tensor
                
        Note:
            Automatically reinitialises batches when exhausted.
            Maintains deterministic order for consistent evaluation.
        """
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        """Get next batch of test examples.
        
        Returns:
            tuple: (GGNNBatchGraph, Tensor) containing:
                - Batched graph with node features and edge types
                - Binary vulnerability labels tensor
                
        Note:
            Automatically reinitialises batches when exhausted.
            Maintains deterministic order for consistent evaluation.
        """
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)

class DataEntry:
    """Represents a single graph data point with features and structure.
    
    Attributes:
        dataset (DataSet): Parent dataset reference
        num_nodes (int): Number of nodes in the graph
        target (int): Vulnerability label (0/1)
        graph (DGLGraph): Graph structure with node features and edge types
        features (Tensor): Node feature matrix
        
    Args:
        dataset (DataSet): Parent dataset container
        num_nodes (int): Number of nodes in the graph
        features (list): List of node feature vectors
        edges (list): Edge list as tuples (source, edge_type, target)
        target (int): Vulnerability classification label
    """
    def __init__(self, dataset: DataSet, num_nodes: int, features, edges, target: int):
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.target = target
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})
