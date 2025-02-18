import torch
from dgl import DGLGraph


class BatchGraph:
    def __init__(self):
        """Initialise a batch graph container for aggregating multiple subgraphs.
        
        Maintains:
        - graph: DGLGraph that aggregates all subgraphs
        - number_of_nodes: Total nodes across all subgraphs
        - graphid_to_nodeids: Mapping from subgraph ID to its node IDs in the batch
        - num_of_subgraphs: Count of subgraphs in batch
        """
        self.graph = DGLGraph()
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0

    def add_subgraph(self, _g: DGLGraph):
        """Add a subgraph to the batch while preserving node/edge features.
        
        Args:
            _g (DGLGraph): Subgraph to add. Must contain:
                - ndata['features']: Node features
                - edata['etype']: Edge type identifiers
        """
        assert isinstance(_g, DGLGraph)
        
        # Get number of nodes in the new subgraph
        num_new_nodes = _g.number_of_nodes()
        
        # Store mapping from subgraph ID to its node IDs in the batched graph
        # Format: {subgraph_id: tensor([start_node_id, start_node_id+1, ..., end_node_id])}
        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes)))
        
        # Add subgraph nodes to batched graph with their features
        self.graph.add_nodes(num_new_nodes, data=_g.ndata)
        
        # Get all edges from the subgraph and adjust their indices for batched graph
        sources, dests = _g.all_edges()
        # Offset edge indices by current node count to prevent overlap with existing nodes
        sources += self.number_of_nodes
        dests += self.number_of_nodes
        
        # Add edges to batched graph with their type information
        self.graph.add_edges(sources, dests, data=_g.edata)
        
        # Update counters
        self.number_of_nodes += num_new_nodes
        self.num_of_subgraphs += 1

    def cuda(self, device: str = None):
        """
        Modifies self.graphid_to_nodeids by sending all nodeids to cuda device
        and remapping them to graphids
        """
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)

    def de_batchify_graphs(self, features: torch.Tensor = None):
        """Reverse batching operation to extract per-subgraph features with padding.
        
        Args:
            features (Tensor, optional): Node features from batched graph. 
                Defaults to using 'features' from graph.ndata.
                
        Returns:
            tuple: (output_vectors, lengths) where:
                - output_vectors: Tensor of shape (num_subgraphs, max_nodes, feat_dim)
                    with zero-padding for subgraphs with fewer nodes
                - lengths: Tensor of original node counts per subgraph
        """
        if features is None:
            features = self.graph.ndata['features']
        assert isinstance(features, torch.Tensor)
        
        # Extract features for each subgraph using stored node ID mappings
        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) 
                  for gid in self.graphid_to_nodeids.keys()]
                  
        # Get original subgraph sizes and find maximum
        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        
        # Pad each subgraph's features to match maximum length
        for i, v in enumerate(vectors):
            # Calculates the size of the zero vector to pad in order to reach max length
            vectors[i] = torch.cat(
                (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), 
                                requires_grad=v.requires_grad,
                                device=v.device)), dim=0)
                                
        # Stack padded vectors and convert lengths to tensor, sending to output device
        output_vectors = torch.stack(vectors)
        lengths = torch.LongTensor(lengths).to(device=output_vectors.device)
        
        return output_vectors, lengths

    def get_network_inputs(self):
        raise NotImplementedError('Must be implemented by subclasses.')


class GGNNBatchGraph(BatchGraph):
    """Batch graph implementation specialised for GGNN model from Devign paper.
    
    Extends BatchGraph with GGNN-specific input formatting:
    - Returns (graph, features, edge_types) tuple
    - Handles CUDA device transfer
    """
    def get_network_inputs(self, cuda: bool = False, device: str = None):
        features = self.graph.ndata['features']
        edge_types = self.graph.edata['etype']
        if cuda:
            self.cuda(device=device)
            return self.graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return self.graph, features, edge_types
