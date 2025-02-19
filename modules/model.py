import torch
from dgl.nn import GatedGraphConv
from data_loader.batch_graph import BatchGraph
from torch import nn
import torch.nn.functional as f


class DevignModel(nn.Module):
    """Implementation of the Devign vulnerability detection model from the paper.
    
    Architecture combines GGNN with convolutional layers.
    
    Args:
        input_dim (int): Dimension of node features
        output_dim (int): GGNN hidden state dimension
        max_edge_types (int): Number of distinct edge types in the graphs
        num_steps (int): Number of GGNN propagation steps (default: 8)
    """
    def __init__(self, input_dim: int, output_dim: int, max_edge_types: int, num_steps: int = 8):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: BatchGraph, cuda=False):
        """Forward pass of the Devign model.
        
        Args:
            batch (GGNNBatchGraph): Batched graph data
            cuda (bool): Whether to use GPU acceleration
            
        Returns:
            Tensor: Binary predictions tensor of shape (batch_size,)
                   with vulnerability probabilities
                   
        Architecture Flow:
            1. GGNN message passing to get node embeddings
            2. Two conv-pool paths:
               - Path 1: GGNN outputs only (h_i)
               - Path 2: Concatenated input + GGNN features (c_i)
            3. Element-wise multiplication of path outputs
            4. Mean pooling and sigmoid for final prediction
        """
        # Get graph inputs and apply GGNN
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        outputs = self.ggnn(graph, features, edge_types)
        
        # De-batchify and prepare features
        x_i, _ = batch.de_batchify_graphs(features)  # Original features
        h_i, _ = batch.de_batchify_graphs(outputs)   # GGNN outputs
        c_i = torch.cat((h_i, x_i), dim=-1)         # Concatenated features
        batch_size, num_node, _ = c_i.size()
        
        # Path 1: Process GGNN outputs
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        
        # Path 2: Process concatenated features
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        
        # Combine paths and get final prediction
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))  # Element-wise multiply
        avg = before_avg.mean(dim=1)                               # Mean pooling
        result = self.sigmoid(avg).squeeze(dim=-1)                 # Binary prediction
        return result


class GGNNSum(nn.Module):
    """Simplified GGNN model.
    
    This model:
    1. Processes input graph with GGNN for message passing
    2. Sums node embeddings to get graph representation
    3. Applies linear classifier and sigmoid for binary prediction
    
    Args:
        input_dim (int): Dimension of input node features
        output_dim (int): Dimension of node embeddings after GGNN
        max_edge_types (int): Number of distinct edge types in graphs
        num_steps (int): Number of GGNN propagation steps (default: 8)
        
    Architecture:
        - GGNN layer: input_dim -> output_dim
        - Sum aggregation: output_dim -> output_dim
        - Linear classifier: output_dim -> 1
        - Sigmoid activation for binary output
    """
    def __init__(self, input_dim: int, output_dim: int, max_edge_types: int, num_steps: int = 8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        
        # GGNN for message passing
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        # Linear layer for classification
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        # Sigmoid for binary prediction
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: BatchGraph, cuda: bool = False):
        """Forward pass of the GGNN-Sum model.
        
        Args:
            batch (BatchGraph): Batched graph data
            cuda (bool): Whether to use GPU acceleration
            
        Returns:
            Tensor: Binary predictions tensor of shape (batch_size,)
                   with vulnerability probabilities
                   
        Steps:
            1. Get graph structure and features
            2. Apply GGNN for node embedding
            3. Sum node embeddings for graph representation
            4. Classify with linear layer and sigmoid
        """
        # Get graph inputs
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        
        # Apply GGNN
        outputs = self.ggnn(graph, features, edge_types)
        
        # De-batchify and sum node embeddings
        h_i, _ = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        
        # Apply sigmoid for binary prediction
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result
