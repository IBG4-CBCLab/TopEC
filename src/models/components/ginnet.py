from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.nn import radius_graph
class GIN(torch.nn.Module):

    def __init__(self, num_atoms, num_classes, hidden_channels, cutoff, dropout):
        super().__init__()
        
        self.cutoff = cutoff
        self.embedding = Linear(num_atoms, hidden_channels)
        self.conv1 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels),
                       BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.dropout = Dropout(p=dropout)
        self.lin1 = Linear(hidden_channels*3, hidden_channels*3)
        self.lin2 = Linear(hidden_channels*3, num_classes)
        self.act = ReLU()
    def forward(self, x, pos, edge_index, batch):
        
        if edge_index is None:

            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch
            )
        
        # Node embeddings 
        h = self.embedding(x)
        
        h1 = self.conv1(h, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)

        # Concatenate graph embeddings
        graph_emb = torch.cat((h1, h2, h3), dim=1)
        
        h = self.lin1(graph_emb)
        h = self.dropout(self.act(h))
        out = self.lin2(h)
                
        return graph_emb, out
        
