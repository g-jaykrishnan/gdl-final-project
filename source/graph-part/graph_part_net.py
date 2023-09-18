import torch
import torch_geometric
from torch_geometric.nn import GCNConv

class GraphPart(torch.nn.Module):

    def __init__(self, in_channels: int, num_classes: int, dropout: float, num_parts: int) -> None:
        super(GraphPart,self).__init__()

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels=in_channels, out_channels=2*in_channels, cached=True, add_self_loops=True, normalize=True))
        self.convs.append(GCNConv(in_channels=2*in_channels, out_channels=in_channels, cached=True, add_self_loops=True, normalize=True))

        self.s_classifier = torch.nn.Linear(in_features=in_channels, out_features=num_classes, bias=False)
        self.ss_classifier = torch.nn.Linear(in_features=in_channels, out_features=num_parts, bias=False)

    def forward(self, adj, x):
        
        z = x
        z_ss = x

        for layer in self.convs:
            z = layer(z, adj)
            z = self.relu(z)
            z = self.dropout(z)
        z = self.ss_classifier(z)

        for layer in self.convs:
            z_ss = layer(z_ss, adj)
            z_ss = self.relu(z_ss)
            z_ss = self.dropout(z_ss)
        z_ss = self.ss_classifier(z_ss)

        return z, z_ss
    
    def test(self, adj, x):

        for layer in self.convs:
            z = layer(x, adj)
            z = self.relu(z)

        return z

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.s_classifier.reset_parameters()
        self.ss_classifier.reset_parameters()

        