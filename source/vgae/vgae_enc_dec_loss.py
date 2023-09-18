import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# class VGAE_decoder(torch.nn.Module):
    
#     def __init__(self, dropout) -> None:
#         super(VGAE_decoder, self).__init__()
#         self.dropout = dropout

#     def forward(self, latent):
#         # latent = F.dropout(latent, training=self.training)
#         pred_adj = torch.matmul(latent, latent.t())
#         pred_adj = F.sigmoid(pred_adj)

#         return pred_adj
    

class VGAE_encoder(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout) -> None:
        super(VGAE_encoder, self).__init__()

        self.conv = GCNConv(in_channels, hidden_channels*2, cached=True)
        self.conv_mu = GCNConv(hidden_channels*2, out_channels, cached=True)
        self.conv_s = GCNConv(hidden_channels*2, out_channels, cached=True)
        self.dropout = dropout
        # self.decoder = VGAE_decoder(dropout)

    def reset_parameters(self):

        self.conv.reset_parameters()
        self.conv_mu.reset_parameters()
        self.conv_s.reset_parameters()

    def forward(self, x, adj):
        
        x = self.conv(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_mu = self.conv_mu(x, adj)
        x_mu = F.relu(x_mu)
        x_mu = F.dropout(x_mu, p=self.dropout, training=self.training)

        x_s = self.conv_s(x, adj)
        x_s = F.relu(x_s)
        x_s = F.dropout(x_s, p=self.dropout, training=self.training)

        return (x_mu, x_s)

    def reparametrize(self, mu, s):
        if self.training:
            std = torch.exp(s)
            eps = torch.rand_like(std)
            out = torch.mul(eps,std) + mu

        return out

    # def forward_all(self, x, adj):

    #     x_mu, x_s = self.encode(x, adj)
    #     out = self.reparametrize(x_mu, x_s)
    #     pred_adj = self.decoder(out)

    #     return pred_adj, x_mu, x_s
    

def loss_function(adj_pred, adj_act, x_mu, x_s, num_nodes):
    cost = F.binary_cross_entropy_with_logits(adj_pred, adj_act)

    kl_loss = -0.5 / num_nodes * torch.mean(torch.sum(1 + 2 * x_s - x_mu.pow(2) - x_s.exp().pow(2), 1))

    loss = kl_loss + cost
    
    return loss