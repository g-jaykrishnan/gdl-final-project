import torch
import torch_sparse
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T 
from vgae_enc_dec_loss import VGAE_encoder, loss_function
    

def train(model, train_data, optimizer):
    model.train()
    optimizer.zero_grad()

    x = model.encode(train_data.x, train_data.adj_t)
    loss = model.kl_loss()
    x = x.to_sparse()
    y = torch.sparse.mm(x, x.t()) # OOM error
    out = model.decoder.forward_all(x)
    # loss = loss_function(adj_pred, train_data.adj_t, x_mu, x_s, train_data.num_nodes)
    loss.backward()
    optimizer.step()

    return loss.item()

def main():

    runs = 1
    epochs = 100
    lr = 0.01
    log_steps = 1
    device = f'cuda:{7}'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    data = data.to(device)
    index_split = dataset.get_idx_split()
    train_index_split = index_split['train'].to(device)
    data = data.subgraph(train_index_split)
    
    data_t = T.ToSparseTensor()(data).to(device)
    
    model = VGAE(encoder=VGAE_encoder(in_channels=data.num_features, hidden_channels=data.num_features, out_channels=dataset.num_classes, dropout=0.5)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for run in range(runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        for epoch in range(epochs):
            loss = train(model, data_t, optimizer)
            if epoch % log_steps == 0:
                print(loss)
if __name__ == "__main__":
    main()