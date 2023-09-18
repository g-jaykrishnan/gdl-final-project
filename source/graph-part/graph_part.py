import torch
import os
import networkx as nx
import torch_geometric
import torch_geometric.utils as tgutils
from ogb.nodeproppred import PygNodePropPredDataset
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_sparse as ts
from torch.utils.tensorboard import SummaryWriter
import dgl
import graph_part_net as net
import datetime

def train(model, data, parts, train_index, test_index, valid_index, ss_loss_weight, epochs, optimizer, writer, run):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss_list = []
    train_acc_list = {
        "s": [],
        "ss": []
    }
    valid_acc_list = {
        "s": [],
        "ss": []
    }
    test_acc_list = {
        "s": [],
        "ss": []
    }

    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        out, out_ss = model.forward(data.adj_t, data.x)
        if ss_loss_weight < 1:
            label_loss = loss_fn(out[train_index], data.y[train_index].squeeze())
            ss_loss = loss_fn(out_ss[train_index], parts[train_index]) 
            loss_final = (1-ss_loss_weight)*label_loss + (ss_loss_weight)*ss_loss
        else:
            ss_loss = loss_fn(out_ss[train_index], parts[train_index]) 
            loss_final = ss_loss

        train_loss_list.append(loss_final.detach().item())
        writer.add_scalar(f"GP-Loss-run{run}", round(loss_final.detach().item(),2), epoch)
        loss_final.backward()
        optimizer.step()

        out_valid_s, out_valid_ss = test(model, data, valid_index)
        out_test_s, out_test_ss = test(model, data, test_index)

        train_acc_s = accuracy(out[train_index], data.y[train_index])
        train_acc_ss = accuracy(out_ss[train_index], parts[train_index])
        valid_acc_s = accuracy(out_valid_s, data.y[valid_index])
        valid_acc_ss = accuracy(out_valid_ss, parts[valid_index])
        test_acc_s = accuracy(out_test_s, data.y[test_index])
        test_acc_ss = accuracy(out_test_ss, parts[test_index])
        
        if epoch == 1 or epoch % 100 == 0:
            print("epoch=", epoch, '\n',
                  "tr_acc_s=", round(train_acc_s.detach().item(),2), '\n',
                  "tr_acc_ss=", round(train_acc_ss.detach().item(),2), '\n',
                  "v_acc_s=", round(valid_acc_s.detach().item(),2), '\n',
                  "v_acc_ss=", round(valid_acc_ss.detach().item(),2), '\n',
                  "te_acc_s=", round(test_acc_s.detach().item(),2), '\n',
                  "te_acc_ss=", round(test_acc_ss.detach().item(),2))
            
        writer.add_scalar(f"GP-Train-Acc-S-run{run}", round(train_acc_s.detach().item(), 2), epoch)
        writer.add_scalar(f"GP-Train-Acc-SS-run{run}", round(train_acc_ss.detach().item(), 2), epoch)
        writer.add_scalar(f"GP-Valid-Acc-S-run{run}", round(valid_acc_s.detach().item(), 2), epoch)
        writer.add_scalar(f"GP-Valid-Acc-SS-run{run}", round(valid_acc_ss.detach().item(), 2), epoch)
        writer.add_scalar(f"GP-Test-Acc-S-run{run}", round(test_acc_s.detach().item(), 2), epoch)
        writer.add_scalar(f"GP-Test-Acc-SS-run{run}", round(test_acc_ss.detach().item(), 2), epoch)

        train_acc_list["s"].append(train_acc_s.detach().item())
        train_acc_list["ss"].append(train_acc_ss.detach().item())
        valid_acc_list["s"].append(valid_acc_s.detach().item())
        valid_acc_list["ss"].append(valid_acc_ss.detach().item())
        test_acc_list["s"].append(test_acc_s.detach().item())
        test_acc_list["ss"].append(test_acc_ss.detach().item())
        
    return train_loss_list, train_acc_list, valid_acc_list, test_acc_list

def test(model, data, index):
    model.eval()

    with torch.no_grad():
        out, out_ss = model.forward(data.adj_t, data.x)
    
    return out[index], out_ss[index]

def accuracy(input, target):
    sig = torch.nn.Sigmoid()
    input = sig(input.detach())
    _, pred = torch.max(input, dim=1)
    if target.ndim > 2:
        raise ValueError("dim > 2")
    elif target.ndim == 2:
        target = target.detach().squeeze()
    else:
        target = target.detach()
    accuracy = (target == pred).sum()
    accuracy = accuracy / len(target)

    return accuracy

def main():
    REPO_ROOT = os.path.dirname(__file__)
    DATA_DIR = os.path.join(os.path.join(REPO_ROOT, "data"), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    writer = SummaryWriter(log_dir=DATA_DIR)

    # get dataset, indices, data
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    split_index = dataset.get_idx_split()

    # defining parameters and device
    part_dim = 80
    dropout = 0.5
    ## the weight of the self supervised term in loss; set to 1 for unsupervised
    ss_loss_weight = 0.8
    lr = 0.001
    runs = 5
    epochs = 1000
    device = f'cuda:{4}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # create dgl graph and partition
    G = tgutils.to_networkx(data=data)
    G = dgl.from_networkx(G)
    parts = dgl.metis_partition_assignment(g=G, k=part_dim)

    # processing data and moving to device
    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric() # has significant effect write about this
    data = data.to(device)
    parts = parts.to(device)

    # dataset = dataset.to(device)
    train_index = split_index['train'].to(device)
    test_index = split_index['test'].to(device)
    valid_index = split_index['valid'].to(device)

    model = net.GraphPart(in_channels=dataset.num_features, num_classes=dataset.num_classes, dropout=dropout, num_parts=part_dim).to(device)
    
    if ss_loss_weight == 1:
        print("Completely unsupervised")
    else:
        print("Using actual graph labels for training")
    
    for run in range(runs):
        print("\n", "Run=", run, '\n')
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loss_list, train_acc_list, valid_acc_list, test_acc_list  = train(model, data, parts, train_index, test_index, valid_index, ss_loss_weight, epochs, optimizer, writer, run)
        train_loss_max = max(train_loss_list)
        train_acc_s_max = max(train_acc_list["s"])
        train_acc_ss_max = max(train_acc_list["ss"])
        valid_acc_s_max = max(valid_acc_list["s"])
        valid_acc_ss_max = max(valid_acc_list["ss"])
        test_acc_s_max = max(test_acc_list["s"])
        test_acc_ss_max = max(test_acc_list["ss"])

        print("Run max", "\n",
              "tr_loss_max=", round(train_loss_max,2), '\n',
              "tr_acc_s_max=", round(train_acc_s_max,2), '\n',
              "tr_acc_ss_max=", round(train_acc_ss_max,2), '\n',
              "v_acc_s_max=", round(valid_acc_s_max,2), '\n',
              "v_acc_ss_max=", round(valid_acc_ss_max,2), '\n',
              "te_acc_s_max=", round(test_acc_s_max,2), '\n',
              "te_acc_ss_max=", round(test_acc_ss_max,2))
        

if __name__ == "__main__":
    main()