import os
import datetime
import random
import torch
import torch.nn as nn
import torch_geometric
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch.utils.tensorboard import SummaryWriter


def test(model, valid_data, valid_labels, test_data, test_labels, evaluator, writer, epoch):
    
    with torch.no_grad():
        valid_out = model(valid_data)
        test_out = model(test_data)

        valid_pred = torch.argmax(valid_out, dim=1, keepdim=True)
        test_pred = torch.argmax(test_out, dim=1, keepdim=True)


        valid_acc = evaluator.eval({
            'y_true': valid_labels,
            'y_pred': valid_pred
        })['acc']
        test_acc = evaluator.eval({
            'y_true': test_labels,
            'y_pred': test_pred
        })['acc']

        accs = {'valid_acc' : valid_acc, 
                'test_acc' : test_acc}

    return accs


class SGC(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(SGC, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        softmax = nn.Softmax(dim=1)
        return softmax(self.linear(x))
    
def main():

    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    data = dataset[0]
    n_classes = dataset.num_classes
    labels = data.y
    train_index_actual = dataset.get_idx_split()["train"]
    valid_index_actual = dataset.get_idx_split()["valid"]
    test_index_actual = dataset.get_idx_split()["test"]
    
    device = f'cuda:{4}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Define parameters
    epochs = 500
    lr = 0.001
    fracs = [0.01, 0.05, 0.1, 0.5, 1]
    runs = 5

    # import embedding
    latents = torch.load(f="/home/sanketh/jay/source/sgc/full_batch_embedding_run_0.pt")
    latent = latents[0]

    for run in range(runs):

        file_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(file_dir, f"logs-run-{run}")
        writer = SummaryWriter(log_dir=log_dir)

        for frac in fracs:
            save_dir = os.path.join(log_dir, f"frac-{frac}")
            os.makedirs(save_dir, exist_ok=True)

            if frac != 1:
                n_train = int(frac * len(train_index_actual))
                n_valid = int(frac * len(valid_index_actual))
                n_test = int(frac * len(test_index_actual))

                train_rands = torch.randint(low=0, high=len(train_index_actual), size=(n_train,))
                valid_rands = torch.randint(low=0, high=len(valid_index_actual), size=(n_valid,))
                test_rands = torch.randint(low=0, high=len(test_index_actual), size=(n_test,))

                train_index = train_index_actual[train_rands]
                valid_index = valid_index_actual[valid_rands]
                test_index = test_index_actual[test_rands]

                train_data = latent[train_index]
                valid_data = latent[valid_index]
                test_data = latent[test_index]
            else:
                train_index = train_index_actual
                valid_index = valid_index_actual
                test_index = test_index_actual

                train_data = latent[train_index]
                valid_data = latent[valid_index]
                test_data = latent[test_index]

            evaluator = Evaluator(name='ogbn-arxiv')

            train_labels = labels[train_index].to(device)
            valid_labels = labels[valid_index].to(device)
            test_labels = labels[test_index].to(device)
            
            train_data = train_data.to(device)
            valid_data = valid_data.to(device)
            test_data = test_data.to(device)
            
            train_index = train_index.to(device)
            valid_index = valid_index.to(device)
            test_index = test_index.to(device)

            model = SGC(in_channels=train_data.size(1), out_channels=n_classes)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()
            accs = {'train_accs' : [],
                    'valid_accs' : [], 
                    'test_accs' : []}
            
            for epoch in range(0, epochs+1):
                model.train()
                optimizer.zero_grad()

                train_out = model(train_data)
                loss = loss_fn(train_out, train_labels.squeeze())

                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss", loss.detach().item(), epoch)
                train_pred = torch.argmax(train_out, dim=1, keepdim=True)
                train_acc =  evaluator.eval({
                    'y_true': train_labels,
                    'y_pred': train_pred
                    })['acc']
                accs['train_accs'].append(train_acc)

                accs_dict = test(model, valid_data, valid_labels, test_data, test_labels, evaluator, writer, epoch)
                accs['valid_accs'].append(accs_dict['valid_acc'])
                accs['test_accs'].append(accs_dict['test_acc'])

                writer.add_scalar(f"train-acc-{frac}-{run}", train_acc, epoch)
                writer.add_scalar(f"valid-acc-{frac}-{run}", accs_dict['valid_acc'], epoch)
                writer.add_scalar(f"test-acc-{frac}-{run}", accs_dict['test_acc'], epoch)
                if epoch == 1 or epoch % 100 == 0:
                    print(f"Epoch: {epoch}, \n Loss: {loss}, \n Train Accuracy: {train_acc}, \n Valid Accuracy: {accs_dict['valid_acc']}, \n Test Accuracy: {accs_dict['test_acc']}")



            torch.save(obj=accs, f=os.path.join(save_dir, "accs.pt"))
        

if __name__ == "__main__":
    main()