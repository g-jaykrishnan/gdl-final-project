import os
from typing import Dict, List, Tuple

from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, PPI, WikiCS
from torch_geometric import transforms as T
from torch_geometric.utils import to_undirected

from gssl import DATA_DIR


def load_dataset(name: str) -> Tuple[Data, List[Dict[str, torch.Tensor]]]:
    ds_path = os.path.join(DATA_DIR, "datasets/", name)

    data = read_ogb_dataset(name=name, path=ds_path)
    data.edge_index = to_undirected(edge_index=data.edge_index, num_nodes=data.num_nodes)
    masks = [
            {
                "train": data.train_mask,
                "val": data.val_mask,
                "test": data.test_mask,
            }
            ]
    
    return data, masks


def read_ogb_dataset(name: str, path: str) -> Data:
    dataset = PygNodePropPredDataset(root=path, name=name)
    split_idx = dataset.get_idx_split()

    data = dataset[0]

    data.train_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.train_mask[split_idx["train"]] = True

    data.val_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.val_mask[split_idx["valid"]] = True

    data.test_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.test_mask[split_idx["test"]] = True

    data.y = data.y.squeeze(dim=-1)

    return data


def load_ppi() -> Tuple[PPI, PPI, PPI]:
    ds_path = os.path.join(DATA_DIR, "datasets/PPI")
    feature_norm = T.NormalizeFeatures()

    train_ppi = PPI(root=ds_path, split="train", transform=feature_norm)
    val_ppi = PPI(root=ds_path, split="val", transform=feature_norm)
    test_ppi = PPI(root=ds_path, split="test", transform=feature_norm)

    return train_ppi, val_ppi, test_ppi
