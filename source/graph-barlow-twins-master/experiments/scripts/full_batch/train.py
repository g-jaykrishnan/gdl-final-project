import json
import os
import sys
import yaml
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

sys.path.append("source/graph-barlow-twins-master")

from gssl.augment import GraphAugmentor
from gssl.datasets import load_dataset
from gssl.full_batch.model import FullBatchModel
from gssl.utils import load_cls_from_str
from gssl.utils import seed


def evaluate_single_graph_full_batch_model(
    dataset_name: str,
    params: dict,
    augmentor: GraphAugmentor,
    logger: SummaryWriter
):
    data, masks = load_dataset(name=dataset_name)

    encoder = load_cls_from_str(params["encoder_cls"])(
        in_dim=data.num_node_features,
        out_dim=params["emb_dim"],
    )

    model = FullBatchModel(
        encoder=encoder,
        augmentor=augmentor,
        lr_base=params["lr_base"],
        total_epochs=params["total_epochs"],
        warmup_epochs=params["warmup_epochs"],
        use_pytorch_eval_model=params["use_pytorch_eval_model"],
    )

    logs = model.fit(
        data=data,
        logger=logger,
        masks=masks[0]
    )

    return logs

def main():
    seed()

    # Read dataset name
    dataset_name = 'ogbn-arxiv'

    # Read params
    params = {
        "total_epochs": 500,
        "warmup_epochs": 100,
        "encoder_cls": "gssl.full_batch.encoders.ThreeLayerGCNEncoder",
        "use_pytorch_eval_model": True,
        "emb_dim": 256,
        "lr_base": 1.e-3
        }

    outs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        f"data/",
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 
                        )

    emb_dir = os.path.join(outs_dir, "embeddings/")
    os.makedirs(emb_dir, exist_ok=True)

    train_metrics = []
    val_metrics = []
    test_metrics = []

    for i in tqdm(range(2), desc="Splits"):
        logger = SummaryWriter(log_dir=os.path.join(outs_dir, f"logs/{i}"))

        augmentor = GraphAugmentor(
            p_x_1=0.1,
            p_e_1=0.1,
        )

        logs = evaluate_single_graph_full_batch_model(
                dataset_name=dataset_name,
                params=params,
                augmentor=augmentor,
                logger=logger
            )

        train_metrics.append(logs[f"train_acc"])
        val_metrics.append(logs[f"val_acc"])
        test_metrics.append(logs[f"test_acc"])

        # Save latent vectors (embeddings)
        torch.save(obj=logs["z"], f=os.path.join(emb_dir, f"full_batch_embedding_run_{i}.pt"))
    
    torch.save(obj=train_metrics, f=os.path.join(emb_dir, f"full_batch_train_run_{i}.pt"))
    torch.save(obj=val_metrics, f=os.path.join(emb_dir,f"full_batch_val_run_{i}.pt"))
    torch.save(obj=test_metrics, f=os.path.join(emb_dir,f"full_batch_test_run_{i}.pt"))

if __name__ == "__main__":
    main()
