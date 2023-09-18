from copy import deepcopy
from typing import Dict, Optional

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from tqdm.auto import tqdm

from gssl.augment import GraphAugmentor
from gssl.loss import barlow_twins_loss
from gssl.tasks import evaluate_node_classification_acc
from gssl.utils import plot_vectors


class FullBatchModel:

    def __init__(
        self,
        encoder: nn.Module,
        augmentor: GraphAugmentor,
        lr_base: float,
        total_epochs: int,
        warmup_epochs: int,
        use_pytorch_eval_model: bool,
    ):
        self._device = torch.device(
            f'cuda:{4}' if torch.cuda.is_available() else "cpu"
        )

        self._encoder = encoder.to(self._device)

        self._optimizer = torch.optim.AdamW(
            params=self._encoder.parameters(),
            lr=lr_base,
            weight_decay=1e-5,
        )
        self._scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=self._optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=total_epochs,
        )

        self._augmentor = augmentor

        self._total_epochs = total_epochs

        self._use_pytorch_eval_model = use_pytorch_eval_model

    def fit(
        self,
        data: Data,
        logger: Optional[SummaryWriter] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> dict:
        self._encoder.train()
        logs = {
            "train_acc": [],
            "val_acc": [],
            "test_acc": [],
            "z": [],
        }

        data = data.to(self._device)

        for epoch in tqdm(iterable=range(self._total_epochs), leave=False):
            self._optimizer.zero_grad()

            (x_a, ei_a), (x_b, ei_b) = self._augmentor(data=data)

            z_a = self._encoder(x=x_a, edge_index=ei_a)
            z_b = self._encoder(x=x_b, edge_index=ei_b)

            loss = barlow_twins_loss(z_a=z_a, z_b=z_b)

            loss.backward()

            # Save metrics on every epoch
            logger.add_scalar("Loss", loss.item(), epoch)
            z = self.predict(data=data)
            self._encoder.train()  # Predict sets `eval()` mode

            accs = evaluate_node_classification_acc(
                z, data, masks=masks,
                use_pytorch=self._use_pytorch_eval_model,
            )

            logger.add_scalar("acc/train", accs["train"], epoch)
            logger.add_scalar("acc/val", accs["val"], epoch)
            logger.add_scalar("acc/test", accs["test"], epoch)

            logs["train_acc"].append(accs["train"])
            logs["val_acc"].append(accs["val"])
            logs["test_acc"].append(accs["test"])
            logger.add_scalar("norm", z.norm(dim=1).mean(), epoch)

            self._optimizer.step()
            self._scheduler.step()

        # Save embedding at the end
        logs["z"].append(deepcopy(z))

        data = data.to("cpu")

        return logs

    def predict(self, data: Data) -> torch.Tensor:
        self._encoder.eval()

        with torch.no_grad():
            z = self._encoder(
                x=data.x.to(self._device),
                edge_index=data.edge_index.to(self._device),
            )

            return z.cpu()


