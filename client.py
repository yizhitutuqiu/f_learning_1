"""
Flower NumPyClient with FedProx proximal term in local training.
Strategy sends "proximal_mu" (or "proximal-mu") via config; client adds (mu/2)*||w - w_global||^2 to loss.
"""
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from dataset import client_dataloaders, load_fedprox_synthetic
from model import MCLR, get_parameters, set_parameters


def _proximal_term(
    model: nn.Module,
    global_params: List[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """(mu/2) * ||w - w_global||^2 summed over all parameters."""
    loss_prox = 0.0
    for p, g in zip(model.parameters(), global_params):
        loss_prox += ((p - torch.tensor(g, device=device, dtype=p.dtype)) ** 2).sum()
    return loss_prox / 2.0


class FedProxClient(fl.client.NumPyClient):
    """Client that adds proximal term when config contains proximal_mu / proximal-mu."""

    def __init__(
        self,
        cid: str,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.01,
        num_epochs: int = 20,
    ):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_parameters(self.model, parameters)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        # Copy of global params for proximal term (w_global)
        global_ndarrays = [np.copy(a) for a in parameters]
        mu = float(config.get("proximal_mu", config.get("proximal-mu", 0.0)))

        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=float(config.get("learning_rate", self.learning_rate)),
            weight_decay=1e-3,
        )
        criterion = nn.CrossEntropyLoss()

        num_samples = 0
        total_loss = 0.0
        local_epochs = int(config.get("local_epochs", self.num_epochs))
        local_epochs = max(1, min(local_epochs, self.num_epochs))
        for _ in range(local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                if mu > 0:
                    loss = loss + mu * _proximal_term(
                        self.model, global_ndarrays, self.device
                    )
                loss.backward()
                optimizer.step()
                num_samples += x.size(0)
                total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / num_samples if num_samples else 0.0
        return (
            get_parameters(self.model),
            num_samples,
            {"loss": avg_loss, "num_samples": num_samples},
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_sum += criterion(logits, y).item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        loss_avg = loss_sum / total if total else 0.0
        accuracy = correct / total if total else 0.0
        return loss_avg, total, {"accuracy": accuracy, "loss": loss_avg}


def client_fn(
    cid: str,
    dataset_name: str,
    batch_size: int = 10,
    num_epochs: int = 20,
    learning_rate: float = 0.01,
    data_root: Optional[str] = None,
) -> fl.client.NumPyClient:
    """Build a single client for Flower simulation."""
    from dataset import FEDPROX_DATA_ROOT

    data_root = data_root or FEDPROX_DATA_ROOT
    clients, train_data, test_data = load_fedprox_synthetic(dataset_name, data_root)
    if cid not in train_data:
        raise ValueError(f"Unknown client {cid}")
    train_loader, test_loader = client_dataloaders(
        cid, train_data, test_data, batch_size=batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MCLR().to(device)
    return FedProxClient(
        cid=cid,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
