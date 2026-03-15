"""
MCLR model matching FedProx synthetic setup: input 60, output 10 (logistic regression).
"""
import numpy as np
import torch
import torch.nn as nn

from dataset import INPUT_DIM, NUM_CLASSES


class MCLR(nn.Module):
    """Logistic regression: 60 -> 10, with optional L2 via optimizer."""

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def get_parameters(model: nn.Module) -> list:
    """Return model parameters as list of numpy arrays (for Flower). Avoids .numpy() when bridge is broken."""
    out = []
    for p in model.parameters():
        t = p.detach().cpu()
        try:
            out.append(t.numpy().copy())
        except RuntimeError:
            out.append(np.array(t.tolist(), dtype=t.dtype if t.is_floating_point() else np.float32))
    return out


def set_parameters(model: nn.Module, parameters: list) -> None:
    """Set model parameters from list of numpy arrays."""
    for p, arr in zip(model.parameters(), parameters):
        p.data = torch.tensor(arr, device=p.device, dtype=p.dtype).clone()
