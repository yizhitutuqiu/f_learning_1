"""
Load FedProx synthetic data (JSON) and expose per-client train/test for Flower.
Data path: ../FedProx/data/{synthetic_iid|synthetic_0_0|synthetic_0.5_0.5|synthetic_1_1}/data/
"""
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


FEDPROX_DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "FedProx", "data"
)
INPUT_DIM = 60
NUM_CLASSES = 10


def load_fedprox_synthetic(
    dataset_name: str,
    data_root: str = FEDPROX_DATA_ROOT,
) -> Tuple[List[str], dict, dict]:
    """
    Load train/test user_data from FedProx synthetic JSON.
    Returns:
        clients: list of client ids (e.g. ['f_00000', ...])
        train_data: {cid: {'x': np.ndarray (n, 60), 'y': np.ndarray (n,)}}
        test_data: same structure
    """
    train_dir = os.path.join(data_root, dataset_name, "data", "train")
    test_dir = os.path.join(data_root, dataset_name, "data", "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"FedProx data not found. Expected dirs: {train_dir}, {test_dir}. "
            "Run from Federated_learning_1/f_learning and ensure FedProx data exists."
        )

    train_data = {}
    test_data = {}
    clients = []

    for dir_path, data_dict in [(train_dir, train_data), (test_dir, test_data)]:
        for f in os.listdir(dir_path):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(dir_path, f), "r") as inf:
                cdata = json.load(inf)
            clients.extend(cdata["users"])
            data_dict.update(cdata["user_data"])

    clients = sorted(set(clients))
    for c in clients:
        for data_dict in (train_data, test_data):
            if c not in data_dict:
                raise KeyError(f"Client {c} missing in data")
            data_dict[c]["x"] = np.array(data_dict[c]["x"], dtype=np.float32)
            data_dict[c]["y"] = np.array(data_dict[c]["y"], dtype=np.int64)

    return clients, train_data, test_data


def client_dataloaders(
    cid: str,
    train_data: dict,
    test_data: dict,
    batch_size: int = 10,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for one client."""
    x_tr = torch.tensor(train_data[cid]["x"], dtype=torch.float32)
    y_tr = torch.tensor(train_data[cid]["y"], dtype=torch.int64)
    x_te = torch.tensor(test_data[cid]["x"], dtype=torch.float32)
    y_te = torch.tensor(test_data[cid]["y"], dtype=torch.int64)

    train_ds = TensorDataset(x_tr, y_tr)
    test_ds = TensorDataset(x_te, y_te)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    return train_loader, test_loader


def get_global_test_loader(
    test_data: dict,
    clients: List[str],
    batch_size: int = 256,
) -> DataLoader:
    """Single DataLoader over all clients' test data for central evaluation."""
    xs, ys = [], []
    for c in clients:
        xs.append(test_data[c]["x"])
        ys.append(test_data[c]["y"])
    x = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(ys, axis=0), dtype=torch.int64)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_global_train_loader(
    train_data: dict,
    clients: List[str],
    batch_size: int = 256,
) -> DataLoader:
    """Single DataLoader over all clients' train data for central training loss (paper-style)."""
    xs, ys = [], []
    for c in clients:
        xs.append(train_data[c]["x"])
        ys.append(train_data[c]["y"])
    x = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(ys, axis=0), dtype=torch.int64)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
