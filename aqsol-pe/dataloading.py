from typing import NamedTuple, Self

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data as PygData
from torch_geometric.datasets import AQSOL


class AqsolBatch(NamedTuple):
    x: torch.Tensor  # values between 0 and 64 representing atom type
    y: torch.Tensor
    edge_index: torch.LongTensor  # shape [2, num_edge_in_batch]
    node_mask: torch.BoolTensor  # shape [batch_len, max_nb_atom]
    batch: torch.LongTensor
    graph_batch: torch.LongTensor

    def to(self, device: torch.device) -> Self:
        return AqsolBatch(
            x=self.x.to(device),
            y=self.y.to(device),
            edge_index=self.edge_index.to(device),
            node_mask=self.node_mask.to(device),
            batch=self.batch.to(device),
            graph_batch=self.graph_batch.to(device),
        )

    def __len__(self) -> int:
        return len(self.node_mask)


def collate_fn(elems: list[PygData]) -> AqsolBatch:
    x = torch.cat([elem.x for elem in elems])
    y = torch.FloatTensor([elem.y for elem in elems])

    edge_index_l = []
    graph_batch_l = []
    inc = 0
    for i, elem in enumerate(elems):
        edge_index_l.append(elem.edge_index + inc)
        graph_batch_l.append(
            torch.full((len(elem.edge_index[0]),), i, device=x.device, dtype=torch.long)
        )
        inc += len(elem.x)

    edge_index = torch.cat(edge_index_l, dim=1)

    max_nodes = max([len(elem.x) for elem in elems])
    node_mask_l = [
        torch.BoolTensor(
            [True] * len(elem.x) + [False] * (max_nodes - len(elem.x)),
        )
        for elem in elems
    ]

    node_mask = torch.stack(node_mask_l)

    batch = torch.cat([torch.full_like(elem.x, i) for i, elem in enumerate(elems)])
    graph_batch = torch.cat(graph_batch_l)

    assert len(batch) == len(x), f"{x.shape}, {batch.shape}"

    return AqsolBatch(x, y, edge_index, node_mask, batch, graph_batch)


def setup_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    AQSOL_ROOT = ".tmp"
    train_dataset = AQSOL(root=AQSOL_ROOT, split="train")
    validation_dataset = AQSOL(root=AQSOL_ROOT, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )

    return train_loader, val_loader
