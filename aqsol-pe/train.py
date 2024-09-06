import statistics
from typing import Literal

import architectures
import dataloading
import mlflow
import numpy as np
import torch
import torch.utils
from safetensors.torch import load_model
from torch_geometric.nn import GATConv, GCNConv, global_max_pool

from ngmb import BatchedSignals, BatchedSparseGraphs
from ngmb.models import GCN

DEVICE = "cuda"
EPOCHS = 500
FEATURES = 65
BATCH_SIZE = 100
LR = 5e-4
DROPOUT = 0
REG = 0.00001
MODEL: Literal[
    "MolGCN",
    "MolGAT",
    "MolTransformer",
    "MolTransformerGCN",
    "MolTransformerGCNPE",
    "MolTransformerMYPE",
    "MolTransformerGCNPE2",
    "MolGATMYPE",
] = "MolGATMYPE"

PE = "gcn8"
PE_DICT = {
    "gat18": "/home/jlagesse/gnnco/mlruns/448395276175764840/39e4f1f138a64a12a09a6c26c7a30e42/artifacts/checkpoint.safetensors",
    "gat12": "/home/jlagesse/gnnco/mlruns/448395276175764840/305d872a6da44d6c8d7aa652dcc8a9bc/artifacts/checkpoint.safetensors",
    "gat8": "/home/jlagesse/gnnco/mlruns/448395276175764840/d8a255872502432da96b6652a522910e/artifacts/checkpoint.safetensors",
    "gcn8": "/home/jlagesse/gnnco/mlruns/448395276175764840/472fdd197a424b7a90ff225a0bd810f7/artifacts/checkpoint.safetensors",
    "gcn12": "/home/jlagesse/gnnco/mlruns/448395276175764840/e872cc3e3d9d418dbf33f5ce47567710/artifacts/checkpoint.safetensors",
    "gcn18": "/home/jlagesse/gnnco/mlruns/448395276175764840/ced15932cfce4127a9c894f2883b1c91/artifacts/checkpoint.safetensors",
}

RUN_NAME = f"{PE}-large"


class MolGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.final_linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.conv1(x, batch.edge_index))
        x = torch.nn.functional.relu(self.conv2(x, batch.edge_index))
        x = self.dropout2(x)
        x = torch.nn.functional.relu(self.conv3(x, batch.edge_index))
        x = global_max_pool(x, batch.batch)
        return self.final_linear(x).flatten()


class MolGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv1 = GATConv(input_dim, hidden_dim // 8, 8)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 8, 8)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GATConv(hidden_dim, hidden_dim // 8, 8)
        self.conv4 = GATConv(hidden_dim, hidden_dim // 8, 8)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.conv5 = GATConv(hidden_dim, hidden_dim // 8, 8)

        self.final_linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.conv1(x, batch.edge_index))
        x = torch.nn.functional.relu(self.conv2(x, batch.edge_index))
        x = self.dropout2(x)
        x = torch.nn.functional.relu(self.conv3(x, batch.edge_index))
        x = torch.nn.functional.relu(self.conv4(x, batch.edge_index))
        x = self.dropout3(x)
        x = torch.nn.functional.relu(self.conv5(x, batch.edge_index))
        x = global_max_pool(x, batch.batch)
        return self.final_linear(x).flatten()


class MolTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)

        self.transformer = architectures.Transformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_heads=hidden_dim // 8,
            num_layers=5,
            d_ff=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)

        batch_len, max_num_atom = batch.node_mask.size()

        padded_sequences = torch.zeros(
            (batch_len, max_num_atom, x.shape[1]), device=x.device, dtype=torch.float
        )
        padded_sequences = padded_sequences.masked_scatter(
            batch.node_mask.unsqueeze(-1), x
        )  # batch_len,max_nb_atom, features_dim

        return self.transformer(
            padded_sequences, batch.node_mask.unsqueeze(-1).unsqueeze(1)
        )


class MolTransformerGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)

        self.conv1 = GCNConv(input_dim, input_dim)
        self.conv2 = GCNConv(input_dim, input_dim)

        self.transformer = architectures.Transformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_heads=hidden_dim // 8,
            num_layers=3,
            d_ff=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)

        x = torch.nn.functional.relu(self.conv1(x, batch.edge_index))
        x = self.conv2(x, batch.edge_index)

        batch_len, max_num_atom = batch.node_mask.size()

        padded_sequences = torch.zeros(
            (batch_len, max_num_atom, x.shape[1]), device=x.device, dtype=torch.float
        )
        padded_sequences = padded_sequences.masked_scatter(
            batch.node_mask.unsqueeze(-1), x
        )  # batch_len, max_nb_atom, features_dim

        return self.transformer(
            padded_sequences, batch.node_mask.unsqueeze(-1).unsqueeze(1)
        )


class MolTransformerGCNPE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)

        self.conv1 = GCNConv(1, input_dim)
        self.conv2 = GCNConv(input_dim, input_dim)

        self.transformer = architectures.Transformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_heads=hidden_dim // 8,
            num_layers=3,
            d_ff=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)

        ones = torch.ones((len(x), 1), device=x.device)
        pe = torch.nn.functional.relu(self.conv1(ones, batch.edge_index))
        pe = self.conv2(pe, batch.edge_index)

        x = x + pe

        batch_len, max_num_atom = batch.node_mask.size()

        padded_sequences = torch.zeros(
            (batch_len, max_num_atom, x.shape[1]), device=x.device, dtype=torch.float
        )
        padded_sequences = padded_sequences.masked_scatter(
            batch.node_mask.unsqueeze(-1), x
        )  # batch_len,max_nb_atom, features_dim

        return self.transformer(
            padded_sequences, batch.node_mask.unsqueeze(-1).unsqueeze(1)
        )


# encoding_model = GAT(3,8,64,32)
encoding_model = GCN(3, 64, 32)

load_model(encoding_model, PE_DICT[PE])
encoding_model = encoding_model.to(DEVICE)


class MolTransformerMYPE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0) -> None:
        super().__init__()

        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)

        self.transformer = architectures.Transformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_heads=hidden_dim // 8,
            num_layers=5,
            d_ff=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)

        batch_len = len(batch)
        max_nb_atom = int(batch.node_mask.shape[1])

        computed_batch = (
            torch.arange(batch.node_mask.numel(), device=x.device, dtype=torch.long)
            // max_nb_atom
        )[batch.node_mask.flatten()]
        input = BatchedSignals(
            torch.ones((computed_batch.numel(), 1), device=x.device), computed_batch
        )
        input_graphs = BatchedSparseGraphs(
            senders=batch.edge_index[0],
            receivers=batch.edge_index[1],
            batch=batch.graph_batch,
            orders=batch.node_mask.sum(dim=-1),
        )

        pe = encoding_model.forward(input, input_graphs).x()

        x = x + pe

        batch_len, max_num_atom = batch.node_mask.size()

        padded_sequences = torch.zeros(
            (batch_len, max_num_atom, x.shape[1]), device=x.device, dtype=torch.float
        )
        padded_sequences = padded_sequences.masked_scatter(
            batch.node_mask.unsqueeze(-1), x
        )  # batch_len,max_nb_atom, features_dim

        return self.transformer(
            padded_sequences, batch.node_mask.unsqueeze(-1).unsqueeze(1)
        )


class MolTransformerGCNPE2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)

        self.conv1 = GCNConv(input_dim, input_dim)
        self.conv2 = GCNConv(input_dim, input_dim)

        self.transformer = architectures.Transformer(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_heads=hidden_dim // 8,
            num_layers=6,
            d_ff=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)

        max_nb_atom = int(batch.node_mask.shape[1])

        computed_batch = (
            torch.arange(batch.node_mask.numel(), device=x.device, dtype=torch.long)
            // max_nb_atom
        )[batch.node_mask.flatten()]
        input = BatchedSignals(
            torch.ones((computed_batch.numel(), 1), device=x.device), computed_batch
        )
        input_graphs = BatchedSparseGraphs(
            senders=batch.edge_index[0],
            receivers=batch.edge_index[1],
            batch=batch.graph_batch,
            orders=batch.node_mask.sum(dim=-1),
        )

        pe = encoding_model.forward(input, input_graphs).x()
        pe = torch.nn.functional.relu(self.conv1(pe, batch.edge_index))
        pe = self.conv2(pe, batch.edge_index)

        x = x + pe

        batch_len, max_num_atom = batch.node_mask.size()

        padded_sequences = torch.zeros(
            (batch_len, max_num_atom, x.shape[1]), device=x.device, dtype=torch.float
        )
        padded_sequences = padded_sequences.masked_scatter(
            batch.node_mask.unsqueeze(-1), x
        )  # batch_len,max_nb_atom, features_dim

        return self.transformer(
            padded_sequences, batch.node_mask.unsqueeze(-1).unsqueeze(1)
        )


class MolGATMYPE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv1 = GATConv(input_dim, hidden_dim // 8, 8)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 8, 8)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GATConv(hidden_dim, hidden_dim // 8, 8)
        self.conv4 = GATConv(hidden_dim, hidden_dim // 8, 8)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.conv5 = GATConv(hidden_dim, hidden_dim // 8, 8)

        self.final_linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)

        max_nb_atom = int(batch.node_mask.shape[1])

        computed_batch = (
            torch.arange(batch.node_mask.numel(), device=x.device, dtype=torch.long)
            // max_nb_atom
        )[batch.node_mask.flatten()]
        input = BatchedSignals(
            torch.ones((computed_batch.numel(), 1), device=x.device), computed_batch
        )
        input_graphs = BatchedSparseGraphs(
            senders=batch.edge_index[0],
            receivers=batch.edge_index[1],
            batch=batch.graph_batch,
            orders=batch.node_mask.sum(dim=-1),
        )

        pe = encoding_model.forward(input, input_graphs).x()

        x = x + pe

        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.conv1(x, batch.edge_index))
        x = torch.nn.functional.relu(self.conv2(x, batch.edge_index))
        x = self.dropout2(x)
        x = torch.nn.functional.relu(self.conv3(x, batch.edge_index))
        x = torch.nn.functional.relu(self.conv4(x, batch.edge_index))
        x = self.dropout3(x)
        x = torch.nn.functional.relu(self.conv5(x, batch.edge_index))
        x = global_max_pool(x, batch.batch)
        return self.final_linear(x).flatten()


def main():
    mlflow.set_experiment(experiment_name="PE-GCN")

    with mlflow.start_run(run_name=RUN_NAME) as _run:
        mlflow.log_params(
            {
                "device": DEVICE,
                "epochs": EPOCHS,
                "features": FEATURES,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "dropout": DROPOUT,
                "reg": REG,
                "model": MODEL,
                "pe": PE,
            }
        )
        train_loader, val_loader = dataloading.setup_data(BATCH_SIZE)

        model = MolTransformerMYPE(32, 128, DROPOUT)
        model = model.to(DEVICE)

        mlflow.log_param(
            "nb_params", sum([np.prod(p.size()) for p in model.parameters()])
        )

        optimizer = torch.optim.Adam(list(model.parameters()), lr=1, weight_decay=REG)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, LR, EPOCHS)
        loss_fn = torch.nn.L1Loss()

        for epoch in range(EPOCHS):
            print("-" * 8 + str(epoch) + "-" * 8)
            model.train()
            losses = []
            maes = []
            for _, batch in enumerate(train_loader):
                batch = batch.to(DEVICE)
                model.zero_grad()

                prediction = model.forward(batch)
                loss = loss_fn(prediction, batch.y)
                losses.append(float(loss))
                maes.append(float((prediction - batch.y).abs().mean().detach()))
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
                optimizer.step()

            scheduler.step()

            mlflow.log_metric("loss/train", statistics.mean(losses), epoch)

            model.eval()
            losses = []
            maes = []
            for i, batch in enumerate(val_loader):
                batch = batch.to(DEVICE)

                prediction = model.forward(batch)
                loss = loss_fn(prediction, batch.y)
                losses.append(float(loss))
                maes.append(float((prediction - batch.y).abs().mean()))
            mlflow.log_metric("loss/val", statistics.mean(losses), epoch)


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        main()
