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
from ngmb.models import GCN, GatedGCN, LaplacianEmbeddings

DEVICE = "cuda"
EPOCHS = 250
FEATURES = 65
BATCH_SIZE = 100
LR = 2e-4
DROPOUT = 0
REG = 0.00001


class MolGCN(torch.nn.Module):
    def __init__(self, input_dim=32, hidden_dim=200, dropout=0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = torch.nn.Embedding(65, input_dim, max_norm=1)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)

        self.final_linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        x = self.embeddings(batch.x)
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.conv1(x, batch.edge_index))
        x = torch.nn.functional.relu(self.conv2(x, batch.edge_index))
        x = self.dropout2(x)
        x = torch.nn.functional.relu(self.conv3(x, batch.edge_index))
        x = torch.nn.functional.relu(self.conv4(x, batch.edge_index))
        x = self.dropout2(x)
        x = torch.nn.functional.relu(self.conv5(x, batch.edge_index))
        x = global_max_pool(x, batch.batch)
        return self.final_linear(x).flatten()


class MolGAT(torch.nn.Module):
    def __init__(self, input_dim=32, hidden_dim=200, dropout=0) -> None:
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
    def __init__(self, input_dim=32, hidden_dim=72, dropout=0) -> None:
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

gape_encoding_model = GatedGCN(4, 48, 32)
load_model(gape_encoding_model, "/home/jlagesse/ngmb/mlruns/288153219095938292/9159de11d66a49e38274c6ef517890f4/artifacts/checkpoint.safetensors")
gape_encoding_model  = gape_encoding_model.to(DEVICE).eval()

laplacian_encoding_model = LaplacianEmbeddings(32)


class MolTransformerLAPE(torch.nn.Module):
    def __init__(self, input_dim=32, hidden_dim=72, dropout=0) -> None:
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

        pe = laplacian_encoding_model.forward(input, input_graphs).x()

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


class MolTransformerGAPE(torch.nn.Module):
    def __init__(self, input_dim=32, hidden_dim=72, dropout=0) -> None:
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

        pe = gape_encoding_model.forward(input, input_graphs).x()

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



def main(model_name):
    mlflow.set_experiment(experiment_name="AQSOL-PE-RUN10")

    with mlflow.start_run(run_name=model_name) as _run:
        mlflow.log_params(
            {
                "device": DEVICE,
                "epochs": EPOCHS,
                "features": FEATURES,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "dropout": DROPOUT,
                "reg": REG,
            }
        )
        train_loader, val_loader = dataloading.setup_data(BATCH_SIZE)

        if model_name == "GCN":
            model = MolGCN()
        if model_name == "GAT":
            model = MolGAT()
        if model_name == "Transformer":
            model = MolTransformer()
        if model_name == "Transformer-LAPE":
            model = MolTransformerLAPE()
        if model_name == "Transformer-GAPE":
            model = MolTransformerGAPE()
        
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

models = ["GCN", "GAT", "Transformer", "Transformer-LAPE", "Transformer-GAPE"]
if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        for model in models:
            print("Running Model: {model}")
            main(model)
