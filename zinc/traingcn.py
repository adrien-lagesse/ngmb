import inspect
import statistics
from urllib.parse import unquote, urlparse

import dataloading
import numpy as np
import torch
import torch.utils
from accelerate import Accelerator
from dataloading import ZINCBatch
from safetensors.torch import load_model
from torch_geometric.nn import GATConv, GatedGraphConv
from torch_geometric.nn.pool import global_max_pool

from ngmb.models import GatedGCN


class GATModel(torch.nn.Module):
    def __init__(self, layers, heads, hidden_dim):
        super(GATModel, self).__init__()
        
        self.layers = layers
        self.node_embeddings = torch.nn.Embedding(28, hidden_dim)
        self.edge_embeddings = torch.nn.Embedding(4, 8)
        self.gat_layers = torch.nn.ModuleList()
        
        self.gat_layers.append(GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, edge_dim=8))
        
        for _ in range(layers - 2):
            self.gat_layers.append(GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, edge_dim=8))
        
        self.gat_layers.append(GATConv(in_channels=hidden_dim, out_channels=hidden_dim//heads, heads=heads, edge_dim=8))
        
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch: ZINCBatch):
        x = self.node_embeddings(batch.x)
        edge_attr = self.edge_embeddings(batch.edge_attr)
        for layer in self.gat_layers:
            x = layer(x, batch.edge_index, edge_attr=edge_attr)
            x = torch.relu(x)
        x = global_max_pool(x, batch.batch)
        out = self.fc(x).flatten()
        return out

class GatedGCNModel(torch.nn.Module):
    def __init__(self, layers, heads, hidden_dim):
        super(GatedGCNModel, self).__init__()
        
        self.layers = layers
        self.node_embeddings = torch.nn.Embedding(28, hidden_dim)
        self.edge_embeddings = torch.nn.Embedding(4, 8)
        self.gat_layers = torch.nn.ModuleList()
        
        self.gat_layers.append(GatedGraphConv(out_channels=hidden_dim,num_layers=2))
        
        for _ in range(layers - 2):
            self.gat_layers.append(GatedGraphConv(out_channels=hidden_dim,num_layers=2))
        
        self.gat_layers.append(GatedGraphConv(out_channels=hidden_dim,num_layers=2))
        
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, batch: ZINCBatch):
        x = self.node_embeddings(batch.x)
        for layer in self.gat_layers:
            x = layer(x, batch.edge_index)
            x = torch.relu(x)
        x = global_max_pool(x, batch.batch)
        out = self.fc(x).flatten()
        return out
    
class WarmupReduceLROnPlateau(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_lr, plateau_scheduler_args):
        """
        Combines a warmup phase with ReduceLROnPlateau scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            warmup_steps (int): Number of warmup steps.
            base_lr (float): Target learning rate after warmup.
            plateau_scheduler_args (dict): Arguments for ReduceLROnPlateau.
        """
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_count = 0
        self.warming_up = True
        self.reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **plateau_scheduler_args
        )
        super().__init__(optimizer)

    def step(self, metrics=None):
        """
        Updates the learning rate based on the current phase.

        Args:
            metrics (float): Metric to monitor (required after warmup).
        """
        if self.warming_up:
            self._warmup_step()
        else:
            if metrics is not None:
                self.reduce_on_plateau.step(metrics)

    def _warmup_step(self):
        """Handles the warmup phase."""
        self.step_count += 1
        new_lr = self.base_lr * (self.step_count / self.warmup_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        if self.step_count >= self.warmup_steps:
            self.warming_up = False

    def get_last_lr(self):
        """Returns the current learning rate."""
        return [group["lr"] for group in self.optimizer.param_groups]


def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != "self":
            kwargs[key] = values[key]
    return kwargs


def main(
    run_name: str,
    experiment_name: str,
    device: torch.device,
    layers: int,
    heads: int,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    epochs: int,
    dropout: float,
    reg: float,
):
        PE_MODEL = "/home/jlagesse/ngmb/mlruns/945853177701240299/9e38c977e3394244b63063797b00b62a/artifacts/checkpoint.safetensors"

        accelerator = Accelerator(log_with="mlflow")
        if accelerator.is_main_process:
            num_gpus = torch.cuda.device_count()

            # Print details of each available GPU
            for i in range(num_gpus):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            if num_gpus == 0:
                    print("No GPUs available.")
                
        gape_encoding_pcqm4mv2 = GatedGCN(4, 48, 32)
        load_model(
                gape_encoding_pcqm4mv2,
                PE_MODEL,
            )
        gape_encoding_pcqm4mv2 = gape_encoding_pcqm4mv2.to(device).eval()
        train_loader, val_loader = dataloading.setup_data(
                batch_size, 32, gape_encoding_pcqm4mv2, device
            )

        model = GATModel(
            layers=layers,
            heads=heads,
            hidden_dim=hidden_dim,
        )

        hparams = {**get_kwargs(), "pe-model": PE_MODEL, "nb_params": sum([np.prod(p.size()) for p in model.parameters()])}
        accelerator.init_trackers(experiment_name, init_kwargs= {"mlflow": {"run_name": run_name}}, config=hparams)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=reg)
        scheduler = WarmupReduceLROnPlateau(
            optimizer,
            warmup_steps=50,
            base_lr=lr,
            plateau_scheduler_args={"factor": 0.5, "min_lr": 1e-5, "patience": 50},
        )
        loss_fn = torch.nn.L1Loss()

        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

        for epoch in range(epochs):
            model.train()
            losses = []
            maes = []
            for _, batch in enumerate(train_loader):
                model.zero_grad()
                prediction = model.forward(batch)
                loss = loss_fn(prediction, batch.y)
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                losses.append(float(loss))
                maes.append(float((prediction - batch.y).abs().mean().detach()))
            scheduler.step(statistics.mean(losses))

            accelerator.log({"loss/train": statistics.mean(losses)}, epoch)

            model.eval()
            losses = []
            maes = []
            for i, batch in enumerate(val_loader):

                prediction = model.forward(batch)
                loss = loss_fn(prediction, batch.y)
                losses.append(float(loss))
                maes.append(float((prediction - batch.y).abs().mean()))
            accelerator.log({"loss/val": statistics.mean(losses)}, epoch)

            if epoch % 5 == 0:
                if accelerator.is_main_process:
                    run = accelerator.get_tracker("mlflow", unwrap=True)
                    checkpoint_path = unquote(
                        urlparse(
                            run.info.artifact_uri
                        ).path
                    )
                    print(f"Trying to log at {checkpoint_path} for epoch {epoch}")
                    accelerator.save_state(checkpoint_path, safe_serialization=True)
                
        accelerator.end_training()


if __name__ == "__main__":
    main(
        run_name="GAT",
        experiment_name="M-Transformer (ZINC)",
        device="cuda",
        layers=10,
        heads=4,
        hidden_dim=96,
        batch_size=32,
        lr=1e-3,
        epochs=2_000,
        dropout=0.025,
        reg=1e-6,
    )
