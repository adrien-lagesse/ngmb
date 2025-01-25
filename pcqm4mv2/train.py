import inspect
import statistics
from urllib.parse import unquote, urlparse

import architectures
import dataloading
import numpy as np
import torch
import torch.utils
from accelerate import Accelerator
from safetensors.torch import load_model

from ngmb.models import GatedGCN

torch._dynamo.config.optimize_ddp = False

class MoleculeTransformer(torch.nn.Module):
    def __init__(self, input_dim, pe_dim, layers, heads, hidden_dim, dropout) -> None:
        super().__init__()

        self.l1 = torch.nn.Linear(input_dim, pe_dim)

        self.transformer = architectures.Transformer(
                input_dim=pe_dim,
                d_model=hidden_dim,
                num_heads=heads,
                num_layers=layers,
                d_ff=hidden_dim,
                dropout=dropout,
            )
        
        # self.transformer = torch.compile(self.transformer, fullgraph=True, backend="cudagraphs")

    def forward(self, batch) -> torch.Tensor:
        batch_len = len(batch)
        x = self.l1(batch.x) + batch.pe

        batch_len, max_num_atom = batch.node_mask.size()

        padded_sequences = torch.zeros(
            (batch_len, max_num_atom, x.shape[1]), device=x.device, dtype=torch.float
        )
        padded_sequences = padded_sequences.masked_scatter(
            batch.node_mask.unsqueeze(-1), x
        )

        return self.transformer(
            padded_sequences, batch.node_mask.unsqueeze(-1).unsqueeze(1)
        )


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

        model = MoleculeTransformer(
            input_dim=9,
            pe_dim=32,
            layers=layers,
            heads=heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        hparams = {**get_kwargs(), "pe-model": PE_MODEL, "nb_params": sum([np.prod(p.size()) for p in model.parameters()])}
        accelerator.init_trackers(experiment_name, init_kwargs= {"mlflow": {"run_name": run_name}}, config=hparams)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=1, weight_decay=reg)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs)
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
            scheduler.step()

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
        run_name="M-Transformer",
        experiment_name="M-Transformer (PCQM4Mv2)",
        device="cuda",
        layers=32,
        heads=16,
        hidden_dim=320,
        batch_size=100,
        lr=1e-4,
        epochs=900,
        dropout=0,
        reg=0.00001,
    )
