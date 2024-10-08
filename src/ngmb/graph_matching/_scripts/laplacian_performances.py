import pathlib
from typing import Literal

import click
import mlflow
import torch
from matplotlib import pyplot as plt

from ngmb.graph_matching import (
    GMDatasetBatch,
    compute_metrics,
    get_kwargs,
    setup_data,
    siamese_similarity,
)
from ngmb.graph_matching._train.utils import build_visualization_batch
from ngmb.models import LaplacianEmbeddings


def log_visualizations(
    run,
    model: torch.nn.Module,
    batch: GMDatasetBatch,
    prefix: Literal["train", "val"],
    device: torch.device,
):
    batch = batch.to(device)
    similarity_matrices = siamese_similarity(model, batch)

    for i in range(len(batch)):
        graph_order = batch.base_graphs[i].order()

        plt.imshow(batch.base_graphs[i].adj().float().detach().cpu().numpy())
        mlflow.log_figure(plt.gcf(), f"{prefix}/graph[{i}]/adj.png")

        plt.imshow(
            torch.logical_xor(
                batch.base_graphs[i].adj(), batch.corrupted_graphs[i].adj()
            )
            .float()
            .detach()
            .cpu()
            .numpy()
        )
        mlflow.log_figure(plt.gcf(), f"{prefix}/graph[{i}]/diff_adj.png")

        plt.imshow(
            similarity_matrices[i]
            .float()
            .detach()
            .cpu()
            .numpy()[:graph_order, :graph_order]
        )
        mlflow.log_figure(plt.gcf(), f"{prefix}/graph[{i}]/sim.png")

        plt.imshow(
            torch.softmax(similarity_matrices[i], dim=1)
            .float()
            .detach()
            .cpu()
            .numpy()[:graph_order, :graph_order]
        )
        mlflow.log_figure(plt.gcf(), f"{prefix}/graph[{i}]/softmax_sim.png")


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to A GMDataset",
)
@click.option("--experiment", type=str, required=True, help="Experiment name")
@click.option("--run-name", type=str, required=True, help="Run name")
@click.option(
    "--features", type=int, required=True, help="Number of features per layer"
)
@click.option("--batch-size", type=int, required=True, help="Batch size")
@click.option("--cuda/--cpu", required=True, help="Training backend")
def compute_laplacian_performances(
    dataset: pathlib.Path,
    experiment: str,
    run_name: str,
    features: int,
    batch_size: int,
    cuda: bool,
):
    device = torch.device("cuda") if cuda else torch.device("cpu")

    mlflow.set_experiment(experiment_name=experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(get_kwargs())

        # Load the training and validation datasets and build suitable loaders to batch the graphs together.
        (train_dataset, val_dataset, train_loader, val_loader) = setup_data(
            dataset_path=dataset,
            batch_size=batch_size,
        )

        # visualization_batch_train = build_visualization_batch(train_dataset, 1)
        # visualization_batch_val = build_visualization_batch(val_dataset, 1)

        gnn_model: torch.nn.Module = LaplacianEmbeddings(k=features)

        train_metrics = {
            f"{k}/train": v
            for (k, v) in compute_metrics(gnn_model, train_loader, device).items()
        }

        val_metrics = {
            f"{k}/val": v
            for (k, v) in compute_metrics(gnn_model, val_loader, device).items()
        }

        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)

        # log_visualizations(run, gnn_model, visualization_batch_train, "train", device)
        # log_visualizations(run, gnn_model, visualization_batch_val, "val", device)


def main():
    compute_laplacian_performances()


if __name__ == "__main__":
    main()
