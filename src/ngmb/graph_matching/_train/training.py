import pathlib
import statistics
import time
from typing import Literal, NamedTuple
from urllib.parse import unquote, urlparse

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.utils.data
from safetensors.torch import save_model
from scipy.optimize import linear_sum_assignment

from ngmb._core import BatchedSignals

from .utils import (
    GMDatasetBatch,
    build_visualization_batch,
    get_kwargs,
    model_factory,
    optimizer_factory,
    setup_data,
)

# @torch.compile
# def siamese_similarity(
#         model: torch.nn.Module, batch: GMDatasetBatch
# )-> torch.FloatTensor:
#     embeddings_base: BatchedSignals = model.forward(
#         batch.base_signals, batch.base_graphs
#     )

#     embeddings_corrupted: BatchedSignals = model.forward(
#         batch.corrupted_signals, batch.corrupted_graphs
#     )

#     alignement_similarities = torch.bmm(
#         embeddings_base.x().reshape((
#                 batch.base_node_masks.shape[0],
#                 batch.base_node_masks.shape[1],
#                 embeddings_base.dim(),
#             )),
#         embeddings_corrupted.x().reshape(
#             (
#                 batch.corrupted_node_masks.shape[0],
#                 batch.corrupted_node_masks.shape[1],
#                 embeddings_corrupted.dim(),
#             )
#         ).transpose(1, 2),
#     )

#     return alignement_similarities


# @torch.compile
def siamese_similarity(
    model: torch.nn.Module, batch: GMDatasetBatch
) -> torch.FloatTensor:
    embeddings_base: BatchedSignals = model.forward(
        batch.base_signals, batch.base_graphs
    )

    embeddings_corrupted: BatchedSignals = model.forward(
        batch.corrupted_signals, batch.corrupted_graphs
    )

    padded_base = torch.zeros(
        (batch.base_node_masks.numel(), embeddings_base.dim()),
        device=embeddings_base.device(),
        requires_grad=True,
    )

    padded_base = padded_base.masked_scatter(
        batch.base_node_masks.reshape(-1, 1), embeddings_base.x()
    )

    padded_corrupted = torch.zeros(
        (batch.corrupted_node_masks.numel(), embeddings_corrupted.dim()),
        device=embeddings_corrupted.device(),
        requires_grad=True,
    )
    padded_corrupted = padded_corrupted.masked_scatter(
        batch.corrupted_node_masks.reshape(-1, 1), embeddings_corrupted.x()
    )
    alignement_similarities = torch.bmm(
        padded_base.reshape(
            (
                batch.base_node_masks.shape[0],
                batch.base_node_masks.shape[1],
                embeddings_base.dim(),
            )
        ),
        padded_corrupted.reshape(
            (
                batch.corrupted_node_masks.shape[0],
                batch.corrupted_node_masks.shape[1],
                embeddings_corrupted.dim(),
            )
        ).transpose(1, 2),
    )

    return alignement_similarities


# def __compute_losses(
#     similarity_matrices: torch.FloatTensor, masks: torch.BoolTensor
# ) -> torch.FloatTensor:
#     diag_logits = torch.diagonal(torch.softmax(similarity_matrices, dim=-1), dim1=-2, dim2=-1)
#     losses = -torch.log(diag_logits + 1e-7).mean(dim=-1)
#     assert losses.shape == torch.Size([len(similarity_matrices),]), f"Wrong loss shape: {losses.shape}"
#     return losses


@torch.vmap
def __compute_losses(
    similarity_matrix: torch.FloatTensor, mask: torch.BoolTensor
) -> torch.FloatTensor:
    similarity_matrix.masked_fill_(torch.logical_not(mask), -float("inf"))
    diag_logits = torch.diag(torch.softmax(similarity_matrix, dim=1))
    diag_logits.masked_fill_(torch.logical_not(mask), 1)
    loss = -torch.log(diag_logits + 1e-7).mean()
    return loss


# @torch.compile
def compute_losses(
    similarity_matrices: torch.FloatTensor, masks: torch.BoolTensor
) -> torch.FloatTensor:
    """
    similarity_matrix: (batch_size, max_nb_node, max_nb_node)
    masks: (batch_size, max_nb_node) s.t masks[i] = [True*nb_node, False*(max_nb_node - nb_node)]
    """
    return __compute_losses(
        similarity_matrices,
        masks,
    )


class AccuraciesResults(NamedTuple):
    top1: torch.FloatTensor
    top3: torch.FloatTensor
    top5: torch.FloatTensor


# def __top_k_accuracy(
#     alignement_similarity: torch.FloatTensor, mask: torch.BoolTensor, top_n: int
# ) -> torch.FloatTensor:
#     _, indices = torch.sort(
#         torch.masked_fill(
#             alignement_similarity, torch.logical_not(mask), -float("inf")
#         ),
#         descending=True,
#     )
#     mask = mask.float()
#     m = (
#         torch.isin(torch.arange(len(alignement_similarity), device=alignement_similarity.device), indices[:, :top_n])
#         .float()
#         .squeeze()
#     )
#     acc = (m * mask).sum() / (mask.sum())
#     return acc


# @torch.no_grad
# def compute_accuracies(
#     alignement_similarities: torch.FloatTensor, masks: torch.BoolTensor
# ) -> AccuraciesResults:
#     batched_top_k_accuracy = torch.vmap(__top_k_accuracy, in_dims=(0, 0, None))
#     return AccuraciesResults(
#         top1=batched_top_k_accuracy(alignement_similarities, masks, 1),
#         top3=batched_top_k_accuracy(alignement_similarities, masks, 3),
#         top5=batched_top_k_accuracy(alignement_similarities, masks, 5),
#     )


def compute_accuracies(
    alignement_similarities: torch.FloatTensor, masks: torch.BoolTensor
) -> AccuraciesResults:
    top1 = torch.empty(
        (len(alignement_similarities),),
        dtype=torch.float,
        device=alignement_similarities.device,
    )
    top3 = torch.empty(
        (len(alignement_similarities),),
        dtype=torch.float,
        device=alignement_similarities.device,
    )
    top5 = torch.empty(
        (len(alignement_similarities),),
        dtype=torch.float,
        device=alignement_similarities.device,
    )
    for i, similarity_matrix in enumerate(alignement_similarities):
        similarity_matrix = similarity_matrix[masks[i]]
        _, indices = torch.sort(similarity_matrix, descending=True)

        top1_indices = indices[:, :1].detach().cpu()
        top1[i] = float(
            torch.isin(torch.arange(len(similarity_matrix)), top1_indices)
            .float()
            .mean()
        )

        top3_indices = indices[:, :3].detach().cpu()
        top3[i] = float(
            torch.isin(torch.arange(len(similarity_matrix)), top3_indices)
            .float()
            .mean()
        )

        top5_indices = indices[:, :5].detach().cpu()
        top5[i] = float(
            torch.isin(torch.arange(len(similarity_matrix)), top5_indices)
            .float()
            .mean()
        )

    return AccuraciesResults(top1=top1, top3=top3, top5=top5)


class LAPResults(NamedTuple):
    permutations: list[torch.LongTensor]
    lap: list[float]


def compute_lap(
    alignement_similarities: torch.FloatTensor, masks: torch.BoolTensor
) -> LAPResults:
    permuations = []
    lap = []
    for similarity_matrix, mask in zip(alignement_similarities, masks):
        similarity_matrix = (
            torch.softmax(similarity_matrix[mask], dim=-1).detach().cpu().numpy()
        )
        idx, permutation_pred = linear_sum_assignment(similarity_matrix, maximize=True)
        permuations.append(
            torch.tensor(
                permutation_pred,
                dtype=torch.long,
                device=alignement_similarities.device,
            )
        )
        lap.append(float((idx == permutation_pred).astype(float).mean()))

    return LAPResults(permutations=permuations, lap=lap)


def compute_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    metrics_l: dict[str, list[float]] = {
        "loss": [],
        "lap": [],
        "top_1": [],
        "top_3": [],
        "top_5": [],
    }

    batch: GMDatasetBatch
    for i, batch in enumerate(loader):
        batch = batch.to(device)

        similarity_matrices = siamese_similarity(model, batch)
        masks = (
            batch.base_node_masks
        )  # The algorithm doesn't work with different size graph pairs

        losses = compute_losses(similarity_matrices, masks)
        metrics_l["loss"].append(float(losses.mean()))

        (top_1, top_3, top_5) = compute_accuracies(similarity_matrices, masks)
        metrics_l["top_1"].append(float(top_1.mean()))
        metrics_l["top_3"].append(float(top_3.mean()))
        metrics_l["top_5"].append(float(top_5.mean()))

        (_permutations, lap) = compute_lap(similarity_matrices, masks)

        metrics_l["lap"].append(statistics.mean(lap))

    return {k: statistics.mean(v) for (k, v) in metrics_l.items()}


@torch.no_grad
def log_visualizations(
    run,
    model: torch.nn.Module,
    batch: GMDatasetBatch,
    prefix: Literal["train", "val"],
    device: torch.device,
    step: int,
):
    batch = batch.to(device)
    similarity_matrices = siamese_similarity(model, batch)

    for i in range(len(batch)):
        graph_order = batch.base_graphs[i].order()

        if step == 0:
            # dot_graph = compare_graphs(batch.base_graphs[i], batch.corrupted_graphs[i])
            # graph_path = unquote(
            #     urlparse(run.info.artifact_uri + f"{prefix}/graph[{i}]").path
            # )
            # dot_graph.render(
            #     "graph-comp",
            #     directory=graph_path,
            #     cleanup=True,
            #     format="svg",
            # )

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
        mlflow.log_figure(plt.gcf(), f"{prefix}/graph[{i}]/sim/{step}.png")

        plt.imshow(
            torch.softmax(similarity_matrices[i], dim=1)
            .float()
            .detach()
            .cpu()
            .numpy()[:graph_order, :graph_order]
        )
        mlflow.log_figure(plt.gcf(), f"{prefix}/graph[{i}]/softmax_sim/{step}.png")


def train(
    *,
    custom_model: torch.nn.Module | None = None,
    dataset: pathlib.Path,
    experiment: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    cuda: bool = True,
    log_frequency: int = 25,
    profile: bool = False,
    model: Literal["GCN", "GIN", "GAT", "GatedGCN", "GATv2"] | None,
    layers: int | None,
    heads: int | None,
    features: int | None,
    out_features: int | None,
    optimizer: Literal["adam", "adam-one-cycle"] = "adam-one-cycle",
    lr: float | None = 5e-4,
    max_lr: float | None = 1e-3,
    start_factor: int | None = 5,
    end_factor: int | None = 500,
    grad_clip: float = 1e-1,
):
    device = torch.device("cuda") if cuda else torch.device("cpu")

    mlflow.set_experiment(experiment_name=experiment)

    with mlflow.start_run(run_name=run_name, log_system_metrics=profile) as run:
        mlflow.log_params(get_kwargs())

        # Load the training and validation datasets and build suitable loaders to batch the graphs together.
        (train_dataset, val_dataset, train_loader, val_loader) = setup_data(
            dataset_path=dataset,
            batch_size=batch_size,
        )

        visualization_batch_train = build_visualization_batch(train_dataset, 1)
        visualization_batch_val = build_visualization_batch(val_dataset, 1)

        # Setting up the GNN model and loading it onto the gpu if needed
        gnn_model: torch.nn.Module
        if custom_model is not None:
            gnn_model = custom_model
        else:
            gnn_model = model_factory(
                model=model,
                layers=layers,
                heads=heads,
                features=features,
                out_features=out_features,
            )
        gnn_model = gnn_model.to(device)

        # Computing the number of parameters in the GNN
        mlflow.log_param(
            "nb_params", sum([np.prod(p.size()) for p in gnn_model.parameters()])
        )

        # Build the optimizer and scheduler
        gnn_optimizer, gnn_scheduler = optimizer_factory(
            gnn_model,
            optimizer=optimizer,
            epochs=epochs,
            lr=lr,
            max_lr=max_lr,
            start_factor=start_factor,
            end_factor=end_factor,
        )

        def forward_pass(gnn_model: torch.nn.Module, batch: GMDatasetBatch) -> float:
            similarity_matrices = siamese_similarity(gnn_model, batch)
            masks = batch.base_node_masks

            losses = compute_losses(similarity_matrices, masks)
            loss = losses.mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnn_model.parameters(), grad_clip)
            gnn_optimizer.step()
            return float(loss.data)

        for epoch in range(epochs):
            mlflow.log_metric("learning_rate", gnn_scheduler.get_last_lr()[0], epoch)

            # Logging
            logging_start_time = time.time()
            if epoch % log_frequency == 0:
                gnn_model.eval()
                train_metrics = {
                    f"{k}/train": v
                    for (k, v) in compute_metrics(
                        gnn_model, train_loader, device
                    ).items()
                }
                val_metrics = {
                    f"{k}/val": v
                    for (k, v) in compute_metrics(gnn_model, val_loader, device).items()
                }
                mlflow.log_metrics(train_metrics, epoch)
                mlflow.log_metrics(val_metrics, epoch)

                log_visualizations(
                    run, gnn_model, visualization_batch_train, "train", device, epoch
                )
                log_visualizations(
                    run, gnn_model, visualization_batch_val, "val", device, epoch
                )
            mlflow.log_metric("log_time", time.time() - logging_start_time, epoch)

            # Training loop
            training_start_time = time.time()
            gnn_model.train()
            batch: GMDatasetBatch
            for i, batch in enumerate(train_loader):
                batch = batch.to(device)

                gnn_model.zero_grad()

                loss = forward_pass(gnn_model, batch)

                mlflow.log_metric(
                    "minibatch_loss",
                    loss,
                    step=i + epoch * len(train_loader),
                )
            gnn_scheduler.step()
            mlflow.log_metric("train_time", time.time() - training_start_time, epoch)

        checkpoint_path = unquote(
            urlparse(run.info.artifact_uri + "/checkpoint.safetensors").path
        )
        save_model(gnn_model, checkpoint_path)
