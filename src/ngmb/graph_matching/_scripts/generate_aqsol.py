import os
import pathlib

import click
import torch
from ngmb import BatchedDenseGraphs, SparseGraph
from ngmb.random import bernoulli_corruption
from safetensors.torch import save_file
from torch_geometric.datasets import AQSOL
from tqdm.auto import tqdm


@click.command()
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=pathlib.Path,
    help="Path to the output directory",
)
@click.option(
    "--noise",
    required=True,
    type=float,
    help="Bernouilli noise corruption",
)
@click.option("--cuda/--cpu", default=False, show_default=True, help="Backend")
def graph_matching_aqsol(
    *,
    output_dir: str | os.PathLike,
    noise: float,
    cuda: bool,
):
    """Generate a Graph Matching Dataset by perturbating AQSOL molecular graphs"""

    AQSOL_ROOT = ".tmp/AQSOL"

    train_dataset = AQSOL(root=AQSOL_ROOT, split="train")
    validation_dataset = AQSOL(root=AQSOL_ROOT, split="val")

    os.makedirs(output_dir)

    def generate_and_save(sparse_dataset, prefix):
        orders_dict: dict[str, torch.LongTensor] = {}
        base_graphs_dict: dict[str, torch.LongTensor] = {}
        corrupted_graphs_dict: dict[str, torch.LongTensor] = {}

        device = torch.device("cuda" if cuda else "cpu")
        with device:
            sparse_graphs = [
                SparseGraph(
                    senders=data.edge_index[0].long(),
                    receivers=data.edge_index[1].long(),
                    order=data.num_nodes,
                ).to(device)
                for data in sparse_dataset
            ]

            for i, base_graph_sparse in tqdm(
                enumerate(sparse_graphs), total=len(sparse_graphs)
            ):
                orders_dict[str(i)] = torch.tensor(
                    [base_graph_sparse.order(), base_graph_sparse.order()],
                    dtype=torch.long,
                )
                base_graph_dense = base_graph_sparse.to_dense()

                base_graphs_dict[str(i)] = bernoulli_corruption(
                    BatchedDenseGraphs.from_graphs([base_graph_dense]),
                    noise,
                    type="node_normalized",
                    no_remove=True,
                )[0].edge_index()

                corrupted_graphs_dict[str(i)] = bernoulli_corruption(
                    BatchedDenseGraphs.from_graphs([base_graph_dense]),
                    noise,
                    type="node_normalized",
                    no_remove=True,
                )[0].edge_index()

        save_file(
            orders_dict,
            filename=os.path.join(output_dir, f"{prefix}-orders.safetensors"),
        )

        save_file(
            base_graphs_dict,
            filename=os.path.join(output_dir, f"{prefix}-base-graphs.safetensors"),
        )

        save_file(
            corrupted_graphs_dict,
            filename=os.path.join(output_dir, f"{prefix}-corrupted-graphs.safetensors"),
        )

    print()
    print("------ Generating the training dataset   ------")
    generate_and_save(train_dataset, prefix="train")
    print()
    print("------ Generating the validation dataset -----")
    generate_and_save(validation_dataset, prefix="val")


def main():
    graph_matching_aqsol()


if __name__ == "__main__":
    main()
