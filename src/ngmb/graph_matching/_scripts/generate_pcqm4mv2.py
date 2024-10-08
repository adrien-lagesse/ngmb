import os
import pathlib
import random

import click
import torch
from ogb.lsc import PCQM4Mv2Dataset
from safetensors.torch import save_file

from ngmb import BatchedDenseGraphs
from ngmb.chem import smiles_to_graph
from ngmb.random import bernoulli_corruption


@click.command()
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=pathlib.Path,
    help="Path to the output directory",
)
@click.option("--n-graphs", required=True, type=int, help="Number of graphs")
@click.option(
    "--n-val-graphs", required=True, type=int, help="Number of validation graphs"
)
@click.option(
    "--noise",
    required=True,
    type=float,
    help="Bernouilli noise corruption",
)
@click.option("--cuda/--cpu", default=False, show_default=True, help="Backend")
def graph_matching_pcqm4mv2(
    *,
    output_dir: str | os.PathLike,
    n_graphs: int,
    n_val_graphs: int,
    noise: float,
    cuda: bool,
):
    """Generate a Graph Matching Dataset by perturbating PCQM4Mv2 molecular graphs"""

    PCQM4MV2_ROOT = ".tmp/PCQM4Mv2"

    full_dataset = PCQM4Mv2Dataset(root=PCQM4MV2_ROOT, only_smiles=True)
    idxs = random.sample(range(len(full_dataset)), n_graphs + n_val_graphs)
    random.shuffle(idxs)

    train_dataset = [
        smiles_to_graph(full_dataset[i][0])[0].to_sparse() for i in idxs[:n_graphs]
    ]
    validation_dataset = [
        smiles_to_graph(full_dataset[i][0])[0].to_sparse() for i in idxs[n_graphs:]
    ]

    os.makedirs(output_dir)

    def generate_and_save(sparse_dataset, prefix):
        orders_dict: dict[str, torch.LongTensor] = {}
        base_graphs_dict: dict[str, torch.LongTensor] = {}
        corrupted_graphs_dict: dict[str, torch.LongTensor] = {}

        device = torch.device("cuda" if cuda else "cpu")
        with device:
            sparse_graphs = [sparse_graph.to(device) for sparse_graph in sparse_dataset]

            for i, base_graph_sparse in enumerate(sparse_graphs):
                orders_dict[str(i)] = torch.tensor(
                    [base_graph_sparse.order(), base_graph_sparse.order()],
                    dtype=torch.long,
                )
                base_graph_dense = base_graph_sparse.to_dense()

                base_graphs_dict[str(i)] = base_graph_sparse.edge_index()

                corrupted_graphs_dict[str(i)] = bernoulli_corruption(
                    BatchedDenseGraphs.from_graphs([base_graph_dense]),
                    noise,
                    type="add",
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
    generate_and_save(train_dataset, prefix="train")
    print()
    generate_and_save(validation_dataset, prefix="val")


def main():
    graph_matching_pcqm4mv2()


if __name__ == "__main__":
    main()
