import os
import os.path
import pathlib

import click
import torch
from ngmb.random import bernoulli_corruption, erdos_renyi
from safetensors.torch import save_file


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
    "--order", required=True, type=int, help="Order of the graphs to generate"
)
@click.option(
    "--density", required=True, type=int, help="Density of the Erdos-Renyi graphs"
)
@click.option("--noise", required=True, type=float, help="Bernouilli noise corruption")
@click.option("--cuda/--cpu", default=False, show_default=True, help="Backend")
def graph_matching_erdos_renyi(
    *,
    output_dir: str | os.PathLike,
    n_graphs: int,
    n_val_graphs: int,
    order: int,
    density: int,
    noise: float,
    cuda: bool,
):
    """Generate a Graph Matching Dataset by perturbating Erdos-Renyi graphs"""
    os.makedirs(output_dir)

    def generate_and_save(N, prefix):
        save_file(
            {str(i): torch.tensor([order, order], dtype=torch.long) for i in range(N)},
            filename=os.path.join(output_dir, f"{prefix}-orders.safetensors"),
        )

        with torch.device("cuda" if cuda else "cpu"):
            p_edge = density / (order - 1)
            base_graphs = erdos_renyi(N, order, p_edge)
            corrupted_graphs = bernoulli_corruption(
                base_graphs, noise=noise, type="node_normalized"
            )

        save_file(
            {str(i): base_graphs[i].edge_index() for i in range(N)},
            filename=os.path.join(output_dir, f"{prefix}-base-graphs.safetensors"),
        )

        save_file(
            {str(i): corrupted_graphs[i].edge_index() for i in range(N)},
            filename=os.path.join(output_dir, f"{prefix}-corrupted-graphs.safetensors"),
        )

    generate_and_save(n_graphs, prefix="train")
    generate_and_save(n_val_graphs, prefix="val")


def main():
    graph_matching_erdos_renyi()
