import os
import pathlib

import click
import torch
from ngmb import BatchedDenseGraphs, SparseGraph
from ngmb.random import bernoulli_corruption, bfs_sub_sampling
from ogb.nodeproppred import PygNodePropPredDataset
from safetensors.torch import save_file
from tqdm.auto import tqdm


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
    "--noise",
    required=True,
    type=float,
    help="Bernouilli noise corruption",
)
@click.option("--cuda/--cpu", default=False, show_default=True, help="Backend")
def graph_matching_ogbn_arxiv(
    *,
    output_dir: str | os.PathLike,
    n_graphs: int,
    n_val_graphs: int,
    order: int,
    noise: float,
    cuda: bool,
):
    """Generate a Graph Matching Dataset by perturbating the OGBN-Arxiv graph"""

    OGBNARXIV_ROOT = ".tmp/OGBN-ARXIV"
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=OGBNARXIV_ROOT)
    edge_index = dataset[0].edge_index
    num_nodes = int(torch.max(dataset.edge_index) + 1)

    os.makedirs(output_dir)

    def generate_and_save(N, prefix):
        orders_dict: dict[str, torch.LongTensor] = {}
        base_graphs_dict: dict[str, torch.LongTensor] = {}
        corrupted_graphs_dict: dict[str, torch.LongTensor] = {}

        device = torch.device("cuda" if cuda else "cpu")
        with device:
            corafull_graph = SparseGraph(
                senders=edge_index[0].long(),
                receivers=edge_index[1].long(),
                order=num_nodes,
            ).to(device)

            subsampled_graphs = bfs_sub_sampling(corafull_graph, N, order)

            for i in tqdm(range(N), total=N):
                base_graph_sparse = subsampled_graphs[i]

                base_graphs_dict[str(i)] = base_graph_sparse.edge_index()

                orders_dict[str(i)] = torch.tensor(
                    [base_graph_sparse.order(), base_graph_sparse.order()],
                    dtype=torch.long,
                )

                base_graph_dense = base_graph_sparse.to_dense()

                corrupted_graph_dense = bernoulli_corruption(
                    BatchedDenseGraphs.from_graphs([base_graph_dense]),
                    noise,
                    type="node_normalized",
                )[0]
                corrupted_graphs_dict[str(i)] = corrupted_graph_dense.edge_index()

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
    generate_and_save(n_graphs, prefix="train")
    print()
    print("------ Generating the validation dataset -----")
    generate_and_save(n_val_graphs, prefix="val")


def main():
    graph_matching_ogbn_arxiv()


if __name__ == "__main__":
    main()
