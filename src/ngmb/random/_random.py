"""
Module providing random operations linked to graphs
"""

import random
from typing import Literal

import torch

from ngmb._core import BatchedDenseGraphs, BatchedSparseGraphs, SparseGraph


def erdos_renyi(
    nb_graphs: int,
    order: int,
    p: float,
    *,
    directed: bool = False,
    self_loops: bool | None = False,
) -> BatchedDenseGraphs:
    """
    Generate a batch of random Erdös-Rényi graphs.

    ### Arguments:
    - nb_graphs: Number of graphs to generate.
    - order: number of nodes in each graph.
    - p: edge probability.
    - directed: if false the graph will be undirected.
    - self_loops: if None, there might be self loops, if False all self loops will be removed, if true they will be added.
    """

    assert 0.0 <= p <= 1, "'p' must be between 0 and 1"

    batch = torch.empty(size=(nb_graphs, order, order), dtype=torch.bool).bernoulli_(p)

    if not directed:
        tri_up = batch.triu(0)
        batch = tri_up | tri_up.transpose(1, 2)

    if self_loops is not None:
        idx = torch.arange(order)
        if self_loops:
            batch[:, idx, idx] = True
        else:
            batch[:, idx, idx] = False

    return BatchedDenseGraphs(batch, torch.full((nb_graphs,), order))


@torch.vmap
def __graph_normalization(adj_matrix: torch.Tensor, mask: torch.BoolTensor):
    """
    density/(1-density) with the same shape as adj_matrix
    """
    order = mask.sum()
    avg_degree = adj_matrix.masked_fill(mask.logical_not(), 0).float().sum() / (
        order - 1
    )
    degrees_matrix = torch.empty_like(adj_matrix, dtype=torch.float)
    degrees_matrix.fill_(avg_degree)
    return (degrees_matrix / (order - 1 - degrees_matrix)).nan_to_num(0, 1)


def bernoulli_corruption(
    batch: BatchedDenseGraphs,
    noise: float,
    *,
    directed: bool = False,
    self_loops: bool = False,
    type: Literal["add", "add_remove"],
) -> BatchedDenseGraphs:
    """
    Apply a Bernoulli corruption to each graph in the batch.

    ### Arguments:
    - batch: graph to corrupt
    - noise: amount of noise to apply
    - directed: if false, the perturbatation will be symmetric
    - self_loops: if false, will not add or remove self loops
    - type: wether to add and remove edges or just to add them.
    """

    assert 0.0 <= noise <= 1, "'noise' must be between 0 and 1"

    masks = batch.get_masks()

    stacked_adjacency_matrices = batch.get_stacked_adj()

    normalization_tensor = __graph_normalization(stacked_adjacency_matrices, masks)

    if type == "add_remove":
        edge_noise = torch.empty_like(
            stacked_adjacency_matrices, dtype=torch.bool
        ).bernoulli_(noise)
    if type == "add":
        edge_noise = torch.zeros_like(stacked_adjacency_matrices, dtype=torch.bool)

    nonedge_noise = torch.empty_like(
        stacked_adjacency_matrices, dtype=torch.bool
    ).bernoulli_(torch.clip(noise * normalization_tensor, 0, 1))

    if not directed:
        tri_up = edge_noise.triu()
        edge_noise = tri_up | tri_up.transpose(1, 2)

        tri_up = nonedge_noise.triu()
        nonedge_noise = tri_up | tri_up.transpose(1, 2)

    if not self_loops:
        idx = torch.arange(int(batch._orders.max()))
        edge_noise[:, idx, idx] = False
        nonedge_noise[:, idx, idx] = False

    corrupted_batch = stacked_adjacency_matrices.clone()
    corrupted_batch[stacked_adjacency_matrices & edge_noise] = False
    corrupted_batch[torch.logical_not(stacked_adjacency_matrices) & nonedge_noise] = (
        True
    )

    return BatchedDenseGraphs(
        corrupted_batch,
        batch.orders().clone(),
    )


def uniform_sub_sampling(
    graph: SparseGraph, n: int, num_nodes: int
) -> BatchedSparseGraphs:
    """
    Randomly sample num_nodes nodes from the graph and extract the graph spanning on those nodes.
    Repeat the process n times to build a batch.
    """
    order = graph.order()

    graphs_l: list[SparseGraph] = []
    for _ in range(n):
        sampled_indices = torch.LongTensor(random.sample(range(0, order), num_nodes))
        graphs_l.append(graph.node_sub_sample(sampled_indices))

    return BatchedSparseGraphs.from_graphs(graphs_l)


def bfs_sub_sampling(
    graph: SparseGraph, n: int, num_nodes: int, *, p: float = 1
) -> BatchedSparseGraphs:
    """
    Sample with the Breadth First Search Method num_nodes nodes from the graph and extract the graph spanning on those nodes.
    Repeat the process n times to build a batch.
    """
    (senders, receivers) = graph.edge_index()
    graphs_l: list[SparseGraph] = []

    for _ in range(n):
        base_node = random.randint(0, graph.order() - 1)
        kept_nodes: set[int] = {base_node}
        while len(kept_nodes) < num_nodes:
            new_nodes = set(
                receivers[
                    torch.isin(senders, torch.LongTensor(list(kept_nodes)))
                ].tolist()
            ).difference(kept_nodes)
            if len(new_nodes) == 0:
                kept_nodes.add(random.randint(0, graph.order() - 1))
            else:
                new_nodes = random.sample(
                    sorted(new_nodes), max(int(p * len(new_nodes)), 1)
                )
                if len(new_nodes) + len(kept_nodes) < num_nodes:
                    kept_nodes = kept_nodes.union(new_nodes)
                else:
                    kept_nodes = kept_nodes.union(
                        random.sample(sorted(new_nodes), num_nodes - len(kept_nodes))
                    )
        graphs_l.append(graph.node_sub_sample(torch.LongTensor(list(kept_nodes))))

    return BatchedSparseGraphs.from_graphs(graphs_l)
