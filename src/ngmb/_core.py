"""
Core classes to simplify working with graphs (sparse and dense) and node signals.
"""

from typing import Self

import torch


class SparseGraph:
    """
    Sparse graph representation.
    """

    _senders: torch.LongTensor
    _receivers: torch.LongTensor
    _order: int

    def __init__(
        self, senders: torch.LongTensor, receivers: torch.LongTensor, order: int
    ) -> None:
        """
        Create a graph with vertices {0,...,order - 1} with the directed edges (senders[i], receivers[i]) for all i in len(senders) (=len(receivers)).
        """

        # Checking the integrity of the arguments
        assert (
            senders.dtype == torch.long and receivers.dtype == torch.long
        ), "'senders' and 'receivers' must be a torch.LongTensor"
        assert (
            senders.dim() == 1 and receivers.dim() == 1
        ), "'senders' and 'receivers' must be 1-dimensional"
        assert len(senders) == len(
            receivers
        ), "'senders' and 'receivers' must be of the same length"
        if len(senders) != 0:
            assert (
                int(torch.max(senders) + 1) <= order
                and int(torch.max(receivers) + 1) <= order
            ), "'senders' and 'receivers' refer to a node bigger than the order"
        assert (
            senders.device == receivers.device
        ), "'senders' and 'receivers' must be on the same device"

        self._senders = senders
        self._receivers = receivers
        self._order = order

    def to(self, device: torch.device) -> Self:
        """
        Move the data to the device.
        """
        return SparseGraph(
            senders=self._senders.to(device),
            receivers=self._receivers.to(device),
            order=self._order,
        )

    def device(self) -> torch.device:
        """
        Get the device on which the data is stored.
        """
        return self._senders.device

    def order(self) -> int:
        """
        Returns the number of vertices in the graph.
        """
        return self._order

    def size(self) -> float:
        """
        Returns the number of edges, directed edges count for 0.5 rather than 1 (for undirected edges).
        """
        return int(0.5 * len(self._senders))

    def to_dense(self) -> "DenseGraph":
        """
        Returns the dense representation of the graph.
        """
        adj = torch.full(
            (self._order, self._order), False, dtype=torch.bool, device=self.device()
        )
        adj[self._senders, self._receivers] = True
        return DenseGraph(adjacency_matrix=adj)

    def adj(self) -> torch.BoolTensor:
        """
        Returns the adjacency matrix of the dense representation of the graph.

        ### Warning:
        Needs to compute the dense representation first.
        """
        return self.to_dense().adj()

    def edge_index(self) -> torch.LongTensor:
        """
        Returns the edge index matrix of dim [2,num_edges].
        """
        return torch.vstack([self._senders, self._receivers])

    def to_batch(self) -> "BatchedSparseGraphs":
        """
        Returns a batch with only this graph inside
        """
        return BatchedSparseGraphs.from_graphs([self])

    def node_sub_sample(self, nodes_sample: torch.LongTensor) -> Self:
        """
        Returns the graph only containing the nodes in `nodes_sample`.
        """
        mask = torch.logical_and(
            torch.isin(self._senders, nodes_sample),
            torch.isin(self._receivers, nodes_sample),
        )

        new_senders = self._senders[mask]
        new_receivers = self._receivers[mask]

        for i, v in enumerate(torch.unique(torch.cat([new_senders, new_receivers]))):
            new_senders[new_senders == v] = i
            new_receivers[new_receivers == v] = i
        
        return SparseGraph(
            new_senders,
            new_receivers,
            max(int(torch.max(new_senders)), int(torch.max(new_receivers))) + 1,
        )


class BatchedSparseGraphs:
    """
    A class abstracting the batch representation of sparse graphs.
    It is the sparse representation of the union of all the graphs in the batch.
    """

    _senders: torch.LongTensor
    _receivers: torch.LongTensor
    _batch: torch.LongTensor
    _orders: torch.LongTensor

    @staticmethod
    def _check_independance(
        senders: torch.LongTensor,
        receivers: torch.LongTensor,
        batch: torch.LongTensor,
        orders: torch.LongTensor,
    ) -> bool:
        num_graphs = len(orders)
        cumsum_orders = orders.cumsum(dim=0)
        vec = torch.zeros((num_graphs + 1,), dtype=torch.long, device=orders.device)
        vec[1:] = cumsum_orders - 1
        for i in range(num_graphs):
            edge_indices = batch == i
            senders_graph_i = senders[edge_indices]
            receivers_graph_i = receivers[edge_indices]
            if not (
                bool(torch.all(int(vec[i]) <= senders_graph_i))
                and bool(torch.all(senders_graph_i <= int(vec[i + 1])))
                and bool(torch.all(int(vec[i]) <= receivers_graph_i))
                and bool(torch.all(receivers_graph_i <= int(vec[i + 1])))
            ):
                return False

        return True

    def __init__(
        self,
        *,
        senders: torch.LongTensor,
        receivers: torch.LongTensor,
        batch: torch.LongTensor,
        orders: torch.LongTensor,
        no_check: bool = False,
    ) -> None:
        """
        Create a graph with vertices {0,...,orders[0] - 1, ..., orders[0] +orders[1] -1,..., orders[0]+...+orders[n] -1} with
        the directed edges (senders[i], receivers[i]) for all i in len(senders) (=len(receivers)).

        ## Parameters
        - senders and receivers: one-dimensional tensor such that (senders[i], receivers[i]) is an edge for the union of all graphs in the batch
        - batch: one-dimensional tensor such that the edge (senders[i], receivers[i]) is an edge in the graph batch[i]
        - orders: one-dimensional tensor such that the graph i has orders[i] vertices.
        """

        # Checking the integrity of the arguments
        assert (
            senders.dtype == torch.long
            and receivers.dtype == torch.long
            and batch.dtype == torch.long
            and orders.dtype == torch.long
        ), "all arguments should have torch.long dtype"
        assert (
            senders.dim() == 1
            and receivers.dim() == 1
            and batch.dim() == 1
            and orders.dim() == 1
        ), "all arguments should have a dimension of 1"
        assert (
            int(torch.max(senders) + 1) <= int(orders.sum())
            and int(torch.max(receivers) + 1) <= int(orders.sum())
        ), "'senders' and 'receivers' refer to a node bigger than the batched graph order"
        assert (
            receivers.device == senders.device
            and batch.device == senders.device
            and orders.device == senders.device
        ), "all tensors must be on the same device"
        if not no_check:
            assert BatchedSparseGraphs._check_independance(
                senders, receivers, batch, orders
            ), "Error while checking the independence of each subgraph"

        self._senders = senders
        self._receivers = receivers
        self._batch = batch
        self._orders = orders

    def from_graphs(graphs: list[SparseGraph]) -> Self:
        """
        Build a batch from list of sparse graphs.
        """

        # Check if all graphs are on the same device
        def _check_device(device) -> None:
            assert device == graphs[0].device(), "all graphs must be on the same device"

        [_check_device(graph.device()) for graph in graphs]

        device = graphs[0].device()

        senders_l: list[torch.LongTensor] = []
        receivers_l: list[torch.LongTensor] = []
        batch_l: list[torch.LongTensor] = []
        orders_l: list[int] = []

        node_shift = 0
        for i, graph in enumerate(graphs):
            num_edge = len(graph._senders)

            senders_l.append(node_shift + graph._senders)
            receivers_l.append(node_shift + graph._receivers)
            batch_l.append(
                torch.tensor([i] * num_edge, dtype=torch.long, device=device)
            )
            orders_l.append(graph.order())

            node_shift += graph.order()

        return BatchedSparseGraphs(
            senders=torch.cat(senders_l),
            receivers=torch.cat(receivers_l),
            batch=torch.cat(batch_l),
            orders=torch.tensor(orders_l, dtype=torch.long, device=device),
            no_check=True,
        )

    def to(self, device: torch.device) -> Self:
        """
        Move the batch to the device.
        """
        return BatchedSparseGraphs(
            senders=self._senders.to(device),
            receivers=self._receivers.to(device),
            batch=self._batch.to(device),
            orders=self._orders.to(device),
        )

    def device(self) -> torch.device:
        """
        Get the device on which the data is stored.
        """
        return self._senders.device

    def __len__(self) -> int:
        """
        Returns the number of graphs in the batch.
        """
        return len(self._orders)

    def __getitem__(self, idx) -> SparseGraph:
        """
        Returns the 'idx' graph in the batch.
        """
        if idx > len(self) - 1:
            raise IndexError(f"{idx} is not present in batch of len {len(self)}")

        mask: torch.BoolTensor = self._batch == idx
        node_shift = int(torch.sum(self._orders[:idx]))
        return SparseGraph(
            senders=self._senders[mask] - node_shift,
            receivers=self._receivers[mask] - node_shift,
            order=int(self._orders[idx]),
        )

    def unbatch(self) -> list[SparseGraph]:
        """
        Unbatch the batch into a list of graphs.
        """
        unbatched_graphs: list[SparseGraph] = []
        for i in range(len(self)):
            unbatched_graphs.append(self[i])
        return unbatched_graphs

    def to_dense(self) -> "BatchedDenseGraphs":
        """
        Tranform the batch into a dense graph representation batch.
        """
        return BatchedDenseGraphs.from_graphs(
            list(map(lambda sparse_graph: sparse_graph.to_dense(), self.unbatch()))
        )

    def edge_index(self) -> torch.LongTensor:
        """
        Returns the edge_index tensor of dim [2, num_edges] (to be used with torch_geometric).
        """
        return torch.vstack([self._senders, self._receivers])

    def orders(self) -> torch.LongTensor:
        return self._orders

    def get_masks(self) -> torch.BoolTensor:
        """
        Return a mask where where masks[i] = [True*nb_node, False*(max_node - nb_node)]
        """
        masks = torch.arange(int(torch.max(self._orders))).repeat(len(self)).reshape(
            (len(self), -1)
        ) < self._orders.reshape((-1, 1))

        return masks


class DenseGraph:
    """
    Dense graph representation
    """

    _adj_matrix: torch.BoolTensor

    def __init__(self, adjacency_matrix: torch.BoolTensor) -> None:
        """
        Create a dense graph from an adjacency matrix.
        """
        assert (
            adjacency_matrix.dtype == torch.bool
        ), "'adjacency_matrix' dtype must be torch.bool"
        assert (
            adjacency_matrix.dim() == 2
            and adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        ), "'adjacency_matrix' must be a square matrix"
        self._adj_matrix = adjacency_matrix

    def to(self, device: torch.device) -> Self:
        """
        Move the graph to device.
        """
        return DenseGraph(adjacency_matrix=self._adj_matrix.to(device))

    def device(self) -> torch.device:
        """
        Get the device on which the data is stored.
        """
        return self._adj_matrix.device

    def order(self) -> int:
        """
        Returns the number of verices in the graph.
        """
        return int(self._adj_matrix.shape[0])

    def size(self) -> float:
        """
        Returns the number of edges, directed edges count for 0.5 rather than 1 (for undirected edges).
        """
        return float(0.5 * torch.count_nonzero(self._adj_matrix).float())

    def adj(self) -> torch.BoolTensor:
        """
        Returns the adjacency matrix of the dense representation of the graph.
        """
        return self._adj_matrix

    def edge_index(self) -> torch.LongTensor:
        """
        Returns the edge index matrix of dim [2,num_edges] (used in torch_geometric).

        ### Warning:
        Needs to compute the sparse representation of the graph.
        """
        return self.to_sparse().edge_index()

    def to_sparse(self) -> SparseGraph:
        """
        Returns the sparse representation of the graph.
        """
        senders, receivers = self._adj_matrix.nonzero(as_tuple=True)
        return SparseGraph(senders=senders, receivers=receivers, order=self.order())

    def to_batch(self) -> "BatchedDenseGraphs":
        """
        Returns a batch with only this graph inside
        """
        return BatchedDenseGraphs.from_graphs([self])


class BatchedDenseGraphs:
    """
    A class abstracting the batch representation of dense graphs in a single stacked tensor.
    """

    _stacked_adj_matrices: torch.BoolTensor
    _orders: torch.LongTensor

    def __init__(
        self, stacked_adjacency_matrices: torch.BoolTensor, orders: torch.LongTensor
    ) -> None:
        """
        Create a representation of a dense graph batch such that stacked_adjacency_matrices[i,:orders[i],:orders[i]]
        is the dense graph representation of the i-th graph in the batch
        """
        # Checking the integrity of the arguments
        assert (
            stacked_adjacency_matrices.dtype == torch.bool
        ), "'stacked_adjacency_matrix' dtype must be torch.bool"
        assert orders.dtype == torch.long, "'orders' dtype must be torch.long"
        assert (
            stacked_adjacency_matrices.dim() == 3
            and stacked_adjacency_matrices.shape[1]
            == stacked_adjacency_matrices.shape[2]
        ), "'stacked_adjacency_matrices' must have shape (batch_size, N,N)"
        assert stacked_adjacency_matrices.shape[1] == int(
            torch.max(orders)
        ), "the order of at least one graph is to big"
        assert stacked_adjacency_matrices.shape[0] == len(
            orders
        ), "'len(stacked_adjacency_matrices) is not equal to len(orders)"
        assert (
            stacked_adjacency_matrices.device == orders.device
        ), "all arguments must be on the same device"

        self._stacked_adj_matrices = stacked_adjacency_matrices
        self._orders = orders

    def from_graphs(graphs: list[DenseGraph]) -> Self:
        """
        Build a batch from list of dense graphs.
        """

        def _check_device(device) -> None:
            assert device == graphs[0].device(), "all graphs must be on the same device"

        [_check_device(graph.device()) for graph in graphs]

        device = graphs[0].device()
        orders = torch.tensor(
            list(map(lambda graph: graph.order(), graphs)),
            dtype=torch.long,
            device=device,
        )
        max_order = int(torch.max(orders))
        stacked_adj_matrices = torch.empty(
            (len(graphs), max_order, max_order), dtype=torch.bool, device=device
        )
        for i, graph in enumerate(graphs):
            order = graph.order()
            stacked_adj_matrices[i, :order, :order].copy_(graph.adj())

        return BatchedDenseGraphs(stacked_adj_matrices, orders)

    def to(self, device: torch.device) -> Self:
        """
        Move the batch to the device.
        """
        return BatchedDenseGraphs(
            stacked_adjacency_matrices=self._stacked_adj_matrices.to(device),
            orders=self._orders.to(device),
        )

    def device(self) -> torch.device:
        """
        Get the device on which the data is stored.
        """
        return self._stacked_adj_matrices.device

    def __len__(self) -> int:
        """
        Returns the number of graphs in the batch.
        """
        return len(self._orders)

    def __getitem__(self, idx) -> DenseGraph:
        """
        Returns the 'idx' graph in the batch.
        """
        if idx > len(self) - 1:
            raise IndexError(f"{idx} is not present in batch of len {len(self)}")

        order = self._orders[idx]
        return DenseGraph(
            adjacency_matrix=self._stacked_adj_matrices[idx, :order, :order]
        )

    def unbatch(self) -> list[DenseGraph]:
        """
        Unbatch the batch into a list of graphs.
        """
        unbatched_graphs: list[DenseGraph] = []
        for i in range(len(self)):
            unbatched_graphs.append(self[i])
        return unbatched_graphs

    def to_sparse(self) -> BatchedSparseGraphs:
        """
        Tranform the batch into a sparse graph representation batch.
        """
        return BatchedSparseGraphs.from_graphs(
            list(map(lambda dense_graph: dense_graph.to_sparse(), self.unbatch()))
        )

    def get_adj(self, i: int) -> torch.BoolTensor:
        """
        Get the adjacency matrix of the i-th graph in the batch
        """
        return self._stacked_adj_matrices[i, : self._orders[i], : self._orders[i]]

    def get_masks(self) -> torch.BoolTensor:
        """
        Return a mask where where masks[i] = [True*nb_node, False*(max_node - nb_node)]
        """
        masks = torch.arange(int(torch.max(self._orders))).repeat(len(self)).reshape(
            (len(self), -1)
        ) < self._orders.reshape((-1, 1))

        return masks

    def get_stacked_adj(self) -> torch.BoolTensor:
        """
        Return the stacked adjacencies matrices.
        ### Warning : the matrices are padded
        """
        return self._stacked_adj_matrices

    def orders(self) -> torch.LongTensor:
        return self._orders


class BatchedSignals:
    """
    Batch representation of a signal on the vertices of a graph
    """

    _signals: torch.Tensor
    _batch: torch.LongTensor

    def __init__(
        self,
        signals: torch.Tensor,
        batch: torch.LongTensor,
    ) -> None:
        """
        signals must be of the shape (total_num_vertices, signal_dimension) and batch must be of shape (total_num_vertices,) such that
        signals[batch == i] is the signal on the i-th graph of the batch
        """
        self._signals = signals
        self._batch = batch

    def to(self, device: torch.device) -> Self:
        """
        Move the batch to the device.
        """
        return BatchedSignals(self._signals.to(device), self._batch.to(device))

    def device(self) -> torch.device:
        """
        Get the device on which the data is stored.
        """
        return self._signals.device

    def from_signals(signals: list[torch.Tensor]) -> Self:
        """
        Build a batch from list of signals.
        """

        # Check if all graphs are on the same device
        def _check(tensor) -> None:
            assert (
                tensor.device == signals[0].device
            ), "all graphs must be on the same device"
            assert (
                tensor.dim() == 2 and tensor.shape[1] == signals[0].shape[1]
            ), "all graphs must have shape (_,signal_dim)"

        [_check(tensor) for tensor in signals]

        device = signals[0].device
        batch = torch.cat(
            [
                torch.tensor(
                    [i] * len(signals[i]),
                    dtype=torch.long,
                    device=device,
                )
                for i in range(len(signals))
            ]
        )
        signals = torch.cat(signals)
        return BatchedSignals(signals, batch)

    def __len__(self) -> int:
        """
        Returns the number of signals in the batch.
        """
        return int(torch.max(self._batch) + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the 'idx' signal in the batch.
        """
        if idx >= len(self):
            raise KeyError(f"{idx} is not in batch of length {len(self)}")
        return self._signals[self._batch == idx]

    def unbatch(self) -> list[torch.Tensor]:
        """
        Unbatch the batch into a list of signals.
        """
        return [self[i] for i in range(len(self))]

    def x(self) -> torch.Tensor:
        """
        Returns that tensor represention of the batch (to be used with torch_geometric).
        """
        return self._signals

    def dim(self) -> int:
        """
        Returns the signal dimension.
        """
        return self._signals.shape[-1]

    def force_stacking(self) -> torch.FloatTensor:
        return self._signals.reshape(len(self), -1, self.dim())
