import torch

from ngmb._core import BatchedSignals, BatchedSparseGraphs


class LaplacianEmbeddings(torch.nn.Module):
    k: int

    def __init__(self, k: int) -> None:
        self.k = k

    @torch.no_grad
    def forward(
        self, batched_signals: BatchedSignals, batched_graphs: BatchedSparseGraphs
    ) -> BatchedSignals:
        signal_l: torch.FloatTensor = []
        for sparse_graph in batched_graphs:
            adj = sparse_graph.adj().float()
            degrees = torch.diag(adj.sum(dim=1).flatten())
            laplacian = torch.eye(len(degrees), device=degrees.device) - torch.sqrt(
                1.0 / (degrees + 1)
            ) @ adj @ torch.sqrt(1.0 / (degrees + 1))
            L, Q = torch.linalg.eigh(laplacian)
            if len(Q) < self.k:
                Q = torch.vstack(
                    [Q, torch.zeros((self.k - len(Q), len(Q)), device=Q.device)]
                )
            signal_l.append(Q[: self.k, :].T)
        return BatchedSignals(signals=torch.cat(signal_l), batch=batched_signals._batch)
