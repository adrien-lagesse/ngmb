import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GraphNorm

from ngmb._core import BatchedSignals, BatchedSparseGraphs

GN_FREQ = 3


class GAT(torch.nn.Module):
    def __init__(self, layers: int, heads: int, features: int, out_features: int):
        super().__init__()
        assert layers > 0

        self.layer0 = GCNConv(in_channels=1, out_channels=features)
        self.layers = torch.nn.ModuleList(
            [
                GATConv(
                    features, features // heads, heads=heads, bias=(i % GN_FREQ == 0)
                )
                for i in range(layers - 1)
            ]
        )
        self.gns = torch.nn.ModuleList(
            [GraphNorm(features) for _ in range(layers // GN_FREQ + 1)]
        )
        self.linear = torch.nn.Linear(features, out_features)

    def forward(
        self, batched_signals: BatchedSignals, batched_graphs: BatchedSparseGraphs
    ) -> BatchedSignals:
        x = batched_signals.x()
        edge_index = batched_graphs.edge_index()
        x = F.relu(self.layer0(x, edge_index))
        for i in range(len(self.layers)):
            if i % GN_FREQ == 0:
                x = self.gns[i // GN_FREQ](x)
            x = x + F.relu(self.layers[i](x, edge_index))

        x = self.linear(x)

        return BatchedSignals(x, batched_signals._batch)
