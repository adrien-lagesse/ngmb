import torch
import torch.nn.functional as F
from torch_geometric.nn import WLConvContinuous

from ngmb._core import BatchedSignals, BatchedSparseGraphs


class WLContinuous(torch.nn.Module):
    def __init__(self, layers: int, out_features: int):
        super().__init__()
        assert layers > 0

        self.layers = torch.nn.ModuleList(
            [WLConvContinuous() for i in range(layers - 1)]
        )

    def forward(
        self, batched_signals: BatchedSignals, batched_graphs: BatchedSparseGraphs
    ) -> BatchedSignals:
        x = batched_signals.x()
        edge_index = batched_graphs.edge_index()
        x = F.relu(self.layer0(x, edge_index))
        for i in range(len(self.layers)):
            x = x + F.relu(self.layers[i](x, edge_index))

        x = self.linear(x)

        return BatchedSignals(x, batched_signals._batch)
