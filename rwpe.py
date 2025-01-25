import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import erdos_renyi_graph

# Step 1: Generate a random graph
num_nodes = 10
edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.3)  # 30% edge probability

# Create a graph data object
x = torch.randn(num_nodes, 1)  # Random node features (optional)
graph = Data(x=x, edge_index=edge_index)

print(f"Initial graph: {graph}")

# Step 2: Apply AddRandomWalkPE transform
rw_pe_transform = AddRandomWalkPE(walk_length=32)
graph_with_pe = rw_pe_transform(graph)

# Step 3: Positional encodings are now in graph.pos
print(f"Graph with positional encodings: {graph_with_pe}")
print(f"Positional encodings (graph.pos):\n{graph_with_pe.random_walk_pe}")