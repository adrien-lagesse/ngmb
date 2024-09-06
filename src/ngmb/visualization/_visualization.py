import graphviz
import matplotlib.colors as clr
import torch
from matplotlib import colormaps

from ngmb._core import DenseGraph, SparseGraph


def plot_graph(
    graph: SparseGraph | DenseGraph,
    *,
    size: float = 8,
    coordinates: dict[int, tuple[float, float]] | None = None,
) -> graphviz.Graph:
    dot = graphviz.Graph(strict=True)
    dot.graph_attr = {"size": str(size), "layout": "neato", "overlap": "scale"}
    dot.node_attr = {
        "label": "",
        "shape": "circle",
        "style": "filled",
        "width": "0.2",
        "fixedsize": "true",
        "fillcolor": "black",
    }
    if coordinates is None:
        [dot.node(str(i)) for i in range(graph.order())]
    else:
        [
            dot.node(str(i), pos=f"{coordinates[i][0]},{coordinates[i][1]}!")
            for i in range(graph.order())
        ]

    [
        dot.edge(str(i), str(j))
        for i, j in zip(graph.edge_index()[0].tolist(), graph.edge_index()[1].tolist())
    ]
    return dot


def plot_similarities(
    graph: SparseGraph | DenseGraph,
    node: int,
    similarity_matix: torch.Tensor,
    *,
    size: float = 8,
    coordinates: dict[int, tuple[float, float]] | None = None,
) -> graphviz.Graph:
    softmax_matrix = (similarity_matix - similarity_matix[node].min()) / (
        similarity_matix[node].max() - similarity_matix[node].min()
    )
    # colormap = clr.LinearSegmentedColormap.from_list(
    #     "similarity_cm", ["#a9d6e5", "#012a4a"]
    # )
    colormap = colormaps.get("viridis")
    dot = graphviz.Graph(strict=True)
    dot.graph_attr = {"size": str(size), "layout": "neato"}
    dot.node_attr = {
        "shape": "circle",
        "label": "",
        "style": "filled",
    }
    if coordinates is None:
        [
            dot.node(
                str(i), fillcolor=clr.to_hex(colormap(float(softmax_matrix[node, i])))
            )
            for i in range(graph.order())
        ]
    else:
        [
            dot.node(
                str(i),
                fillcolor=clr.to_hex(colormap(float(softmax_matrix[node, i]))),
                pos=f"{coordinates[i][0]},{coordinates[i][1]}!",
            )
            for i in range(graph.order())
        ]
    [
        dot.edge(str(i), str(j))
        for i, j in zip(graph.edge_index()[0].tolist(), graph.edge_index()[1].tolist())
    ]
    return dot


def compare_graphs(
    graph1: DenseGraph | SparseGraph,
    graph2: DenseGraph | SparseGraph,
    *,
    size: float = 8,
    coordinates: dict[int, tuple[float, float]] | None = None,
) -> graphviz.Graph:
    dot = graphviz.Graph(strict=True)
    if coordinates is None:
        dot.graph_attr = {"size": str(size)}
    else:
        dot.graph_attr = {"size": str(size), "layout": "neato"}

    dot.node_attr = {
        "label": "",
        "shape": "circle",
        "style": "filled",
        "width": "0.2",
        "fixedsize": "true",
    }
    dot.edge_attr = {"weight": "15"}
    max_order = max(graph1.order(), graph2.order())

    for i in range(max_order):
        if i >= graph1.order():
            node_color = "green"
        elif i >= graph2.order():
            node_color = "red"
        else:
            node_color = "black"

        pos = ""
        if (coordinates is not None) and (i in coordinates.keys()):
            pos = f"{coordinates[i][0]},{coordinates[i][1]}!"

        dot.node(str(i), fillcolor=node_color, pos=pos)

    adj1 = graph1.adj()
    adj2 = graph2.adj()
    for i in range(max_order):
        for j in range(max_order):
            try:
                if bool(adj1[i, j]) and bool(adj2[i, j]):
                    edge_color = "black"
                    dot.edge(str(i), str(j), color=edge_color)
            except:  # noqa: E722
                pass

            try:
                if bool(adj1[i, j]) and not bool(adj2[i, j]):
                    edge_color = "red"
                    dot.edge(str(i), str(j), color=edge_color)
            except:  # noqa: E722
                pass

            try:
                if not bool(adj1[i, j]) and bool(adj2[i, j]):
                    edge_color = "green"
                    dot.edge(str(i), str(j), color=edge_color)
            except:  # noqa: E722
                pass

    return dot
