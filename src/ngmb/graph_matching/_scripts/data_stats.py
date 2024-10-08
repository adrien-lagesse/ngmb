import pathlib
import pprint
import statistics

import click
import torch

from ngmb.graph_matching import GMDataset, GMDatasetItem


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to A GMDataset",
)
@click.option("--text/--json", default=True, show_default=True, help="Output format")
def gm_dataset_stats(dataset: pathlib.Path, text: bool):
    def generate_stats(training: bool = True) -> tuple[int, dict[str, list[float]]]:
        data = GMDataset(dataset, validation=not training)
        nb_graphs = len(data)
        graph_stats: dict[str, list[float]] = {
            "base_order": [],
            "corrupted_order": [],
            "base_density": [],
            "corrupted_density": [],
            "correlation": [],
        }

        item: GMDatasetItem
        for item in data:
            graph_stats["base_order"].append(item.base_graph.order())
            graph_stats["corrupted_order"].append(item.corrupted_graph.order())
            graph_stats["base_density"].append(
                2 * item.base_graph.size() / item.base_graph.order()
            )
            graph_stats["corrupted_density"].append(
                2 * item.corrupted_graph.size() / item.corrupted_graph.order()
            )

            min_node = min(item.base_graph.order(), item.corrupted_graph.order())
            min_size = min(item.base_graph.size(), item.corrupted_graph.size())
            if min_size > 1:
                max_edges = min_node * (min_node - 1)
                d_common = (
                    item.base_graph.adj()[:min_node, :min_node]
                    * item.corrupted_graph.adj()[:min_node, :min_node]
                ).sum() / max_edges + 1e-7
                d1 = (
                    item.base_graph.adj()[:min_node, :min_node].sum() / max_edges + 1e-7
                )
                d2 = (
                    item.corrupted_graph.adj()[:min_node, :min_node].sum() / max_edges
                    + 1e-7
                )
                graph_stats["correlation"].append(
                    float(torch.abs(d_common - d1 * d2) / torch.sqrt(d1 * (1 - d1) * d2*(1 - d1)))
                )
            else:
                graph_stats["correlation"].append(1.0)
        return nb_graphs, graph_stats

    def print_text_stats(training: bool = True) -> None:
        nb_graphs, graphs_stats = generate_stats(training)

        print("=" * 50)
        print("=" * 19 + ("= TRAINING =" if training else " VALIDATION ") + "=" * 19)
        print("=" * 50)
        print(f"Number of graph pairs: {nb_graphs}")
        print(
            f"Average base graph order: {statistics.mean(graphs_stats['base_order'])}   (stdev = {statistics.stdev(graphs_stats['base_order'])}, min = {min(graphs_stats['base_order'])}, max = {max(graphs_stats['base_order'])}"
        )
        print(
            f"Average corrupted graph order: {statistics.mean(graphs_stats['corrupted_order'])}   (stdev = {statistics.stdev(graphs_stats['corrupted_order'])}, min = {min(graphs_stats['corrupted_order'])}, max = {max(graphs_stats['corrupted_order'])})"
        )
        print(
            f"Average base graph density: {statistics.mean(graphs_stats['base_density'])}   (stdev = {statistics.stdev(graphs_stats['base_density'])},, min = {min(graphs_stats['base_density'])}, max = {max(graphs_stats['base_density'])})"
        )
        print(
            f"Average corrupted graph density: {statistics.mean(graphs_stats['corrupted_density'])}   (stdev = {statistics.stdev(graphs_stats['corrupted_density'])}, min = {min(graphs_stats['corrupted_density'])}, max = {max(graphs_stats['corrupted_density'])})"
        )
        print(
            f"Average pair correlatioin: {statistics.mean(graphs_stats['correlation'])}   (stdev = {statistics.stdev(graphs_stats['correlation'])}, min = {min(graphs_stats['correlation'])}, max = {max(graphs_stats['correlation'])})"
        )

    def json_stats(training: bool = True) -> dict[str, float]:
        nb_graphs, graphs_stats = generate_stats(training)
        return {
            "nb_graphs": nb_graphs,
            "base_order_avg": statistics.mean(graphs_stats["base_order"]),
            "base_order_stdev": statistics.stdev(graphs_stats["base_order"]),
            "corrupted_order_avg": statistics.mean(graphs_stats["corrupted_order"]),
            "corrupted_order_stdev": statistics.stdev(graphs_stats["corrupted_order"]),
            "base_density_avg": statistics.mean(graphs_stats["base_density"]),
            "base_density_stdev": statistics.stdev(graphs_stats["base_density"]),
            "corrupted_density_avg": statistics.mean(graphs_stats["corrupted_density"]),
            "corrupted_density_stdev": statistics.stdev(
                graphs_stats["corrupted_density"]
            ),
            "correlation_avg": statistics.mean(graphs_stats["correlation"]),
            "correlation_stdev": statistics.stdev(graphs_stats["correlation"]),
        }

    if text:
        print_text_stats(training=True)
        print()
        print_text_stats(training=False)
        print()
    else:
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(
            {
                "training": json_stats(training=True),
                "validation": json_stats(training=False),
            }
        )


def main():
    gm_dataset_stats()
