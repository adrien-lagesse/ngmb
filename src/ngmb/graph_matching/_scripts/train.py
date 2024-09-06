import pathlib

import click
import ngmb


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    required=True,
    help="Path to A GMDataset",
)
@click.option("--experiment", type=str, required=True, help="Experiment name")
@click.option("--run-name", type=str, required=True, help="Run name")
@click.option("--epochs", type=int, required=True, help="Number of training epochs")
@click.option("--batch-size", type=int, required=True, help="Batch size")
@click.option("--cuda/--cpu", required=True, help="Training backend")
@click.option(
    "--log-frequency", type=int, required=True, help="Logging frequency in epoch number"
)
@click.option(
    "--profile/--no-profile",
    default=False,
    show_default=True,
    help="Logging profiling info",
)
@click.option(
    "--model",
    type=click.Choice(["GCN", "GIN", "GAT", "GatedGCN", "GATv2"]),
    required=True,
    help="Model type",
)
@click.option("--layers", type=int, help="Number of layers")
@click.option("--features", type=int, help="Number of features per layer")
@click.option("--heads", type=int, help="Number of attention heads (only for gat)")
@click.option(
    "--out-features", type=int, help="Number of output feautures of the model"
)
@click.option(
    "--optimizer",
    type=click.Choice(["adam", "adam-one-cycle"]),
    required=True,
    help="Optimizer",
)
@click.option("--lr", type=float, help="Learning rate (only for adam)")
@click.option(
    "--max-lr", type=float, help="Maximum learning rate (only for adam-one-cycle)"
)
@click.option(
    "--start-factor",
    type=float,
    help="One cycle start factor (only for adam-one-cycle)",
)
@click.option(
    "--end-factor", type=float, help="One cycle end factor (only for adam-one-cycle)"
)
@click.option("--grad-clip", type=float, help="Gradient clipping")
def graph_matching_siamese_train(**kwargs):
    def require_options(option, value, required_option: str | list[str]):
        if not isinstance(required_option, list):
            required_option = [required_option]
        for r in required_option:
            if kwargs[option] == value:
                if kwargs[r] is None:
                    raise click.BadOptionUsage(
                        r, f"'{r}' must be set for {option}={value}"
                    )

    require_options(
        "model", "GCN", required_option=["layers", "features", "out_features"]
    )
    require_options(
        "model", "GIN", required_option=["layers", "features", "out_features"]
    )
    require_options(
        "model", "GAT", required_option=["layers", "features", "heads", "out_features"]
    )
    require_options(
        "model", "GatedGCN", required_option=["layers", "features", "out_features"]
    )
    require_options(
        "model",
        "GATv2",
        required_option=["layers", "features", "heads", "out_features"],
    )

    require_options("optimizer", "adam", required_option=["lr", "grad_clip"])
    require_options(
        "optimizer",
        "adam-one-cycle",
        required_option=["max_lr", "start_factor", "end_factor", "grad_clip"],
    )

    ngmb.graph_matching.train(**kwargs)


def main():
    graph_matching_siamese_train()


if __name__ == "__main__":
    main()
