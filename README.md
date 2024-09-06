# Benchmarking GNNs by aligning graphs

The `ngmb` (**Noisy Graph Matching Benchmark**) package simplifies benchmarking GNNs on the graph alignement task (graph matching) with correlated pairs of graphs.
It is based on PyTorch and the default models are written using the Pytorch Geometric package.

Several functionalities are provided:
- Generating Graph Matching Datasets (from synthetic data or pre-existing datasets used by the GNN community).
- A set of pre-generated and fixed Graph Matching Datasets.
- A framework to benchmark different GNNs architecture against Graph Matching Datasets.
- Using GNNs trained on the benchmark datasets to generate high quality Graph Positional Encodings.

# Graph Matching problem for benchmarking

## Dataset generation

We provide several command line application to generate graph matching datasets:

- **gm-generate-er** : Generate Erdos-Renyi GM datasets.
- **gm-generate-aqsol** : Generate GM datasets based on the AQSOL dataset
- **gm-generate-karateclub** : Generate GM datasets based on the KarateClub Benchmark dataset.
- **gm-generate-corafull** : Generate GM datasets based on the CoraFull Benchmark dataset.
- **gm-generate-ogbn-arxiv** : Generate GM datasets based on the OGBN-Arxiv Benchmark dataset.
- **gm-generate-pcqm4mv2** : Generate GM datasets based on the PCQM4Mv2 Benchmark dataset.

To know more about them run:

`gm-generate-er --help`

`gm-generate-aqsol --help`

`gm-generate-karateclub --help`

`gm-generate-corafull --help`

`gm-generate-ogbn-arxiv --help`

`gm-generate-pcqm4mv2 --help`


Once you have a dataset, you can print key statistics with `gm-data-stats`

## Training

### Architectures in the library (`ngmb.models`)

Use the `gm-train` command line tool to train a Siamese Graph Matching model. (run `gm-train --help` for more information and see `scripts/train-siamese-gm.sh` for an example).

### Custom architectures

Use the API.


# Running the Repo

We use [Rye](https://rye.astral.sh/) to manage the python project. See the documentation for a complete guide.

### Quick installation (Linux and MacOS)

`curl -sSf https://rye.astral.sh/get | bash`

`echo 'source "$HOME/.rye/env"' >> ~/.profile    # For Bash`

`echo 'source "$HOME/.rye/env"' >> ~/.zprofile   # For ZSH`

You may have to restart you shell.

### Cloning the repo

`git clone https://github.com/adrien-lagesse/ngmb.git`

`cd ngmb`

`rye sync`

`rye list`

You sould have a list of all the dependencies of the project.



