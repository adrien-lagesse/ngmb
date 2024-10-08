import itertools
import random
import time

import click

import ngmb


def dataset(name, noise):
    if name == "ER":
        return f"/scratch/jlagesse/ngmb-data/ER[100,8,{noise}]"
    if name == "PCQM4Mv2":
        return f"/scratch/jlagesse/ngmb-data/PCQM4Mv2[{noise}]"
    if name == "OGBN-Arxiv":
        return f"/scratch/jlagesse/ngmb-data/OGBN-Arxiv[100,{noise}]"
    if name == "AQSOL":
        return f"/scratch/jlagesse/ngmb-data/AQSOL[{noise}]"
    if name == "CoraFull":
        return f"/scratch/jlagesse/ngmb-data/CoraFull[100,{noise}]"

def outlayer(db_name):
    if db_name == "ER":
        return 64
    if db_name == "PCQM4Mv2":
        return 32
    if db_name == "OGBN-Arxiv":
        return 64
    if db_name == "AQSOL":
        return 32
    if db_name == "CoraFull":
        return 64
    
def dim(model):
    if model == "GCN":
        return 128
    if model == "GIN":
        return 93
    if model == "GatedGCN":
        return 48
    if model == "GAT":
        return 128
    if model == "GATv2":
        return 96

def batch_size(db_name):
    if db_name == "PCQM4Mv2":
        return 1000
    else:
        return 100
    
datasets = ["ER", "PCQM4Mv2", "OGBN-Arxiv", "AQSOL", "CoraFull"]
models = ["GCN", "GIN", "GatedGCN", "GAT", "GATv2"]
noises = [0.04, 0.06, 0.08, 0.12, 0.15, 0.18, 0.24, 0.3]

configs = list(itertools.product(datasets, models, noises))

@click.command()
@click.option('-i')
def work(i: int):
    time.sleep(random.randint(5,45))
    data, model, noise = configs[int(i)]
    ngmb.graph_matching.train(
        dataset= dataset(data, noise),
        experiment=f"{data}-RUN7",
        run_name=f"{model}-{noise}",
        epochs=300,
        batch_size=batch_size(data),
        cuda=True,
        log_frequency=25,
        profile=True,
        model=model,
        layers=4,
        features=dim(model),
        heads=8,
        out_features=outlayer(data),
        optimizer="adam-one-cycle",
        max_lr=3e-3,
        start_factor=5,
        end_factor=500,
        grad_clip=0.1
    )


if __name__ == "__main__":
    work()


