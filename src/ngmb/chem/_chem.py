import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from ngmb._core import DenseGraph


def smiles_to_graph(smiles: str) -> tuple[DenseGraph, dict[int, tuple[float, float]]]:
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)
    AllChem.Compute2DCoords(mol)
    coordinates: dict[int, tuple[float, float]] = {}
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        coordinates[i] = (positions.x, positions.y)

    return DenseGraph(adjacency_matrix=torch.BoolTensor(adjacency_matrix)), coordinates
