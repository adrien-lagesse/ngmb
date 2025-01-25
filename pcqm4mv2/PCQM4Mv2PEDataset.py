import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import torch
from torch_geometric.data import Data, OnDiskDataset, download_url, extract_zip
from torch_geometric.data.data import BaseData
from torch_geometric.io import fs
from torch_geometric.utils import from_smiles as _from_smiles
from tqdm import tqdm

from ngmb import BatchedSignals, SparseGraph


def generate_pe(num_nodes, edge_index, model, device):
    signal = BatchedSignals.from_signals([torch.ones(num_nodes,1)]).to(device)
    graphs = SparseGraph(edge_index[0], edge_index[1], num_nodes).to_batch().to(device)
    pe = model.forward(signal, graphs).x().detach().cpu()
    assert pe.shape == torch.Size((num_nodes, 32)), f"Expected {(num_nodes, 32)}, got {pe.shape}"
    return pe

class PCQM4Mv2PEDataset(OnDiskDataset):
    url = ('https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/'
           'pcqm4m-v2.zip')

    split_mapping = {
        'train': 'train',
        'val': 'valid',
        'test': 'test-dev',
        'holdout': 'test-challenge',
    }

    def __init__(
        self,
        pe_dim: int,
        model, 
        device,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        backend: str = 'sqlite',
        from_smiles: Optional[Callable] = None,
    ) -> None:
        assert split in ['train', 'val', 'test', 'holdout']

        self.pe_dim = pe_dim
        self.model = model
        self.device = device

        schema = {
            'x': dict(dtype=torch.int64, size=(-1, 9)),
            'pe': dict(dtype=torch.float32, size=(-1, pe_dim)),
            'edge_index': dict(dtype=torch.int64, size=(2, -1)),
            'edge_attr': dict(dtype=torch.int64, size=(-1, 3)),
            'smiles': str,
            'y': float,
        }

        self.from_smiles = from_smiles or _from_smiles
        super().__init__(root, transform, backend=backend, schema=schema)

        split_idx = fs.torch_load(self.raw_paths[1])
        self._indices = split_idx[self.split_mapping[split]].tolist()

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('pcqm4m-v2', 'raw', 'data.csv.gz'),
            osp.join('pcqm4m-v2', 'split_dict.pt'),
        ]

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0])

        data_list: List[Data] = []
        iterator = enumerate(zip(df['smiles'], df['homolumogap']))
        for i, (smiles, y) in tqdm(iterator, total=len(df)):
            data = self.from_smiles(smiles)
            data.pe = generate_pe(data.x.shape[0], data.edge_index, self.model, self.device)
            data.y = y

            data_list.append(data)
            if i + 1 == len(df) or (i + 1) % 1000 == 0:  # Write batch-wise:
                self.extend(data_list)
                data_list = []

    def serialize(self, data: BaseData) -> Dict[str, Any]:
        assert isinstance(data, Data)
        return dict(
            x=data.x,
            pe=data.pe,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y,
            smiles=data.smiles,
        )

    def deserialize(self, data: Dict[str, Any]) -> Data:
        return Data.from_dict(data)