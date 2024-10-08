"""
Module defining the GMDataset and GMDatasetItem classes to iteract with a Graph Matching Dataset.
"""

import os
import os.path
from typing import NamedTuple, Self, override

import torch.utils.data
from safetensors.torch import load_file

from ngmb._core import SparseGraph


class GMDatasetItem(NamedTuple):
    base_graph: SparseGraph
    corrupted_graph: SparseGraph


class GMDataset(torch.utils.data.Dataset):
    """
    Graph Matching Dataset abstraction class.
    """

    single_base_graph: bool

    base_graphs: dict[int, SparseGraph]
    corrupted_graphs: dict[int, SparseGraph]

    def __init__(
        self,
        root: str | os.PathLike,
        *,
        validation: bool = False,
    ) -> None:
        """
        Load a Graph Mathching Dataset.

        ## Arguments:
        - root: Path to the dataset
        - validation: if True, will load the validation dataset rather than the training dataset.
        """
        super().__init__()
        prefix = "val" if validation else "train"
        try:
            orders = {
                int(k): v
                for k, v in load_file(
                    os.path.join(root, f"{prefix}-orders.safetensors")
                ).items()
            }
            self.base_graphs = {
                int(k): SparseGraph(
                    senders=v[0], receivers=v[1], order=int(orders[int(k)][0])
                )
                for k, v in load_file(
                    os.path.join(root, f"{prefix}-base-graphs.safetensors")
                ).items()
            }
            self.corrupted_graphs = {
                int(k): SparseGraph(
                    senders=v[0], receivers=v[1], order=int(orders[int(k)][1])
                )
                for k, v in load_file(
                    os.path.join(root, f"{prefix}-corrupted-graphs.safetensors")
                ).items()
            }

            self.single_base_graph = not (len(self.base_graphs) > 1)

        except:  # noqa: E722
            raise RuntimeError("Unable to load database")

    @override
    def __len__(self) -> int:
        """
        Number of GMDatasetItem into the dataset.
        """
        return len(self.corrupted_graphs)

    @override
    def __getitem__(self, index) -> GMDatasetItem:
        """
        Return the index-th GMDatasetItem in the dataset.
        """

        if self.single_base_graph:
            return GMDatasetItem(
                self.base_graphs[0],
                self.corrupted_graphs[index],
            )
        else:
            return GMDatasetItem(
                self.base_graphs[index],
                self.corrupted_graphs[index],
            )

    @override
    def __iter__(self) -> Self:
        self.iter_index = 0
        return self

    @override
    def __next__(
        self,
    ) -> GMDatasetItem:
        if self.iter_index < len(self):
            res = self[self.iter_index]
            self.iter_index += 1
            return res
        else:
            raise StopIteration

    @override
    def __repr__(self) -> str:
        return f"GMDataset({len(self)})"
