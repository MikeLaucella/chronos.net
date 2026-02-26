"""
chrono_set.py

Geodata dataset for extracting bounding box data from multiple temporal sources.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from chronos.data.geo_array import GeoArrayDataset
from chronos.data.models import BoundingBoxQuery


class ChronosCollator:
    """Collator class implementation"""

    def __init__(self, keep_query: bool = False):
        self.keep_query = keep_query

    def __call__(self, batch: list[dict]) -> dict:
        batched = {}
        query = []

        for data in batch:
            for key, value in data.items():
                if key == 'query':
                    continue
                if key not in batched:
                    batched[key] = []

                batched[key].append(value)

            if self.keep_query and 'query' in data:
                query.append(data['query'])

        out = {
            key: torch.stack(batched[key], dim=0)
            for key in batched.keys()
        }

        if self.keep_query:
            out['query'] = query

        return out


class ChronosDataset(Dataset[dict[str, Tensor]]):
    """Chronos dataset from array sources."""

    def __init__(self,
                 input_sets: dict[str, GeoArrayDataset],
                 transforms: any = None,
                 keep_query: bool = False):
        self.input_sets = input_sets
        self.transforms = transforms
        self.keep_query = keep_query
        self.length = self._verify_lengths()

    def _verify_lengths(self):
        lengths = {len(ds) for ds in self.input_sets.values()}
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent dataset lengths: {lengths}")

        return list(lengths)[0]

    def _get_item(self, query: BoundingBoxQuery):
        # get the image data
        input_data = {}
        for key, dataset in self.input_sets.items():
            input_data[key] = dataset[query]

        output = dict(**input_data)

        # apply transforms
        output = self.transforms(output) if self.transforms \
            else output

        if self.keep_query:
            output['query'] = query

        return output

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, query: BoundingBoxQuery):
        # route the data fetch based on mask presence in the data
        return self._get_item(query)
