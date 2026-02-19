"""
geo_sampler.py

Geodata samplers for extracting bounding box queries from geodata sources.
"""

from abc import abstractmethod
from typing import Generator

import torch
import numpy as np
from torch.utils.data import Sampler

from chronos.data.models import BoundingBoxQuery


class TileSampler(Sampler):
    """A tile based sampler for the first pass of step selections."""

    def __init__(self, metadata, steps_per_epoch, entropy_weight=0.3):
        """
        metadata: list of dicts with 'centroid' and 'entropy'
        steps_per_epoch: number of samples per epoch
        entropy_weight: how much to bias toward high-entropy tiles
        """
        self.steps = steps_per_epoch

        # spatial uniformity: cluster centroids into bins
        centroids = np.array([m["centroid"] for m in metadata])
        xs, ys = centroids[:,0], centroids[:,1]

        # simple spatial bins (4x4 grid)
        x_bins = np.digitize(xs, np.quantile(xs, np.linspace(0,1,5)))
        y_bins = np.digitize(ys, np.quantile(ys, np.linspace(0,1,5)))

        # compute spatial bin IDs
        bin_ids = x_bins * 10 + y_bins

        # compute weights
        ent = np.array([m["entropy"] for m in metadata])
        ent = (ent - ent.min()) / (ent.max() - ent.min() + 1e-6)

        # combine spatial uniformity + entropy
        # each bin gets equal mass, entropy redistributes within bin
        weights = np.zeros(len(metadata))
        for b in np.unique(bin_ids):
            idx = np.where(bin_ids == b)[0]
            w = ent[idx] * entropy_weight + (1 - entropy_weight)
            w = w / w.sum()
            weights[idx] = w / len(np.unique(bin_ids))

        self.weights = torch.tensor(weights, dtype=torch.float)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.steps, replacement=True).tolist())

    def __len__(self):
        return self.steps


class WindowSampler:
    """A window sampler that samples windows within a tile."""

    @abstractmethod
    def next(self, tile_id: int):
        """Given a tile ID, sample a window within that tile."""
        pass


class RandomUnfiformWindowSampler(WindowSampler):
    def __init__(
        self,
        height,
        width,
        window=512
    ):
        """
        tile_sampler: another Sampler that yields tile_ids
        tile_shapes: list of (H, W) per tile
        boundary_dists: list of distance maps or None
        """
        self.height = height
        self.width = width
        self.window = window

    def _sample_uniform(self, H, W):
        y = np.random.randint(0, H - self.window)
        x = np.random.randint(0, W - self.window)
        return y, x

    def next(self, _: int):
        y, x = self._sample_uniform(self.height, self.width)
        return [y, y+self.window, x, x+self.window]


class BoundaryDistWindowSampler(WindowSampler):
    """A window sampler that samples more frequently near boundaries."""

    def __init__(self, boundary_dists, window=512, boundary_ratio=0.3):
        """
        tile_sampler: another Sampler that yields tile_ids
        tile_shapes: list of (H, W) per tile
        boundary_dists: list of distance maps or None
        """
        self.boundary_dists = boundary_dists
        self.window = window
        self.boundary_ratio = boundary_ratio

    def _sample_boundary(self, dist_map):
        H, W = dist_map.shape
        flat = dist_map.flatten().astype(np.float32)
        p = np.exp(-flat / 5.0)
        p /= p.sum()

        idx = np.random.choice(len(flat), p=p)
        y0, x0 = divmod(idx, W)

        y = max(0, min(y0 - self.window // 2, H - self.window))
        x = max(0, min(x0 - self.window // 2, W - self.window))
        return y, x

    def next(self, tile_id: int):
        y, x = self._sample_boundary(self.boundary_dists[tile_id])
        return [y, y+self.window, x, x+self.window]


class TrainingSampler(Sampler):
    """Sample a number of epoch step queries from tiles"""

    def __init__(self, tile_sampler: TileSampler, window_sampler: WindowSampler):
        super().__init__()
        self.tile_sampler = tile_sampler
        self.window_sampler = window_sampler

    def _generate_queries(self) -> Generator[BoundingBoxQuery, None, None]:
        for tile in self.tile_sampler:
            window = self.window_sampler.next(tile)
            yield BoundingBoxQuery(
                index=tile,
                miny=window[0],
                maxy=window[1],
                minx=window[2],
                maxx=window[3],
            )

    def __iter__(self):
        return iter(self._generate_queries())

    def __len__(self):
        return self.steps


class TestingSampler(Sampler):
    """Sample a number of epoch step queries from tiles"""

    def __init__(self, tile_metadata, window_size=512):
        super().__init__()
        self.tile_metadata = tile_metadata
        self.window_size = window_size

    def _generate_queries(self) -> Generator[BoundingBoxQuery, None, None]:
        for tile in self.tile_metadata:
            H, W = tile['shape']
            for y in range(0, H, self.window_size):
                for x in range(0, W, self.window_size):
                    window = [y, min(y+self.window_size, H), x, min(x+self.window_size, W)]
                    yield BoundingBoxQuery(
                        index=tile['index'],
                        miny=window[0],
                        maxy=window[1],
                        minx=window[2],
                        maxx=window[3],
                    )

    def __iter__(self):
        return iter(self._generate_queries())

    def __len__(self):
        return self.steps


class GeoSamplerBuilder:
    """Builder to create ChronosSampler with different strategies."""

    def __init__(self,
                 window_size=512,
                 boundary_ratio=0.3,
                 entropy_weight=0.3,
                 steps_per_epoch=1000,
                 boundary_dists=None):
        self.window_size = window_size
        self.boundary_ratio = boundary_ratio
        self.entropy_weight = entropy_weight
        self.steps_per_epoch = steps_per_epoch
        self.boundary_dists = boundary_dists

    def _validate(self):
        if self.boundary_dists is None:
            raise ValueError("Boundary distance maps must be provided")

    def testing(self, tiles):
        return TestingSampler(tiles, window_size=self.window_size)

    def validation(self, tiles):
        return TestingSampler(tiles, window_size=self.window_size)

    def training(self, tiles) -> Sampler:
        self._validate()
        tile_sampler = TileSampler(
            tiles,
            steps_per_epoch=self.steps_per_epoch,
            entropy_weight=self.entropy_weight)

        window_sampler = BoundaryDistWindowSampler(
            boundary_dists=self.boundary_dists,
            window=self.window_size,
            boundary_ratio=self.boundary_ratio)

        return TrainingSampler(tile_sampler, window_sampler)
