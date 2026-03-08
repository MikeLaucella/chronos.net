"""
geo_sampler.py

Geodata samplers for extracting bounding box queries from geodata sources.
"""

from abc import abstractmethod
from collections import Counter
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
        empty = np.array([m["percent_empty"] for m in metadata])
        ent_norm = ent / (ent.max() + 1e-6)
        empty_norm = empty / (empty.max() + 1e-6)

        score = 0.5 * ent_norm + 0.5 * (1 - empty_norm)
        self.p = score / (score.sum() + 1e-6)
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
        #return iter(torch.multinomial(self.weights, self.steps, replacement=True).tolist())
        return iter(np.random.choice(len(self.p), self.steps, p=self.p, replace=True))

    def __len__(self):
        return self.steps


class WindowSampler:
    """A window sampler that samples windows within a tile."""

    @abstractmethod
    def next(self, tile_id: int):
        """Given a tile ID, sample a window within that tile."""
        pass

    @abstractmethod
    def next(self, tile_id: int, count: int):
        """Given a tile ID and count, sample a window within that tile."""
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

    def next(self, tile_id: int):
        return self.next(tile_id, 1)

    def next(self, _: int, count: int):
        windows = []
        for _ in range(count):
            y, x = self._sample_uniform(self.height, self.width)
            windows.append([y, y+self.window, x, x+self.window])

        return windows


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

    def _sample_valid_window(self, valid_small, H, W, window, factor=8):
        # 1. Get the list of 'True' indices from the 1/8th scale map
        # (e.g., if the map is 875x750, these are coordinates in that range)
        ys, xs = np.where(valid_small > 0)
        
        if len(ys) == 0:
            # Fallback: if no data exists, pick a random crop in the image
            return np.random.randint(0, H - window), np.random.randint(0, W - window)

        # 2. Pick a random anchor
        idx = np.random.randint(len(ys))
        
        # 3. SCALE UP the anchor to full resolution
        # anchor_y and anchor_x are now in the 0-7000 range
        anchor_y = ys[idx] * factor
        anchor_x = xs[idx] * factor
        
        # 4. Add your Jitter (H//2 or W//2 of the WINDOW size)
        # This fills in the gaps between the 8px steps and adds variety
        half_win = window // 2
        y_start = anchor_y - half_win + np.random.randint(-128, 128)
        x_start = anchor_x - half_win + np.random.randint(-128, 128)
        
        # 5. Final Safety Clamp
        # This prevents the 512x512 window from 'falling off' the image edges
        y_final = np.clip(y_start, 0, H - window)
        x_final = np.clip(x_start, 0, W - window)
        
        return int(y_final), int(x_final)

    def next(self, tile_id: int):
        y, x = self._sample_valid_window(self.boundary_dists[tile_id][:], 7000, 6000, 8)
        return [y, y+self.window, x, x+self.window]

    def next(self, tile_id: int, count: int):
        windows = []
        boundaries = self.boundary_dists[tile_id][:]

        for _ in range(count):
            y, x = self._sample_valid_window(boundaries, 7000, 6000, self.window)
            windows.append([y, y+self.window, x, x+self.window])

        return windows

class SlidingSampler(Sampler):
    """Sample a number of epoch step queries from tiles"""

    def __init__(self, tile_sampler: TileSampler, window_sampler: WindowSampler):
        super().__init__()
        self.tile_sampler = tile_sampler
        self.window_sampler = window_sampler

    def _generate_queries(self):
        tiles = list(self.tile_sampler)
        tiles = Counter(tiles)

        windows = []
        for tile, count in tiles.items():
            sampled_windows = self.window_sampler.next(tile, count)
            for window in sampled_windows:
                windows.append(BoundingBoxQuery(
                    index=tile,
                    miny=window[0],
                    maxy=window[1],
                    minx=window[2],
                    maxx=window[3],
                ))

        # shuffle the windows to mix tiles together
        np.random.shuffle(windows)
        return iter(windows)

    def __iter__(self):
        return iter(self._generate_queries())

    def __len__(self):
        return len(self.tile_sampler)


class FixedGridSampler(Sampler):
    """Sample a number of epoch step queries from tiles"""

    def __init__(self, tile_metadata, boundaries, window_size=512):
        super().__init__()
        self.tile_metadata = tile_metadata
        self.window_size = window_size
        self.boundaries = boundaries
        self.factor = 8
        self._queries = list(self._generate_queries())

    def _generate_queries(self):
        for tile in self.tile_metadata:
            coords = self.boundaries[tile['index']][:] 
            H, W = tile['shape']

            step = self.window_size // self.factor
            H_small = coords.shape[0]
            W_small = coords.shape[1]
            y_grid = np.arange(0, H_small - step + 1, step)
            x_grid = np.arange(0, W_small - step + 1, step)

            for y_s in y_grid:
                for x_s in x_grid:
                    if coords[y_s, x_s]:
                        y = int(y_s * self.factor)
                        x = int(x_s * self.factor)
                        y_start = int(max(0, min(y, H - self.window_size)))
                        x_start = int(max(0, min(x, W - self.window_size)))
                        yield BoundingBoxQuery(
                            index=tile['index'],
                            miny=y_start,
                            maxy=y_start + self.window_size, # Fixed Size
                            minx=x_start,
                            maxx=x_start + self.window_size, # Fixed Size
                        )

    def _generate_queries2(self) -> Generator[BoundingBoxQuery, None, None]:
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
        return iter(self._queries)

    def __len__(self):
        return len(self._queries)


class StaticSampler(Sampler):
    """A static sampler that returns a fixed set of queries."""

    def __init__(self, queries: list[BoundingBoxQuery], shuffle: bool=True, pad: tuple[int, int]=None):
        super().__init__()
        self.shuffle = shuffle
        self.queries = self._pad_queries(queries, pad)

    def _pad_queries(self, queries: list[BoundingBoxQuery], pad) -> list[BoundingBoxQuery]:
        if not pad:
            return queries

        pad_y, pad_x = pad
        return [
            BoundingBoxQuery(
                index=query.index,
                miny=max(0, query.miny - pad_y),
                maxy=min(7000, query.maxy + pad_y),
                minx=max(0, query.minx - pad_x),
                maxx=min(6000, query.maxx + pad_x)
            )
            for query in queries
        ]

    def __iter__(self):
        queries = self.queries
        if self.shuffle:
            np.random.shuffle(queries)

        return iter(queries)

    def __len__(self):
        return len(self.queries)


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

    def grid_sampler(self, tiles):
        return FixedGridSampler(tiles, self.boundary_dists, window_size=self.window_size)

    def static_sampler(self, queries: list[BoundingBoxQuery], shuffle: bool=True, pad: tuple[int, int]=None):
        return StaticSampler(queries, shuffle=shuffle, pad=pad)

    def dynamic_sampler(self, tiles) -> Sampler:
        tile_sampler = TileSampler(
            tiles,
            steps_per_epoch=self.steps_per_epoch,
            entropy_weight=self.entropy_weight)

        if self.boundary_dists:
            window_sampler = BoundaryDistWindowSampler(
                boundary_dists=self.boundary_dists,
                window=self.window_size,
                boundary_ratio=self.boundary_ratio)
        else:
            window_sampler = RandomUnfiformWindowSampler(
                7000, 6000, self.window_size)

        return SlidingSampler(tile_sampler, window_sampler)
