"""
geo_tiles.py

The geo tiles interface for fetching tile data based on index.
"""

from abc import abstractmethod
import zarr


class GeoTiles:
    """The geo tiles interface for fetching tile data based on index."""

    @abstractmethod
    def __getitem__(self, index: int):
        """Get the tile data for the given index."""
        pass

    @abstractmethod
    def __len__(self):
        """Get the number of tiles in the geo tiles collection."""
        pass


class LazyZarr:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self._root = None

    @property
    def root(self):
        if self._root is None:
            self._root = zarr.open(self.root_path, mode="r")

        return self._root

    def __getitem__(self, group_name: str):
        return ZarrTiles(self.root[group_name])


class ZarrTiles(GeoTiles):
    """The zarr tiles implementation of the GeoTiles interface."""

    def __init__(self, group: zarr.Group):
        self.group = group

    def _get_tileid(self, idx: int) -> str:
        return f'tile_{idx:03d}'

    def __getitem__(self, index: int):
        """Get the tile data for the given index."""
        tile_id = self._get_tileid(index)
        return self.group[tile_id]

    def __len__(self):
        """Get the number of tiles in the geo tiles collection."""
        return len(self.group)
