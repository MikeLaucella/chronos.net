"""
geo_array.py

Geodata dataset for extracting bounding box data from array sources.
"""

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from chronos.data.models import BoundingBoxQuery


class GeoArrayDataset(Dataset[dict[str, Tensor]]):
    """Geodata dataset from array sources."""

    def __init__(self,
                 images: np.array,
                 channels: int = 3,
                 channel_offset: int = 0,
                 transforms: any = None):
        self.images = images
        self.channels = channels
        self.channel_offset = channel_offset
        self.transforms = transforms

    def _get_image_data(self, query: BoundingBoxQuery) -> np.array:
        """Get the image data for the bounding box query.

        :param query: The bounding box query
        :type query: BoundingBoxQuery
        :return: All the related data
        :rtype: np.array
        """
        image = self.images[query.index]
        return image[
            self.channel_offset:self.channel_offset+self.channels,
            query.miny:query.maxy,
            query.minx:query.maxx
        ] # CxHxW

    def _get_item(self, query: BoundingBoxQuery):
        # get the image data
        image = self._get_image_data(query)

        # apply transforms
        return self.transforms(image) if self.transforms else image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, query: BoundingBoxQuery):
        # route the data fetch based on mask presence in the data
        return self._get_item(query)
