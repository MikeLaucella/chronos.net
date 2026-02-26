"""
module.py

Helper module for setting up training data loaders and models.
"""

import logging

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import albumentations as A
import torch

import chronos.data.chrono_transform as T
from chronos.data.geo_tiles import LazyZarr
from chronos.data.geo_array import GeoArrayDataset
from chronos.data.geo_sampler import GeoSamplerBuilder
from chronos.data.chrono_set import ChronosDataset, ChronosCollator


class ChronosDataModule(LightningDataModule):
    """The data module for Chronos 1m dataset."""

    def __init__(self,
                zarr_dir: str,
                images: list[str] = {'naip_hist'},
                masks: list[str] = {'masks'},
                steps_per_epoch: int = 1000,
                batch_size: int = 16,
                accumulate: int = 1,
                win_size: int = 512,
                workers: int = 4,
                keep_query: bool = False):
        super().__init__()

        # mandatory parameters
        self.zarr_dir = zarr_dir
        self.image_keys = set(images)
        self.mask_keys = set(masks)
        self.data_keys = list(images) + list(masks)

        # optional parameters
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.win_size = win_size
        self.workers = workers
        self.keep_query = keep_query
        self.accumulate = accumulate

        self._logger = logging.getLogger('chronos.data.module')

        self._collator = ChronosCollator(keep_query=keep_query)

        mu = 0.6493820041822563
        sigma = 0.17010567701149337

        begin_transforms = T.Compose([
            T.Squeeze(keys=['masks']),
            T.ToChannelLast(keys=['naip_hist', 'naip', 'eros_hist']),
            T.ImageTransform(images=['naip_hist', 'naip', 'eros_hist', 'masks'], aug=A.Resize(512, 512))
        ])

        end_transforms = T.Compose([
            T.ImageTransform(images=['naip_hist', 'eros_hist'], aug=A.ToRGB()),
            T.ImageTransform(images=['eros_hist'], aug=A.Normalize(mean=[mu, mu, mu], std=[sigma, sigma, sigma])),
            T.ImageTransform(images=['naip_hist'], aug=A.Normalize(mean=[mu, mu, mu], std=[sigma, sigma, sigma])),
            T.ToTensor(keys=['naip_hist', 'naip', 'eros_hist', 'masks']),
            T.ToType(keys=['masks'], dtype=torch.long)
        ])

        deform_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.05, p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5)
            ], p=0.2),
            A.GaussNoise(std_range=(0.2, 0.3), p=0.3),
        ])

        geometric_transforms = T.JointAlbumentationTransforms(
            augs=[A.D4(p=1.0)],
            images=self.image_keys,
            masks=self.mask_keys
        )

        self.test_transform = T.Compose([
            begin_transforms,
            T.FDA('naip_hist', 'eros_hist', beta_limit=(0.01, 0.04), p=1.0),
            end_transforms])

        self.train_transform = T.Compose([
            begin_transforms,
            geometric_transforms,
            T.FDA('naip_hist', 'eros_hist', beta_limit=(0.01, 0.04), p=1.0),
            #T.ImageTransform(images=['naip_hist'], aug=deform_transforms),
            end_transforms
        ])

        self.labels = [
            'Empty',
            'Parking Lot',
            'Building',
            'Grass',
            'Road',
            'Tree',
            'Agriculture',
            'Water',
            'Baren',
            'Other'
        ]

    def _filter_dataset(self, sets: dict[str, GeoArrayDataset]) -> dict[str, GeoArrayDataset]:
        return {key: ds for key, ds in sets.items() if key in self.data_keys}

    def _extract_metadata(self, z, type: str):
        metadata = z.attrs['metadata']
        return [metadata['tiles'][i] for i in metadata[type]]

    def setup(self, stage: str = None):
        """Setup the datasets and geo index."""
        # setup the zarr dataset
        self._logger.info('Loading zarr root from %s', self.zarr_dir)
        _zarr_root = LazyZarr(self.zarr_dir)

        self._sets = self._filter_dataset({
            'naip_hist': GeoArrayDataset(_zarr_root['naip_hist'], channels=1),
            'naip': GeoArrayDataset(_zarr_root['naip'], channels=3),
            'eros_hist': GeoArrayDataset(_zarr_root['eros_hist'], channels=1),
            'masks': GeoArrayDataset(_zarr_root['masks/urbanwatch'], channels=1)
        })

        self._logger.info('Loading samplers for tiles')
        _samplers = GeoSamplerBuilder(
            boundary_dists=_zarr_root['masks/distances_down'],
            window_size=self.win_size,
            steps_per_epoch=self.steps_per_epoch * self.batch_size * self.accumulate)

        self._train_sampler = _samplers.training(self._extract_metadata(_zarr_root.root, 'train'))
        self._val_sampler = _samplers.validation(self._extract_metadata(_zarr_root.root, 'val'))
        self._test_sampler = _samplers.testing(self._extract_metadata(_zarr_root.root, 'test'))

    def train_dataloader(self) -> DataLoader:
        """Get the Chronos training DataLoader.

        :return: The Chronos training DataLoader
        :rtype: DataLoader
        """
        ds = ChronosDataset(self._sets, self.train_transform, self.keep_query)
        return DataLoader(ds,
            batch_size=self.batch_size,
            #persistent_workers=True,
            #num_workers=self.workers,
            sampler=self._train_sampler,
            collate_fn=self._collator)

    def val_dataloader(self) -> DataLoader:
        """Get the Chronos validation DataLoader.

        :return: The Chronos validation DataLoader
        :rtype: DataLoader
        """
        ds = ChronosDataset(self._sets, self.test_transform, self.keep_query)
        return DataLoader(ds,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.workers,
            sampler=self._val_sampler,
            collate_fn=self._collator)

    def test_dataloader(self) -> DataLoader:
        """Get the Chronos test DataLoader.

        :return: The Chronos test DataLoader
        :rtype: DataLoader
        """
        ds = ChronosDataset(self._sets, self.test_transform, self.keep_query)
        return DataLoader(ds,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.workers,
            sampler=self._test_sampler,
            collate_fn=self._collator)
