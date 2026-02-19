"""
module.py

Helper module for setting up training data loaders and models.
"""

import logging

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import albumentations as A
import zarr

from chronos.data.geo_array import GeoArrayDataset
from chronos.data.geo_sampler import GeoSamplerBuilder
from chronos.data.chrono_set import ChronosDataset, ChronosCollator


class ChronosDataModule(LightningDataModule):
    """The data module for Chronos 1m dataset."""

    def __init__(self,
                zarr_dir: str,
                data_keys: list[str] = ['naip_hist', 'masks'],
                steps_per_epoch: int = 1000,
                batch_size: int = 16,
                win_size: int = 512,
                workers: int = 4,
                keep_query: bool = False):
        super().__init__()

        # mandatory parameters
        self.zarr_dir = zarr_dir
        self.data_keys = data_keys

        # optional parameters
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.win_size = win_size
        self.workers = workers
        self.keep_query = keep_query

        self._logger = logging.getLogger('chronos.data.module')

        self.key_binding, self.reverse_mapping, self.tr_args = self._build_bindings(data_keys)
        self._collator = ChronosCollator(self.key_binding, keep_query=keep_query)

        self.test_transform = A.Compose([
            A.ToTensorV2()
        ], additional_targets=self.tr_args)

        self.train_transform = A.Compose([
            A.ToTensorV2()
        ], additional_targets=self.tr_args)

    def _build_bindings(self, data_keys: list[str]):
        key_mapping = {}
        reverse_mapping = {}
        tr_args = {}

        for key in data_keys:
            if key == 'naip_hist':
                key_mapping['image'] = key
                reverse_mapping[key] = 'image'
                tr_args['image'] = 'image'
            elif key == 'naip':
                key_mapping['image1'] = key
                reverse_mapping[key] = 'image1'
                tr_args['image1'] = 'image'
            elif key == 'eros_hist':
                key_mapping['image2'] = key
                reverse_mapping[key] = 'image2'
                tr_args['image2'] = 'image'
            elif key == 'masks':
                key_mapping['masks'] = key
                reverse_mapping[key] = 'masks'
                tr_args['masks'] = 'mask'
            else:
                raise ValueError(f'Unknown data key: {key}')

        return key_mapping, reverse_mapping, tr_args

    def _filter_dataset(self, sets: dict[str, GeoArrayDataset]) -> dict[str, GeoArrayDataset]:
        filtered = {}

        for name in self.sets.keys():
            binding = self.reverse_mapping.get(name)
            if binding is None:
                raise ValueError(f'No binding found for data key: {name}')
            
            filtered[binding] = sets[binding]

        return filtered

    def _extract_metadata(self, z, type: str):
        metadata = z.attrs['metadata']
        return [metadata['tiles'][i] for i in metadata[type]]

    def setup(self, stage: str = None):
        """Setup the datasets and geo index."""
        # setup the zarr dataset
        self._logger.info('Loading zarr root from %s', self.zarr_dir)
        _zarr_root = zarr.open(self.zarr_dir, mode='r')

        self._sets = self._filter_dataset({
            'naip_hist': GeoArrayDataset(_zarr_root['naip_hist'], channels=1, channel_offset=1),
            'naip': GeoArrayDataset(_zarr_root['naip'], channels=3),
            'eros_hist': GeoArrayDataset(_zarr_root['eros_hist'], channels=1),
            'masks': GeoArrayDataset(_zarr_root['masks/urbanwatch'], channels=1)
        })

        self._logger.info('Loading samplers for tiles')
        _samplers = GeoSamplerBuilder(
            boundary_dists=_zarr_root['masks/distances'],
            window_size=self.win_size,
            steps_per_epoch=self.steps_per_epoch)

        self._train_sampler = _samplers.training(self._extract_metadata(_zarr_root, 'train'))
        self._val_sampler = _samplers.validation(self._extract_metadata(_zarr_root, 'val'))
        self._test_sampler = _samplers.testing(self._extract_metadata(_zarr_root, 'test'))

    def train_dataloader(self) -> DataLoader:
        """Get the Chronos training DataLoader.

        :return: The Chronos training DataLoader
        :rtype: DataLoader
        """
        ds = ChronosDataset(self._sets, self.train_transform, self.keep_query)
        return DataLoader(ds,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.workers,
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
