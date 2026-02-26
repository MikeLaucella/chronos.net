"""
chrono_transform.py

The transformation utilities for Chronos dataset.
"""

from abc import abstractmethod
import albumentations as A
import torch


class Transform:
    @abstractmethod
    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        pass


class Compose(Transform):
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        for t in self.transforms:
            sample = t(sample)

        return sample


class ImageTransform(Transform):
    def __init__(self, images, aug):
        self.images = images
        self.aug = aug

    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        for key in self.images:
            if key in sample:
                sample[key] = self.aug(image=sample[key])["image"]

        return sample


class ToTensor(Transform):
    def __init__(self, keys: list[str]):
        self.keys = keys

    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        for key in self.keys:
            if key in sample:
                tensor = torch.from_numpy(sample[key])
                if tensor.ndim == 3 and tensor.shape[-1] <= 4: # HxWxC -> CxHxW
                    tensor = tensor.permute(2, 0, 1)

                sample[key] = tensor

        return sample


class ToType(Transform):
    def __init__(self, keys: list[str], dtype):
        self.keys = keys
        self.dtype = dtype

    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        for key in self.keys:
            if key in sample:
                sample[key] = sample[key].to(self.dtype)

        return sample


class Squeeze(Transform):
    def __init__(self, keys, axis: int=0):
        self.keys = keys
        self.axis = axis

    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        for key in self.keys:
            if key in sample:
                sample[key] = sample[key].squeeze(self.axis)

        return sample


class ToChannelLast(Transform):
    def __init__(self, keys: list[str]):
        self.keys = keys

    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        for key in self.keys:
            if key in sample:
                sample[key] = sample[key].transpose(1, 2, 0)

        return sample


class FDA(Transform):
    def __init__(self,
                 source_key: str,
                 ref_key: str,
                 beta_limit: float | tuple[float, float] = None,
                 p: float=0.5):
        self.source_key = source_key
        self.ref_key = ref_key
        self.fda = A.FDA(metadata_key='reference_image', p=p, beta_limit=beta_limit)

    def __call__(self, sample: dict[str, any]) -> dict[str, any]:
        if self.source_key not in sample or self.ref_key not in sample:
            return sample

        src = sample[self.source_key]
        ref = sample[self.ref_key]
        sample[self.source_key] = self.fda(image=src, reference_image=[ref])['image']
        return sample


class JointAlbumentationTransforms(Transform):
    def __init__(self, augs, images: list[str], masks: list[str]):
        self.images = list(images)
        self.masks = list(masks)

        # Build mapping and additional_targets
        self.additional_targets, self.forward_map, self.reverse_map = \
            self._build_key_map(self.images, self.masks)

        # Albumentations pipeline
        self.aug = A.Compose(augs, additional_targets=self.additional_targets)

    def _build_key_map(self, images, masks):
        additional = {}
        forward = {}
        reverse = {}

        # First image → "image"
        if images:
            forward[images[0]] = "image"
            reverse["image"] = images[0]

        # First mask → "mask"
        if masks:
            forward[masks[0]] = "mask"
            reverse["mask"] = masks[0]

        # Additional images
        for i, key in enumerate(images[1:], start=1):
            alb_key = f"image{i}"
            forward[key] = alb_key
            reverse[alb_key] = key
            additional[alb_key] = "image"

        # Additional masks
        for i, key in enumerate(masks[1:], start=1):
            alb_key = f"mask{i}"
            forward[key] = alb_key
            reverse[alb_key] = key
            additional[alb_key] = "mask"

        return additional, forward, reverse

    def __call__(self, sample):
        # Build Albumentations input dict
        data = {self.forward_map[k]: sample[k] for k in self.images + self.masks}

        # Run Albumentations
        out = self.aug(**data)

        # Map back to original keys
        for alb_key, orig_key in self.reverse_map.items():
            sample[orig_key] = out[alb_key]

        return sample
