"""
factory.py

The model factory for segmentation models.
"""

from torch import nn

import chronos.model.segformer.hf_segformer as hf
from chronos.model.fda.bolt import FDABolt
from chronos.model.segformer.bolt import SegformerBolt


def _to_seg_bolt(segformer: nn.Module, args: dict, labels: list[str]) -> SegformerBolt:
    return SegformerBolt(
        segformer=segformer,
        classes=len(labels),
        lr=args.get('lr', 6e-5))


def _to_fda_bolt(segformer: nn.Module, args: dict, labels: list[str]) -> FDABolt:
    return FDABolt(
        model=segformer,
        lr=args.get('lr', 6e-5),
        classes=labels,
        eta=1.0
    )


def get_model(kwargs: dict, labels: list[int] = None) -> nn.Module:
    """Get the segmentation model based on the provided arguments.

    :param args: The arguments
    :type args: dict
    :return: The segmentation model
    :rtype: nn.Module
    """
    model_name = kwargs.get('model', None)

    match model_name:
        case 'b0': return _to_seg_bolt(hf.b0(labels), kwargs, labels)
        case 'b1': return _to_seg_bolt(hf.b1(labels), kwargs, labels)
        case 'b2': return _to_seg_bolt(hf.b2(labels), kwargs, labels)
        case 'b3': return _to_seg_bolt(hf.b3(labels), kwargs, labels)
        case 'b4': return _to_seg_bolt(hf.b4(labels), kwargs, labels)
        case 'fda1': return _to_fda_bolt(hf.b1(labels), kwargs, labels)
        case 'fda2': return _to_fda_bolt(hf.b2(labels), kwargs, labels)
        case _: raise ValueError(f'Unknown model name: {model_name}')
