"""
hf_segformer.py

Models and wrappers for Huggingface Segformer models.
"""

import torch
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class HfSegformer(nn.Module):
    """Huggingface Segformer wrapper."""

    def __init__(self, segformer: nn.Module):
        super().__init__()
        self.segformer = segformer

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.segformer(image)
        logits = torch.nn.functional.interpolate(
            input=out.logits,
            size=(512, 512),#image.shape[-2:],
            mode='bilinear',
            align_corners=False)

        return logits


def _get_segformer_model(name: str, labels: list[str]) -> nn.Module:
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    hf_segformer = SegformerForSemanticSegmentation.from_pretrained(
        name,
        num_channels=3,
        return_dict=True,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    #hf_segformer.hidden_dropout_prob = 0.1
    #hf_segformer.attention_dropout = 0.1

    return HfSegformer(hf_segformer)


def b0(labels: list[str]) -> SegformerForSemanticSegmentation:
    """Get the Segformer B0 model for segmentation tasks.

    :return: The Segformer B0 model
    :rtype: SegformerForSemanticSegmentation
    """
    #return _get_segformer_model("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", labels)
    #return _get_segformer_model("nvidia/segformer-b0-finetuned-ade-512-512", labels)
    return _get_segformer_model("nvidia/mit-b0", labels)


def b1(labels: list[str]) -> SegformerForSemanticSegmentation:
    """Get the Segformer B1 model for segmentation tasks.

    :return: The Segformer B1 model
    :rtype: SegformerForSemanticSegmentation
    """
    #return _get_segformer_model("nvidia/segformer-b1-finetuned-cityscapes-1024-1024", labels)
    return _get_segformer_model("nvidia/mit-b1", labels)


def b2(labels: list[str]) -> SegformerForSemanticSegmentation:
    """Get the Segformer B2 model for segmentation tasks.

    :return: The Segformer B2 model
    :rtype: SegformerForSemanticSegmentation
    """
    #return _get_segformer_model("nvidia/segformer-b2-finetuned-cityscapes-1024-1024", labels)
    return _get_segformer_model("nvidia/mit-b2", labels)


def b3(labels: list[str]) -> SegformerForSemanticSegmentation:
    """Get the Segformer B3 model for segmentation tasks.

    :return: The Segformer B3 model
    :rtype: SegformerForSemanticSegmentation
    """
    #return _get_segformer_model("nvidia/segformer-b3-finetuned-cityscapes-1024-1024", labels)
    return _get_segformer_model("nvidia/mit-b3", labels)


def b4(labels: list[str]) -> SegformerForSemanticSegmentation:
    """Get the Segformer B4 model for segmentation tasks.

    :return: The Segformer B4 model
    :rtype: SegformerForSemanticSegmentation
    """
    return _get_segformer_model("nvidia/segformer-b4-finetuned-cityscapes-1024-1024", labels)
