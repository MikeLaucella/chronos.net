"""
model.py

Data model classes for datasets.
"""

from dataclasses import dataclass


@dataclass
class BoundingBoxQuery:
    """Bounding box query."""

    index: int
    minx: int
    miny: int
    maxx: int
    maxy: int
