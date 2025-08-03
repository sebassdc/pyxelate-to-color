from dataclasses import dataclass, asdict
from typing import List

@dataclass
class ImageMetadata:
    file_id: str
    original_filename: str
    timestamp: str
    downsample_by: int
    palette: int
    upscale: int
    colors: List[List[int]]
    file_size: int
    original_size: tuple = None
    pixelated_size: tuple = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
