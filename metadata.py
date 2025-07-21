import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import uuid

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

class MetadataManager:
    def __init__(self, metadata_file: str = "processed_images_metadata.json"):
        self.metadata_file = metadata_file
        self.ensure_metadata_file()

    def ensure_metadata_file(self):
        """Ensure the metadata file exists."""
        if not os.path.exists(self.metadata_file):
            self.save_metadata([])

    def load_metadata(self) -> List[ImageMetadata]:
        """Load all metadata from the JSON file."""
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            return [ImageMetadata.from_dict(item) for item in data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_metadata(self, metadata_list: List[ImageMetadata]):
        """Save metadata list to JSON file."""
        data = [item.to_dict() for item in metadata_list]
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_image_metadata(self,
                          file_id: str,
                          original_filename: str,
                          downsample_by: int,
                          palette: int,
                          upscale: int,
                          colors: List[List[int]],
                          file_size: int,
                          original_size: tuple = None,
                          pixelated_size: tuple = None) -> ImageMetadata:
        """Add new image metadata."""
        metadata = ImageMetadata(
            file_id=file_id,
            original_filename=original_filename,
            timestamp=datetime.now().isoformat(),
            downsample_by=downsample_by,
            palette=palette,
            upscale=upscale,
            colors=colors,
            file_size=file_size,
            original_size=original_size,
            pixelated_size=pixelated_size
        )

        # Load existing metadata
        metadata_list = self.load_metadata()

        # Add new metadata
        metadata_list.append(metadata)

        # Save updated metadata
        self.save_metadata(metadata_list)

        return metadata

    def get_all_metadata(self) -> List[ImageMetadata]:
        """Get all image metadata sorted by timestamp (newest first)."""
        metadata_list = self.load_metadata()
        return sorted(metadata_list, key=lambda x: x.timestamp, reverse=True)

    def get_metadata_by_id(self, file_id: str) -> Optional[ImageMetadata]:
        """Get metadata for a specific file ID."""
        metadata_list = self.load_metadata()
        for metadata in metadata_list:
            if metadata.file_id == file_id:
                return metadata
        return None

    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file ID."""
        metadata_list = self.load_metadata()
        original_length = len(metadata_list)
        metadata_list = [m for m in metadata_list if m.file_id != file_id]

        if len(metadata_list) < original_length:
            self.save_metadata(metadata_list)
            return True
        return False

    def cleanup_orphaned_metadata(self) -> int:
        """Remove metadata entries for files that no longer exist."""
        metadata_list = self.load_metadata()
        existing_metadata = []
        cleaned_count = 0

        for metadata in metadata_list:
            result_file = f"static/outputs/{metadata.file_id}_result.png"
            pixelated_file = f"static/outputs/{metadata.file_id}_pixelated.png"

            if os.path.exists(result_file) and os.path.exists(pixelated_file):
                existing_metadata.append(metadata)
            else:
                cleaned_count += 1

        if cleaned_count > 0:
            self.save_metadata(existing_metadata)

        return cleaned_count

    def get_stats(self) -> Dict:
        """Get statistics about processed images."""
        metadata_list = self.load_metadata()

        if not metadata_list:
            return {
                'total_images': 0,
                'total_size_mb': 0,
                'avg_palette_size': 0,
                'most_common_downsample': 0,
                'date_range': None
            }

        total_size = sum(m.file_size for m in metadata_list)
        palette_sizes = [m.palette for m in metadata_list]
        downsample_values = [m.downsample_by for m in metadata_list]

        from collections import Counter
        most_common_downsample = Counter(downsample_values).most_common(1)[0][0]

        timestamps = [datetime.fromisoformat(m.timestamp) for m in metadata_list]

        return {
            'total_images': len(metadata_list),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'avg_palette_size': round(sum(palette_sizes) / len(palette_sizes), 1),
            'most_common_downsample': most_common_downsample,
            'date_range': {
                'earliest': min(timestamps).strftime('%Y-%m-%d'),
                'latest': max(timestamps).strftime('%Y-%m-%d')
            }
        }
