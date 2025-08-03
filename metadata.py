import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import uuid

# I will need the new Database class
from database import Database

from models import ImageMetadata

class MetadataDBManager:
    def __init__(self, db_path: str = "database.db"):
        self.db = Database(db_path)

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
        """Add new image metadata to the database."""
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
        self.db.add_image_metadata(metadata)
        return metadata

    def get_all_metadata(self) -> List[ImageMetadata]:
        """Get all image metadata from the database."""
        return self.db.get_all_metadata()

    def get_metadata_by_id(self, file_id: str) -> Optional[ImageMetadata]:
        """Get metadata for a specific file ID from the database."""
        return self.db.get_metadata_by_id(file_id)

    def delete_metadata(self, file_id: str) -> bool:
        """Delete metadata for a specific file ID from the database."""
        return self.db.delete_metadata(file_id)

    def cleanup_orphaned_metadata(self) -> int:
        """Remove metadata entries for files that no longer exist."""
        all_metadata = self.get_all_metadata()
        cleaned_count = 0

        for metadata in all_metadata:
            result_file = f"static/outputs/{metadata.file_id}_result.png"
            pixelated_file = f"static/outputs/{metadata.file_id}_pixelated.png"

            if not os.path.exists(result_file) or not os.path.exists(pixelated_file):
                self.delete_metadata(metadata.file_id)
                cleaned_count += 1
        
        return cleaned_count

    def get_stats(self) -> Dict:
        """Get statistics about processed images from the database."""
        metadata_list = self.get_all_metadata()

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
        most_common_downsample = Counter(downsample_values).most_common(1)[0][0] if downsample_values else 0

        timestamps = [datetime.fromisoformat(m.timestamp) for m in metadata_list]

        return {
            'total_images': len(metadata_list),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'avg_palette_size': round(sum(palette_sizes) / len(palette_sizes), 1) if palette_sizes else 0,
            'most_common_downsample': most_common_downsample,
            'date_range': {
                'earliest': min(timestamps).strftime('%Y-%m-%d') if timestamps else 'N/A',
                'latest': max(timestamps).strftime('%Y-%m-%d') if timestamps else 'N/A'
            }
        }
