
import os
import sqlite3
from typing import List, Optional

from models import ImageMetadata


class Database:
    def __init__(self, db_path: str = "database.db"):
        self.db_path = db_path
        self.create_table()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def create_table(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT,
                    timestamp TEXT,
                    downsample_by INTEGER,
                    palette INTEGER,
                    upscale INTEGER,
                    colors TEXT,
                    file_size INTEGER,
                    original_size TEXT,
                    pixelated_size TEXT
                )
            ''')
            conn.commit()

    def add_image_metadata(self, metadata: ImageMetadata) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO images (id, original_filename, timestamp, downsample_by, palette, upscale, colors, file_size, original_size, pixelated_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.file_id,
                metadata.original_filename,
                metadata.timestamp,
                metadata.downsample_by,
                metadata.palette,
                metadata.upscale,
                str(metadata.colors),
                metadata.file_size,
                str(metadata.original_size),
                str(metadata.pixelated_size)
            ))
            conn.commit()

    def get_all_metadata(self) -> List[ImageMetadata]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [self._row_to_metadata(row) for row in rows]

    def get_metadata_by_id(self, file_id: str) -> Optional[ImageMetadata]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            return self._row_to_metadata(row) if row else None

    def delete_metadata(self, file_id: str) -> bool:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM images WHERE id = ?", (file_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_metadata(self, row: tuple) -> ImageMetadata:
        return ImageMetadata(
            file_id=row[0],
            original_filename=row[1],
            timestamp=row[2],
            downsample_by=row[3],
            palette=row[4],
            upscale=row[5],
            colors=eval(row[6]),
            file_size=row[7],
            original_size=eval(row[8]),
            pixelated_size=eval(row[9])
        )
