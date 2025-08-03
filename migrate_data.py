import json
from database import Database
from models import ImageMetadata

def migrate_data():
    db = Database()
    with open("processed_images_metadata.json", "r") as f:
        data = json.load(f)
        for item in data:
            metadata = ImageMetadata.from_dict(item)
            db.add_image_metadata(metadata)

if __name__ == "__main__":
    migrate_data()
