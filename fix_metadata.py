#!/usr/bin/env python3
"""
Script to fix malformed color data in processed_images_metadata.json
This script converts colors from [[r, g, b]] format to [r, g, b] format
"""

import json
import os
import shutil
from datetime import datetime

def fix_metadata_colors():
    """Fix the color data structure in metadata file."""
    metadata_file = "processed_images_metadata.json"

    if not os.path.exists(metadata_file):
        print("❌ Metadata file not found!")
        return False

    # Create backup
    backup_file = f"{metadata_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(metadata_file, backup_file)
    print(f"✅ Created backup: {backup_file}")

    # Load existing metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error loading JSON: {e}")
        return False

    print(f"📊 Found {len(metadata_list)} metadata entries")

    # Fix color structure
    fixed_count = 0
    for metadata in metadata_list:
        if 'colors' in metadata and metadata['colors']:
            original_colors = metadata['colors']
            fixed_colors = []

            for color_entry in original_colors:
                if isinstance(color_entry, list) and len(color_entry) == 1 and isinstance(color_entry[0], list):
                    # This is malformed: [[r, g, b]] -> [r, g, b]
                    fixed_colors.append(color_entry[0])
                    fixed_count += 1
                elif isinstance(color_entry, list) and len(color_entry) == 3:
                    # This is already correct: [r, g, b]
                    fixed_colors.append(color_entry)
                else:
                    print(f"⚠️  Unexpected color format: {color_entry}")
                    fixed_colors.append(color_entry)

            metadata['colors'] = fixed_colors

    # Save fixed metadata
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        print(f"✅ Fixed {fixed_count} color entries")
        print(f"✅ Updated metadata file: {metadata_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving fixed metadata: {e}")
        # Restore backup
        shutil.copy2(backup_file, metadata_file)
        print(f"🔄 Restored backup")
        return False

if __name__ == "__main__":
    success = fix_metadata_colors()
    if success:
        print("🎉 Metadata fix completed successfully!")
    else:
        print("💥 Metadata fix failed!")
