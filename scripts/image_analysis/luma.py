import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def calculate_luma(image_path):
    """Calculate the Luma value for an image using the formula L = 0.299R + 0.587G + 0.114B"""
    try:
        # Read image in BGR format (OpenCV default)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract R, G, B channels
        r = image_rgb[:, :, 0].astype(np.float64)
        g = image_rgb[:, :, 1].astype(np.float64)
        b = image_rgb[:, :, 2].astype(np.float64)
        
        # Calculate luma using the formula L = 0.299R + 0.587G + 0.114B
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Return the mean luma value across all pixels
        return float(np.mean(luma))
        
    except Exception as e:
        print(f"Error calculating luma for {image_path}: {e}")
        return None


def update_json_with_luma(json_file_path, image_dir, resume=True):
    """Update existing JSON file with luma values for all images"""
    
    # Load existing JSON data
    if not Path(json_file_path).exists():
        print(f"JSON file not found: {json_file_path}")
        return 0
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    image_dir = Path(image_dir)
    updated_count = 0
    skipped_count = 0
    
    print(f"Processing {len(data)} images from {json_file_path}")
    
    for image_name, image_data in tqdm(data.items(), desc=f"Adding luma to {Path(json_file_path).name}"):
        # Skip if luma already exists and resume is True
        if resume and 'luma' in image_data.get('characteristics', {}):
            skipped_count += 1
            continue
        
        # Construct image path
        image_path = image_dir / image_name
        
        if not image_path.exists():
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # Calculate luma
        luma_value = calculate_luma(str(image_path))
        
        if luma_value is not None:
            # Ensure characteristics section exists
            if 'characteristics' not in image_data:
                image_data['characteristics'] = {}
            
            # Add luma to characteristics
            image_data['characteristics']['luma'] = luma_value
            updated_count += 1
    
    # Save updated JSON
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated {updated_count} images with luma values")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images (already had luma values)")
    
    return updated_count


def main():
    parser = argparse.ArgumentParser(description="Add luma values to existing image characteristics JSON files")
    parser.add_argument("--train_dir", default="data/raw/images/train", help="Path to training images directory")
    parser.add_argument("--val_dir", default="data/raw/images/val", help="Path to validation images directory")
    parser.add_argument("--train_json", default="results/images/train.json", help="Path to training JSON file")
    parser.add_argument("--val_json", default="results/images/val.json", help="Path to validation JSON file")
    parser.add_argument("--no-resume", action="store_true", help="Recalculate luma for all images (don't skip existing)")
    
    args = parser.parse_args()
    
    resume = not args.no_resume
    
    print("=== Adding Luma Values to Image Characteristics ===")
    print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")
    print()
    
    total_updated = 0
    
    # Process training images
    if Path(args.train_json).exists():
        print("Processing training images...")
        train_updated = update_json_with_luma(args.train_json, args.train_dir, resume=resume)
        total_updated += train_updated
        print()
    else:
        print(f"Training JSON file not found: {args.train_json}")
        print("Run extract_image_characteristics.py first to generate the JSON files.")
    
    # Process validation images
    if Path(args.val_json).exists():
        print("Processing validation images...")
        val_updated = update_json_with_luma(args.val_json, args.val_dir, resume=resume)
        total_updated += val_updated
        print()
    else:
        print(f"Validation JSON file not found: {args.val_json}")
        print("Run extract_image_characteristics.py first to generate the JSON files.")
    
    print(f"Complete! Added luma values to {total_updated} images total.")
    print(f"Luma formula used: L = 0.299R + 0.587G + 0.114B")


if __name__ == "__main__":
    main()