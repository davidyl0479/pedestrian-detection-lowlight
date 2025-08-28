import cv2
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as gpu_ndimage
import pywt
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def setup_gpu_memory_limit(max_utilization_percent=85):
    """Set GPU memory pool limit to control utilization"""
    try:
        import cupy as cp
        
        # Get total GPU memory
        device = cp.cuda.Device()
        total_memory = device.mem_info[1]  # total memory in bytes
        
        # Calculate limit based on percentage
        memory_limit = int(total_memory * max_utilization_percent / 100)
        
        # Set memory pool limit
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=memory_limit)
        
        print(f"GPU memory limit set to {max_utilization_percent}% ({memory_limit / 1024**3:.1f} GB)")
        
    except Exception as e:
        print(f"Warning: Could not set GPU memory limit: {e}")
        print("Continuing without memory limit...")


def fast_multiscale_laplacian_gpu(gray_gpu, window_size=64):
    """GPU-accelerated multi-scale Laplacian - same structure as original"""
    
    def weighted_variance_gpu(window_gpu):
        # Pre-define Laplacian kernels
        laplacian_3 = cp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=cp.float64)
        laplacian_5 = cp.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0], 
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=cp.float64)
        laplacian_7 = cp.array([
            [0, 0, 0, -1, 0, 0, 0],
            [0, 0, -2, -4, -2, 0, 0],
            [0, -2, -4, -8, -4, -2, 0],
            [-1, -4, -8, 48, -8, -4, -1],
            [0, -2, -4, -8, -4, -2, 0],
            [0, 0, -2, -4, -2, 0, 0],
            [0, 0, 0, -1, 0, 0, 0]
        ], dtype=cp.float64)
        
        # Apply Laplacian filters
        lap_3 = gpu_ndimage.convolve(window_gpu, laplacian_3)
        lap_5 = gpu_ndimage.convolve(window_gpu, laplacian_5) 
        lap_7 = gpu_ndimage.convolve(window_gpu, laplacian_7)
        
        # Calculate variances
        var_3 = float(cp.var(lap_3))
        var_5 = float(cp.var(lap_5))
        var_7 = float(cp.var(lap_7))
        
        return 0.6*var_3 + 0.3*var_5 + 0.1*var_7
    
    # Extract windows (same structure as original)
    h, w = gray_gpu.shape
    windows = [gray_gpu[y:y+window_size, x:x+window_size] 
               for y in range(0, h-window_size+1, window_size//2)
               for x in range(0, w-window_size+1, window_size//2)]
    
    return np.percentile([weighted_variance_gpu(w) for w in windows], 75)


def compare_wavelets_block_mad(image, block_size=64):
    """Test multiple wavelets and return results for comparison"""
    wavelets = ['db1', 'db4', 'bior2.2']
    
    def weighted_mad(block, wavelet):
        _, (LH, HL, HH) = pywt.dwt2(block, wavelet)
        mad_LH = np.median(np.abs(LH.flatten() - np.median(LH.flatten()))) / 0.6745
        mad_HL = np.median(np.abs(HL.flatten() - np.median(HL.flatten()))) / 0.6745  
        mad_HH = np.median(np.abs(HH.flatten() - np.median(HH.flatten()))) / 0.6745
        return 0.3 * mad_LH + 0.3 * mad_HL + 0.4 * mad_HH
    
    blocks = [image[y:y+block_size, x:x+block_size] 
              for y in range(0, image.shape[0]-block_size+1, block_size)
              for x in range(0, image.shape[1]-block_size+1, block_size)]
    
    results = {}
    for wavelet in wavelets:
        results[wavelet] = np.percentile([weighted_mad(block, wavelet) for block in blocks], 75)
    
    return results


def calculate_all_metrics_gpu(image_path):
    """Single-pass GPU-accelerated metric calculation"""
    
    image = cv2.imread(image_path)
    gray_cpu = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_gpu = cp.asarray(gray_cpu)
    
    # Basic metrics
    mean_intensity = float(cp.mean(gray_gpu))
    std_intensity = float(cp.std(gray_gpu))
    rms_contrast = std_intensity / (mean_intensity + 1e-8)
    
    # Sobel edge density
    sobel_x = gpu_ndimage.sobel(gray_gpu, axis=1)
    sobel_y = gpu_ndimage.sobel(gray_gpu, axis=0)
    sobel_magnitude = cp.sqrt(sobel_x**2 + sobel_y**2)
    edge_density = float(cp.mean(sobel_magnitude))
    
    # HSL lightness
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsl_lightness = np.mean(hls[:,:,1]) / 255.0
    
    # NOW we can call the helper functions
    gray_cpu_for_blocks = cp.asnumpy(gray_gpu)
    wavelet_mad = compare_wavelets_block_mad(gray_cpu_for_blocks)
    laplacian_score = fast_multiscale_laplacian_gpu(gray_gpu)
    
    # GPU memory cleanup
    del gray_gpu, sobel_x, sobel_y, sobel_magnitude
    cp.get_default_memory_pool().free_all_blocks()
    
    return {
        'rms_contrast': rms_contrast,
        'edge_density': edge_density,
        'wavelet_mad_db1': wavelet_mad['db1'],
        'wavelet_mad_db4': wavelet_mad['db4'], 
        'wavelet_mad_bior2_2': wavelet_mad['bior2.2'],
        'laplacian_score': laplacian_score,
        'avg_intensity': mean_intensity,
        'hsl_lightness': hsl_lightness
    }


def load_annotations(annotation_file):
    """Load and organize annotation data by filename"""
    if not Path(annotation_file).exists():
        print(f"Warning: Annotation file not found: {annotation_file}")
        return {}, {}
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Create lookup tables
    image_data = {img['file_name']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    return image_data, categories, annotations_by_image


def process_image_directory(image_dir, output_file, annotation_file=None, batch_size=None, resume=False):
    """Process all images in a directory and save one JSON file with nested structure"""
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    # Load annotation data if provided
    image_data, categories, annotations_by_image = {}, {}, {}
    if annotation_file:
        image_data, categories, annotations_by_image = load_annotations(annotation_file)
        print(f"Loaded annotations for {len(image_data)} images, {len(categories)} categories")
    
    # Load existing results if resuming
    results = {}
    if resume and Path(output_file).exists():
        with open(output_file, 'r') as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing results")
        # Filter out already processed images
        processed_names = set(results.keys())
        image_files = [f for f in image_files if f.name not in processed_names]
    
    if batch_size is None:
        # Process all at once
        batches = [image_files]
        batch_desc = f"Processing {image_dir.name}"
    else:
        # Split into batches
        batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]
        batch_desc = f"Processing {image_dir.name} (batch {{batch_num}}/{len(batches)})"
    
    for batch_num, batch_files in enumerate(batches, 1):
        current_desc = batch_desc.format(batch_num=batch_num) if batch_size else batch_desc
        
        for image_file in tqdm(batch_files, desc=current_desc):
            try:
                metrics = calculate_all_metrics_gpu(str(image_file))
                
                # Get annotation data for this image
                img_info = image_data.get(image_file.name, {})
                annotations = []
                
                if img_info and img_info.get('id') in annotations_by_image:
                    for ann in annotations_by_image[img_info['id']]:
                        # Add category name to annotation
                        category = categories.get(ann['category_id'], {})
                        ann_with_category = ann.copy()
                        ann_with_category['category_name'] = category.get('name', 'unknown')
                        annotations.append(ann_with_category)
                
                # Structure the data as requested
                result_data = {
                    "characteristics": metrics
                }
                
                # Add image metadata if available
                if img_info:
                    result_data["image_info"] = {
                        "height": img_info.get("height"),
                        "width": img_info.get("width"),
                        "file_name": img_info.get("file_name"),
                        "id": img_info.get("id")
                    }
                
                # Add annotations if available
                if annotations:
                    result_data["annotations"] = annotations
                
                results[image_file.name] = result_data
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        # Save results after each batch
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if batch_size:
            print(f"Completed batch {batch_num}/{len(batches)} - {len(results)} total results saved")
    
    print(f"Saved {len(results)} image metrics to {output_file}")
    return len(results)


def get_user_preferences():
    """Get user preferences for batch size and GPU utilization"""
    print("=== Image Characteristics Extraction Configuration ===")
    print()
    
    # Get batch size
    while True:
        try:
            batch_input = input("Enter batch size (press Enter for no batching): ").strip()
            if batch_input == "":
                batch_size = None
                break
            batch_size = int(batch_input)
            if batch_size > 0:
                break
            else:
                print("Please enter a positive number or press Enter for no batching.")
        except ValueError:
            print("Please enter a valid number or press Enter for no batching.")
    
    # Get GPU utilization
    while True:
        try:
            gpu_input = input("Enter max GPU utilization percentage (default: 85): ").strip()
            if gpu_input == "":
                max_gpu_utilization = 85
                break
            max_gpu_utilization = int(gpu_input)
            if 1 <= max_gpu_utilization <= 100:
                break
            else:
                print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Please enter a valid number between 1 and 100.")
    
    # Ask about resuming
    while True:
        resume_input = input("Resume from existing results? (y/n, default: n): ").strip().lower()
        if resume_input in ['', 'n', 'no']:
            resume = False
            break
        elif resume_input in ['y', 'yes']:
            resume = True
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
    
    print()
    print(f"Configuration:")
    print(f"  - Batch size: {'No batching' if batch_size is None else batch_size}")
    print(f"  - Max GPU utilization: {max_gpu_utilization}%")
    print(f"  - Resume: {'Yes' if resume else 'No'}")
    print()
    
    return batch_size, max_gpu_utilization, resume


def main():
    parser = argparse.ArgumentParser(description="Extract image characteristics from train and val directories")
    parser.add_argument("--train_dir", default="data/raw/images/train", help="Path to training images directory")
    parser.add_argument("--val_dir", default="data/raw/images/val", help="Path to validation images directory")
    parser.add_argument("--train_annotations", default="data/raw/annotations/train/nightowls_training.json", help="Path to training annotations")
    parser.add_argument("--val_annotations", default="data/raw/annotations/val/nightowls_validation.json", help="Path to validation annotations")
    parser.add_argument("--output_base", default="results/images", help="Base output directory for results")
    parser.add_argument("--batch_size", type=int, default=None, help="Process images in batches of this size (if not provided, will prompt)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results (if not provided, will prompt)")
    parser.add_argument("--max_gpu_utilization", type=int, default=None, help="Maximum GPU memory utilization percentage (if not provided, will prompt)")
    parser.add_argument("--interactive", action="store_true", default=True, help="Use interactive mode (default)")
    parser.add_argument("--no-interactive", action="store_false", dest="interactive", help="Disable interactive mode")
    
    args = parser.parse_args()
    
    # Get user preferences if in interactive mode and values not provided
    if args.interactive and (args.batch_size is None or args.max_gpu_utilization is None):
        batch_size, max_gpu_utilization, resume = get_user_preferences()
        # Override with command line args if provided
        if args.batch_size is not None:
            batch_size = args.batch_size
        if args.max_gpu_utilization is not None:
            max_gpu_utilization = args.max_gpu_utilization
        if hasattr(args, 'resume') and args.resume:
            resume = args.resume
    else:
        batch_size = args.batch_size
        max_gpu_utilization = args.max_gpu_utilization or 85
        resume = args.resume
    
    # Setup GPU memory limit
    setup_gpu_memory_limit(max_gpu_utilization)
    
    # Create output directory
    output_dir = Path(args.output_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train images
    print("Processing training images...")
    train_count = process_image_directory(
        args.train_dir, 
        output_dir / "train.json",
        annotation_file=args.train_annotations,
        batch_size=batch_size,
        resume=resume
    )
    
    # Process validation images
    print("Processing validation images...")
    val_count = process_image_directory(
        args.val_dir, 
        output_dir / "val.json",
        annotation_file=args.val_annotations,
        batch_size=batch_size,
        resume=resume
    )
    
    # Save combined summary
    summary = {
        "train_count": train_count,
        "val_count": val_count,
        "total_processed": train_count + val_count
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Complete! Processed {summary['total_processed']} images total.")
    print(f"Results saved to: {output_dir / 'train.json'} and {output_dir / 'val.json'}")


if __name__ == "__main__":
    main()