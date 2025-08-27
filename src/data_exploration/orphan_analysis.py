"""
Orphan Images Analysis Module

This module provides comprehensive analysis of orphan images (images without annotations
or tracking data) in the NightOwls dataset. It combines statistical analysis with
visual sampling to understand the characteristics of unannotated images.

Functions:
    analyze_orphan_images_comprehensive: Unified analysis of orphan images with statistics and visualization
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_orphan_images_comprehensive(images_df: pd.DataFrame, 
                                      annotations_df: pd.DataFrame, 
                                      sequences_discovered: List[Dict[str, Any]], 
                                      images_dir: Path, 
                                      split_name: str, 
                                      num_samples: int = 6) -> Dict[str, Any]:
    """
    Comprehensive analysis of orphan images with statistics and visualization.
    
    This function combines the functionality of:
    - find_images_not_in_sequences()
    - find_images_without_tracking_data()  
    - comprehensive_sequence_coverage_analysis()
    - analyze_orphaned_images_content() visualization
    
    Args:
        images_df: DataFrame containing image information
        annotations_df: DataFrame containing annotation data
        sequences_discovered: List of discovered sequences
        images_dir: Path to the images directory
        split_name: Name of the data split
        num_samples: Number of sample images to visualize
        
    Returns:
        Dictionary containing comprehensive orphan analysis results
    """
    if images_df is None or images_df.empty:
        logger.error(f"No image data available for {split_name}")
        return {}
    
    print(f"\n{split_name.upper()} COMPREHENSIVE ORPHAN IMAGES ANALYSIS")
    print("=" * 60)
    
    # Get all image IDs from the dataset
    all_image_ids = set(images_df['id'].unique())
    total_images = len(all_image_ids)
    
    print(f"Total images in dataset: {total_images:,}")
    
    # Method 1: Images in discovered sequences
    sequence_image_ids = set()
    if sequences_discovered:
        for seq in sequences_discovered:
            sequence_image_ids.update(seq['sequence_images'])
    
    # Method 2: Images with tracking data
    images_with_tracking = set()
    if annotations_df is not None and not annotations_df.empty:
        images_with_tracking = set(
            annotations_df[annotations_df['tracking_id'].notna()]['image_id'].unique()
        )
    
    # Method 3: Images with any annotations
    images_with_annotations = set()
    if annotations_df is not None and not annotations_df.empty:
        images_with_annotations = set(annotations_df['image_id'].unique())
    
    # Calculate orphan categories
    orphaned_from_sequences = all_image_ids - sequence_image_ids
    orphaned_from_tracking = all_image_ids - images_with_tracking
    orphaned_no_annotations = all_image_ids - images_with_annotations
    
    # The truly orphaned images (no sequences, tracking, or annotations)
    truly_orphaned = orphaned_from_sequences & orphaned_from_tracking & orphaned_no_annotations
    
    print(f"\nImage categorization:")
    print(f"  Images in discovered sequences: {len(sequence_image_ids):,}")
    print(f"  Images with tracking data: {len(images_with_tracking):,}")
    print(f"  Images with any annotations: {len(images_with_annotations):,}")
    
    print(f"\nOrphan image analysis:")
    print(f"  Not in any sequence: {len(orphaned_from_sequences):,} ({len(orphaned_from_sequences)/total_images*100:.1f}%)")
    print(f"  No tracking data: {len(orphaned_from_tracking):,} ({len(orphaned_from_tracking)/total_images*100:.1f}%)")
    print(f"  No annotations at all: {len(orphaned_no_annotations):,} ({len(orphaned_no_annotations)/total_images*100:.1f}%)")
    
    print(f"\nTruly orphaned images:")
    print(f"  Completely isolated (no sequences, tracking, or annotations): {len(truly_orphaned):,}")
    
    if len(truly_orphaned) > 0:
        truly_orphaned_percentage = len(truly_orphaned) / total_images * 100
        print(f"  Percentage of total dataset: {truly_orphaned_percentage:.1f}%")
        
        truly_orphaned_df = images_df[images_df['id'].isin(truly_orphaned)]
        print(f"  Sample truly orphaned files:")
        for filename in truly_orphaned_df['file_name'].head(5):
            print(f"    {filename}")
    
    # Overlap analysis
    print(f"\nOverlap analysis:")
    sequences_and_tracking = sequence_image_ids & images_with_tracking
    print(f"  In sequences AND have tracking: {len(sequences_and_tracking):,}")
    
    in_sequences_no_tracking = sequence_image_ids - images_with_tracking
    print(f"  In sequences but NO tracking: {len(in_sequences_no_tracking):,}")
    
    has_tracking_no_sequences = images_with_tracking - sequence_image_ids
    print(f"  Has tracking but NOT in sequences: {len(has_tracking_no_sequences):,}")
    
    # Visualize sample orphaned images
    if len(truly_orphaned) > 0:
        _visualize_orphaned_images(truly_orphaned_df, images_dir, split_name, num_samples)
    else:
        print(f"\nNo truly orphaned images to visualize")
    
    # Create comprehensive results
    results = {
        'split_name': split_name,
        'total_images': total_images,
        'images_in_sequences': len(sequence_image_ids),
        'images_with_tracking': len(images_with_tracking),
        'images_with_annotations': len(images_with_annotations),
        'orphaned_from_sequences': len(orphaned_from_sequences),
        'orphaned_from_tracking': len(orphaned_from_tracking),
        'orphaned_no_annotations': len(orphaned_no_annotations),
        'truly_orphaned': len(truly_orphaned),
        'truly_orphaned_percentage': len(truly_orphaned) / total_images * 100 if total_images > 0 else 0,
        'sequences_and_tracking_match': len(sequences_and_tracking),
        'orphaned_image_ids': list(truly_orphaned),
        'coverage_stats': {
            'sequence_coverage': len(sequence_image_ids) / total_images * 100 if total_images > 0 else 0,
            'tracking_coverage': len(images_with_tracking) / total_images * 100 if total_images > 0 else 0,
            'annotation_coverage': len(images_with_annotations) / total_images * 100 if total_images > 0 else 0
        }
    }
    
    # Summary interpretation
    print(f"\nSummary interpretation:")
    if len(truly_orphaned) > 0:
        print(f"  These {len(truly_orphaned):,} orphaned images likely represent:")
        print(f"    • Negative examples (no objects of interest)")
        print(f"    • Background scenes without pedestrians/bicycles/motorbikes")  
        print(f"    • Low-light scenes where no objects are visible/labelable")
        print(f"  This is normal for object detection datasets")
        
        # Calculate useful/negative example ratio
        useful_images = total_images - len(truly_orphaned)
        if useful_images > 0:
            ratio = len(truly_orphaned) / useful_images
            print(f"  Negative-to-positive ratio: {ratio:.2f}:1")
    else:
        print(f"  All images have annotations - unusual for object detection datasets")
    
    return results


def _visualize_orphaned_images(orphaned_images_df: pd.DataFrame, 
                              images_dir: Path, 
                              split_name: str, 
                              num_samples: int) -> None:
    """
    Visualize sample orphaned images to understand their characteristics.
    
    Args:
        orphaned_images_df: DataFrame containing orphaned image information
        images_dir: Path to the images directory  
        split_name: Name of the data split
        num_samples: Number of images to visualize
    """
    if orphaned_images_df.empty:
        return
    
    print(f"\nVisualizing sample orphaned images:")
    
    # Sample orphaned images
    sample_size = min(num_samples, len(orphaned_images_df))
    orphaned_sample = orphaned_images_df.sample(sample_size)
    
    # Create visualization
    num_cols = 3
    num_rows = (sample_size + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
    fig.suptitle(f'{split_name} - Sample Orphaned Images (No Annotations)', fontsize=14)
    
    # Handle axes array
    if sample_size == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten() if sample_size > 1 else axes
    
    for idx, (_, img_info) in enumerate(orphaned_sample.iterrows()):
        ax = axes_flat[idx]
        img_path = images_dir / img_info['file_name']
        
        if img_path.exists():
            try:
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                
                # Add image info
                title = f"ID: {img_info['id']}\n{img_info['file_name']}"
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                
            except Exception as e:
                logger.warning(f"Error loading {img_info['file_name']}: {e}")
                ax.text(0.5, 0.5, f"Error loading:\n{img_info['file_name']}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Error - ID: {img_info['id']}", fontsize=10)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"File not found:\n{img_info['file_name']}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Missing - ID: {img_info['id']}", fontsize=10)
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(sample_size, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_orphan_summary(orphan_results: Dict[str, Any]) -> str:
    """
    Generate a text summary of orphan analysis results.
    
    Args:
        orphan_results: Results dictionary from analyze_orphan_images_comprehensive
        
    Returns:
        Formatted text summary of the analysis
    """
    if not orphan_results:
        return "No orphan analysis results available"
    
    split_name = orphan_results.get('split_name', 'Unknown')
    total = orphan_results.get('total_images', 0)
    orphaned = orphan_results.get('truly_orphaned', 0)
    percentage = orphan_results.get('truly_orphaned_percentage', 0)
    
    coverage = orphan_results.get('coverage_stats', {})
    seq_coverage = coverage.get('sequence_coverage', 0)
    track_coverage = coverage.get('tracking_coverage', 0)
    ann_coverage = coverage.get('annotation_coverage', 0)
    
    summary = f"""
{split_name} Orphan Images Summary:
- Total images: {total:,}
- Orphaned images: {orphaned:,} ({percentage:.1f}%)
- Sequence coverage: {seq_coverage:.1f}%
- Tracking coverage: {track_coverage:.1f}%
- Annotation coverage: {ann_coverage:.1f}%

This indicates that {percentage:.1f}% of images serve as negative examples,
which is typical for object detection datasets.
    """.strip()
    
    return summary