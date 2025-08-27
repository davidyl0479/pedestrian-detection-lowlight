"""
Sequence Discovery Module

This module provides core algorithms for discovering video sequences in the NightOwls dataset
by analyzing tracking IDs and their relationships across images. It uses graph traversal
techniques to identify connected components of tracking IDs that appear together in frames.

These functions serve as building blocks for more comprehensive analysis in other modules.

Functions:
    discover_sequences_from_tracking_id: Discover complete sequence from a seed tracking ID
    discover_all_sequences_comprehensive: Find all video sequences in the dataset
    analyze_discovered_sequences_simple: Provide basic statistics on discovered sequences
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_sequences_from_tracking_id(df: pd.DataFrame, seed_tracking_id: int, 
                                      verbose: bool = True) -> Dict[str, Any]:
    """
    Discover complete sequence from a seed tracking ID through iterative expansion.
    
    This function uses a graph traversal approach to find all images and tracking IDs
    that are connected to the seed tracking ID through shared frames.
    
    Process:
    1. Start with seed tracking ID → get all its images
    2. Find ALL other tracking IDs in those images
    3. Get ALL images from those tracking IDs  
    4. Repeat until no new images/tracking IDs are found (closure)
    
    Args:
        df: DataFrame containing annotations with 'tracking_id' and 'image_id' columns
        seed_tracking_id: Starting tracking ID for sequence discovery
        verbose: Whether to print iteration details
        
    Returns:
        Dictionary containing sequence information:
        - sequence_id: Unique identifier for the sequence
        - seed_tracking_id: Original seed tracking ID
        - sequence_images: List of all image IDs in the sequence
        - tracking_ids: List of all tracking IDs in the sequence
        - total_frames: Number of unique images in the sequence
        - total_objects: Number of unique tracking IDs in the sequence
        - iterations_to_converge: Number of iterations needed to find closure
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return {}
    
    if 'tracking_id' not in df.columns or 'image_id' not in df.columns:
        logger.error("DataFrame must contain 'tracking_id' and 'image_id' columns")
        return {}
    
    # Step 1: Get all images for the seed tracking ID
    seed_images = set(df[df['tracking_id'] == seed_tracking_id]['image_id'].unique())
    
    if not seed_images:
        logger.warning(f"No images found for tracking ID {seed_tracking_id}")
        return {}
    
    # Step 2: Find ALL tracking IDs that appear in those images
    related_tracking_ids = set(df[df['image_id'].isin(seed_images)]['tracking_id'].unique())
    
    # Step 3: Get ALL images from those tracking IDs (expand the sequence)
    all_sequence_images = set(df[df['tracking_id'].isin(related_tracking_ids)]['image_id'].unique())
    
    # Step 4: Repeat until no new images are found (closure)
    prev_size = 0
    iteration = 0
    
    while len(all_sequence_images) != prev_size:
        iteration += 1
        prev_size = len(all_sequence_images)
        
        # Find all tracking IDs in current image set
        all_tracking_ids = set(df[df['image_id'].isin(all_sequence_images)]['tracking_id'].unique())
        
        # Expand image set with all images from those tracking IDs
        all_sequence_images = set(df[df['tracking_id'].isin(all_tracking_ids)]['image_id'].unique())
        
        if verbose:
            print(f"    Iteration {iteration}: {len(all_tracking_ids)} tracking IDs, {len(all_sequence_images)} images")
    
    return {
        'sequence_id': f"seq_{seed_tracking_id}",
        'seed_tracking_id': seed_tracking_id,
        'sequence_images': sorted(all_sequence_images),
        'tracking_ids': sorted(all_tracking_ids),
        'total_frames': len(all_sequence_images),
        'total_objects': len(all_tracking_ids),
        'iterations_to_converge': iteration
    }


def discover_all_sequences_comprehensive(df: pd.DataFrame, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Discover all video sequences in the dataset by finding connected components
    of tracking IDs and images. Ensures no tracking ID is missed.
    
    Args:
        df: DataFrame containing annotations with 'tracking_id' and 'image_id' columns
        verbose: Whether to print detailed progress information
        
    Returns:
        List of dictionaries, each containing sequence information
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return []
    
    if 'tracking_id' not in df.columns or 'image_id' not in df.columns:
        logger.error("DataFrame must contain 'tracking_id' and 'image_id' columns")
        return []
    
    processed_tracking_ids = set()
    sequences = []
    all_tracking_ids = sorted(df['tracking_id'].unique())
    
    print(f"SEQUENCE DISCOVERY ANALYSIS")
    print("-" * 40)
    print(f"Total tracking IDs to process: {len(all_tracking_ids):,}")
    print(f"Total images in dataset: {df['image_id'].nunique():,}")
    
    for i, tracking_id in enumerate(all_tracking_ids):
        if tracking_id not in processed_tracking_ids:
            # Discover sequence starting from this tracking ID
            sequence_info = discover_sequences_from_tracking_id(df, tracking_id, verbose=verbose)
            
            if sequence_info:  # Only add non-empty sequences
                sequences.append(sequence_info)
                
                # Mark all tracking IDs in this sequence as processed
                processed_tracking_ids.update(sequence_info['tracking_ids'])
                
                # Print progress for significant sequences
                if len(sequences) <= 5 or len(sequence_info['tracking_ids']) > 10:
                    tracking_ids_display = (sequence_info['tracking_ids'] 
                                          if len(sequence_info['tracking_ids']) <= 10 
                                          else f"{len(sequence_info['tracking_ids'])} tracking IDs")
                    print(f"Sequence {len(sequences)}: {tracking_ids_display}, {len(sequence_info['sequence_images'])} images")
    
    print(f"\nSequence discovery complete: {len(sequences)} distinct sequences found")
    print(f"Processed all {len(all_tracking_ids):,} tracking IDs")
    
    return sequences


def analyze_discovered_sequences_simple(sequences: List[Dict[str, Any]], 
                                       dataset_name: str) -> Dict[str, Any]:
    """
    Simple analysis of discovered sequences without detailed breakdowns.
    
    Args:
        sequences: List of sequence dictionaries from discover_all_sequences_comprehensive
        dataset_name: Name of the dataset (e.g., "TRAINING", "VALIDATION")
        
    Returns:
        Dictionary containing sequence analysis statistics
    """
    if not sequences:
        logger.warning(f"No sequences provided for analysis: {dataset_name}")
        return {}
    
    total_sequences = len(sequences)
    total_tracking_ids = sum(len(seq['tracking_ids']) for seq in sequences)
    total_images = sum(len(seq['sequence_images']) for seq in sequences)
    avg_tracking_ids = total_tracking_ids / total_sequences if total_sequences > 0 else 0
    avg_images = total_images / total_sequences if total_sequences > 0 else 0
    
    print(f"\n{dataset_name.upper()} SEQUENCE ANALYSIS")
    print("-" * 40)
    print(f"Total sequences: {total_sequences:,}")
    print(f"Total tracking IDs: {total_tracking_ids:,}")
    print(f"Total images: {total_images:,}")
    print(f"Average tracking IDs per sequence: {avg_tracking_ids:.2f}")
    print(f"Average images per sequence: {avg_images:.2f}")
    
    return {
        'total_sequences': total_sequences,
        'total_tracking_ids': total_tracking_ids,
        'total_images': total_images,
        'avg_tracking_ids_per_sequence': avg_tracking_ids,
        'avg_images_per_sequence': avg_images
    }


