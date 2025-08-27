"""
Bounding Box Analysis Module

This module provides functions for analyzing bounding box properties and quality
in the NightOwls dataset. It focuses on statistical analysis of bbox dimensions,
coordinates, and identifying problematic annotations.

Functions:
    analyze_bounding_boxes: Analyze bounding box properties and size distributions
    investigate_negative_bbox_values: Investigate annotations with negative coordinates
    analyze_bbox_coordinates_distribution: Analyze coordinate distribution patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_bounding_boxes(annotations_df: pd.DataFrame, categories: Dict[int, str], 
                          split_name: str) -> Optional[pd.DataFrame]:
    """
    Analyze bounding box properties in detail.
    
    Args:
        annotations_df: DataFrame containing annotation data
        categories: Dictionary mapping category IDs to names
        split_name: Name of the data split
        
    Returns:
        DataFrame with detailed bounding box analysis, or None if no valid data
    """
    if annotations_df is None or annotations_df.empty:
        logger.error(f"No annotations available for {split_name}")
        return None
    
    print(f"\n{split_name.upper()} BOUNDING BOX ANALYSIS")
    print("-" * 40)
    
    # Extract bounding box data
    bbox_data = []
    for _, row in annotations_df.iterrows():
        if 'bbox' in row and row['bbox'] is not None and len(row['bbox']) >= 4:
            x, y, w, h = row['bbox'][:4]
            if w > 0 and h > 0:  # Only valid bboxes
                bbox_data.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'center_x': x + w/2, 'center_y': y + h/2,
                    'area': w * h,
                    'aspect_ratio': w / h,
                    'category_id': row.get('category_id', -1),
                    'ignore': row.get('ignore', 0)
                })
    
    if not bbox_data:
        logger.warning("No valid bounding boxes found")
        return None
    
    bbox_df = pd.DataFrame(bbox_data)
    
    # Overall statistics
    print(f"Total valid bounding boxes: {len(bbox_df):,}")
    
    # Size statistics
    print(f"\nSize statistics:")
    print(f"  Width  - Min: {bbox_df['width'].min():.1f} | Max: {bbox_df['width'].max():.1f} | Mean: {bbox_df['width'].mean():.1f}")
    print(f"  Height - Min: {bbox_df['height'].min():.1f} | Max: {bbox_df['height'].max():.1f} | Mean: {bbox_df['height'].mean():.1f}")
    print(f"  Area   - Min: {bbox_df['area'].min():.0f} | Max: {bbox_df['area'].max():.0f} | Mean: {bbox_df['area'].mean():.0f}")
    
    # Aspect ratio analysis
    print(f"\nAspect ratio analysis:")
    aspect_stats = bbox_df['aspect_ratio'].describe()
    print(f"  Min: {aspect_stats['min']:.3f} | Max: {aspect_stats['max']:.3f} | Mean: {aspect_stats['mean']:.3f}")
    
    # Size categories
    bbox_df['size_category'] = pd.cut(bbox_df['area'], 
                                     bins=[0, 1000, 2000, 3000, 4000, 5000, 10000, 20000, float('inf')],
                                     labels=['0-1K', '1-2K', '2-3K', '3-4K', '4-5K', '5-10K', '10-20K', '20K+'])
    
    print(f"\nSize distribution:")
    size_dist = bbox_df['size_category'].value_counts()
    
    # Define the correct order for size categories
    size_order = ['0-1K', '1-2K', '2-3K', '3-4K', '4-5K', '5-10K', '10-20K', '20K+']
    
    # Display in the correct order
    for category in size_order:
        if category in size_dist.index:
            count = size_dist[category]
            percentage = count / len(bbox_df) * 100
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
    
    # Category-specific analysis
    if categories:
        print(f"\nCategory-specific analysis:")
        for cat_id, cat_name in categories.items():
            cat_boxes = bbox_df[bbox_df['category_id'] == cat_id]
            if len(cat_boxes) > 0:
                print(f"  {cat_name} (ID {cat_id}): {len(cat_boxes):,} boxes")
                print(f"    Average size: {cat_boxes['area'].mean():.0f} pixels²")
                print(f"    Average aspect ratio: {cat_boxes['aspect_ratio'].mean():.3f}")
                
                # Size distribution for this category
                cat_size_dist = cat_boxes['size_category'].value_counts()
                print(f"    Size breakdown:")
                # Order the category-specific breakdown
                for size_cat in size_order:
                    if size_cat in cat_size_dist.index:
                        count = cat_size_dist[size_cat]
                        cat_percentage = count / len(cat_boxes) * 100
                        print(f"      {size_cat}: {count:,} ({cat_percentage:.1f}%)")
    
    return bbox_df


def investigate_negative_bbox_values(annotations_df: pd.DataFrame, split_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Investigate negative bbox values and analyze their category distribution.
    
    Args:
        annotations_df: DataFrame containing annotation data
        split_name: Name of the data split
        
    Returns:
        List of dictionaries containing negative bbox issues, or None if no issues found
    """
    print(f"\n{split_name.upper()} NEGATIVE BBOX VALUES INVESTIGATION")
    print("-" * 50)
    
    if annotations_df is None or annotations_df.empty:
        logger.error(f"No annotation data available for {split_name}")
        return None
    
    # Find annotations with negative values
    negative_bbox_annotations = []
    
    for idx, row in annotations_df.iterrows():
        if 'bbox' in row and row['bbox'] is not None and len(row['bbox']) >= 4:
            x, y, w, h = row['bbox'][:4]
            if w < 0 or h < 0 or x < 0 or y < 0:
                # Calculate area for size analysis (use absolute values for area calculation)
                area = abs(w) * abs(h) if w != 0 and h != 0 else 0
                
                negative_bbox_annotations.append({
                    'annotation_id': row.get('id', 'unknown'),
                    'image_id': row.get('image_id', 'unknown'),
                    'category_id': row.get('category_id', 'unknown'),
                    'tracking_id': row.get('tracking_id', 'unknown'),
                    'bbox': [x, y, w, h],
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area,
                    'issue': []
                })
                
                # Classify the issues
                if x < 0:
                    negative_bbox_annotations[-1]['issue'].append('negative_x')
                if y < 0:
                    negative_bbox_annotations[-1]['issue'].append('negative_y')
                if w < 0:
                    negative_bbox_annotations[-1]['issue'].append('negative_width')
                if h < 0:
                    negative_bbox_annotations[-1]['issue'].append('negative_height')
    
    print(f"Found {len(negative_bbox_annotations):,} annotations with negative values")
    
    if not negative_bbox_annotations:
        print("No negative bbox issues found")
        return None
    
    # Analyze the issues by type
    issue_counts = defaultdict(int)
    for ann in negative_bbox_annotations:
        for issue in ann['issue']:
            issue_counts[issue] += 1
    
    print(f"\nIssue breakdown:")
    for issue, count in issue_counts.items():
        percentage = count / len(negative_bbox_annotations) * 100
        print(f"  {issue.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
    
    # Analyze by category
    category_issues = defaultdict(list)
    issue_type_by_category = defaultdict(lambda: defaultdict(int))
    
    for issue in negative_bbox_annotations:
        cat_id = issue.get('category_id', 'unknown')
        category_issues[cat_id].append(issue)
        
        # Count issue types per category
        for issue_type in issue['issue']:
            issue_type_by_category[cat_id][issue_type] += 1
    
    # Display category breakdown
    category_names = {1: 'Pedestrian', 2: 'Bicycle', 3: 'Motorbike', 4: 'Ignore'}
    total_negative_issues = len(negative_bbox_annotations)
    
    print(f"\nBreakdown by category:")
    
    for cat_id, issues in category_issues.items():
        cat_name = category_names.get(cat_id, f'Category_{cat_id}')
        count = len(issues)
        percentage = count / total_negative_issues * 100
        
        print(f"\n  {cat_name} (ID {cat_id}): {count:,} issues ({percentage:.1f}%)")
        
        # Show issue types for this category
        issue_types = issue_type_by_category[cat_id]
        for issue_type, type_count in issue_types.items():
            type_percentage = type_count / count * 100
            print(f"    {issue_type}: {type_count:,} ({type_percentage:.1f}%)")
    
    # Size breakdown analysis
    print(f"\nBreakdown by size:")
    
    # Convert to DataFrame for easier analysis
    negative_df = pd.DataFrame(negative_bbox_annotations)
    
    # Add size categories using specified ranges
    negative_df['size_category'] = pd.cut(negative_df['area'], 
                                         bins=[0, 1000, 2000, 3000, 4000, 5000, 10000, 20000, float('inf')],
                                         labels=['0-1K', '1-2K', '2-3K', '3-4K', '4-5K', '5-10K', '10-20K', '20K+'])
    
    # Define the correct order for size categories
    size_order = ['0-1K', '1-2K', '2-3K', '3-4K', '4-5K', '5-10K', '10-20K', '20K+']
    
    # Get size distribution
    size_dist = negative_df['size_category'].value_counts()
    
    # Display in the correct order
    for size_cat in size_order:
        if size_cat in size_dist.index:
            count = size_dist[size_cat]
            percentage = count / len(negative_df) * 100
            print(f"  {size_cat}: {count:,} ({percentage:.1f}%)")
            
            # Show breakdown by issue type for this size category
            size_issues = negative_df[negative_df['size_category'] == size_cat]
            size_issue_types = defaultdict(int)
            for _, row in size_issues.iterrows():
                for issue_type in row['issue']:
                    size_issue_types[issue_type] += 1
            
            if size_issue_types:
                print(f"    Issue types:")
                for issue_type, type_count in size_issue_types.items():
                    type_percentage = type_count / count * 100
                    print(f"      {issue_type}: {type_count:,} ({type_percentage:.1f}%)")
    
    # Size statistics for negative bboxes
    valid_areas = negative_df[negative_df['area'] > 0]['area']
    if len(valid_areas) > 0:
        print(f"\nSize statistics for negative bboxes:")
        print(f"  Min area: {valid_areas.min():.0f} pixels²")
        print(f"  Max area: {valid_areas.max():.0f} pixels²")
        print(f"  Mean area: {valid_areas.mean():.0f} pixels²")
        print(f"  Median area: {valid_areas.median():.0f} pixels²")
        print(f"  Zero area count: {len(negative_df[negative_df['area'] == 0]):,}")
    
    return negative_bbox_annotations


def analyze_bbox_coordinates_distribution(annotations_df: pd.DataFrame, split_name: str) -> Optional[Dict[str, Any]]:
    """
    Analyze the distribution of x and y coordinates in bounding boxes.
    
    Args:
        annotations_df: DataFrame containing annotation data
        split_name: Name of the data split
        
    Returns:
        Dictionary containing coordinate distribution analysis, or None if no valid data
    """
    if annotations_df is None or annotations_df.empty:
        logger.error(f"No annotations available for {split_name}")
        return None
    
    print(f"\n{split_name.upper()} BOUNDING BOX COORDINATES DISTRIBUTION")
    print("-" * 50)
    
    # Extract coordinate data
    x_coords = []
    y_coords = []
    
    for _, row in annotations_df.iterrows():
        if 'bbox' in row and row['bbox'] is not None and len(row['bbox']) >= 4:
            x, y, w, h = row['bbox'][:4]
            x_coords.append(x)
            y_coords.append(y)
    
    if not x_coords:
        logger.warning("No valid bounding box coordinates found")
        return None
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # Statistics
    print(f"Total bounding boxes analyzed: {len(x_coords):,}")
    
    print(f"\nX-coordinate statistics:")
    print(f"  Min: {x_coords.min():.1f}")
    print(f"  Max: {x_coords.max():.1f}")
    print(f"  Mean: {x_coords.mean():.1f}")
    print(f"  Median: {np.median(x_coords):.1f}")
    print(f"  Std: {x_coords.std():.1f}")
    print(f"  Negative count: {np.sum(x_coords < 0):,} ({np.sum(x_coords < 0)/len(x_coords)*100:.1f}%)")
    
    print(f"\nY-coordinate statistics:")
    print(f"  Min: {y_coords.min():.1f}")
    print(f"  Max: {y_coords.max():.1f}")
    print(f"  Mean: {y_coords.mean():.1f}")
    print(f"  Median: {np.median(y_coords):.1f}")
    print(f"  Std: {y_coords.std():.1f}")
    print(f"  Negative count: {np.sum(y_coords < 0):,} ({np.sum(y_coords < 0)/len(y_coords)*100:.1f}%)")
    
    # Negative coordinates analysis
    if np.sum(x_coords < 0) > 0 or np.sum(y_coords < 0) > 0:
        print(f"\nNegative coordinates analysis:")
        
        if np.sum(x_coords < 0) > 0:
            negative_x = x_coords[x_coords < 0]
            print(f"  Negative X coordinates:")
            print(f"    Count: {len(negative_x):,}")
            print(f"    Min: {negative_x.min():.1f}")
            print(f"    Max: {negative_x.max():.1f}")
            print(f"    Mean: {negative_x.mean():.1f}")
        
        if np.sum(y_coords < 0) > 0:
            negative_y = y_coords[y_coords < 0]
            print(f"  Negative Y coordinates:")
            print(f"    Count: {len(negative_y):,}")
            print(f"    Min: {negative_y.min():.1f}")
            print(f"    Max: {negative_y.max():.1f}")
            print(f"    Mean: {negative_y.mean():.1f}")
    
    # Percentile analysis
    print(f"\nPercentile analysis:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    print(f"  X-coordinate percentiles:")
    for p in percentiles:
        val = np.percentile(x_coords, p)
        print(f"    {p:2d}th: {val:7.1f}")
    
    print(f"  Y-coordinate percentiles:")
    for p in percentiles:
        val = np.percentile(y_coords, p)
        print(f"    {p:2d}th: {val:7.1f}")
    
    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'x_stats': {
            'min': x_coords.min(),
            'max': x_coords.max(),
            'mean': x_coords.mean(),
            'std': x_coords.std(),
            'negative_count': np.sum(x_coords < 0)
        },
        'y_stats': {
            'min': y_coords.min(),
            'max': y_coords.max(),
            'mean': y_coords.mean(),
            'std': y_coords.std(),
            'negative_count': np.sum(y_coords < 0)
        }
    }