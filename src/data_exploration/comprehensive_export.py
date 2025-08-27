"""
Comprehensive Data Export Module

This module provides unified data export functionality that combines sequence analysis
with problematic bounding box analysis. It creates comprehensive DataFrames and CSV
exports that integrate information from multiple analysis stages.

Functions:
    create_comprehensive_sequence_dataframe: Unified function combining sequence and bbox problem analysis
    export_comprehensive_analysis: Export comprehensive analysis to CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_comprehensive_sequence_dataframe(sequences_discovered: List[Dict[str, Any]], 
                                          annotations_df: pd.DataFrame,
                                          images_df: pd.DataFrame,
                                          negative_issues: List[Dict[str, Any]] = None,
                                          split_name: str = "") -> pd.DataFrame:
    """
    Create comprehensive sequence dataframe with original create_sequence_dataframe format plus problematic analysis.
    
    This function returns the original sequence-level structure (one row per sequence) with added columns:
    - category_counts: Count of annotations per category
    - problematic_category_counts: Count of unique annotations with coordinate issues per category
    - category_specific_issues: Dict of category -> issue types with counts (negative_x, negative_y, etc.)
    - frames_completely_outside: Dict of category -> frame positions with completely outside bboxes
    - frames_partially_outside: Dict of category -> frame positions with partially outside bboxes
    
    Granularity: One row per sequence_id
    
    Args:
        sequences_discovered: List of discovered sequences from sequence_discovery module
        annotations_df: DataFrame containing annotation data
        images_df: DataFrame containing image information (for bbox visibility analysis)
        negative_issues: List of negative bbox issues from bbox_analysis module (unused, kept for compatibility)
        split_name: Name of the data split
        
    Returns:
        DataFrame with sequence-level analysis including problematic bbox information
    """
    if not sequences_discovered:
        logger.error("No sequences provided")
        return pd.DataFrame()
    
    print(f"\n{split_name.upper()} COMPREHENSIVE SEQUENCE DATAFRAME CREATION")
    print("-" * 60)
    
    # Category mapping
    category_names = {1: 'Pedestrian', 2: 'Bicycle', 3: 'Motorbike', 4: 'Ignore'}
    
    # Create mapping of image_id to image dimensions for bbox visibility analysis
    image_dimensions = {}
    if images_df is not None and not images_df.empty:
        for _, img_row in images_df.iterrows():
            image_dimensions[img_row['id']] = {
                'width': img_row.get('width', 0),
                'height': img_row.get('height', 0)
            }
        print(f"Loaded dimensions for {len(image_dimensions)} images")
    
    # Helper functions for bbox analysis
    def analyze_bbox_issues(bbox):
        """Analyze a bbox for coordinate issues (actual problems)."""
        if not bbox or len(bbox) < 4:
            return []
        
        x, y, w, h = bbox[:4]
        issues = []
        
        # Check for negative coordinates (these are the actual issues)
        if x < 0:
            issues.append('negative_x')
        if y < 0:
            issues.append('negative_y')
        if w < 0:
            issues.append('negative_width')
        if h < 0:
            issues.append('negative_height')
        
        return issues
    
    def analyze_bbox_visibility(bbox, img_width, img_height):
        """Analyze bbox visibility status (not issues, just analysis)."""
        if not bbox or len(bbox) < 4 or img_width <= 0 or img_height <= 0:
            return None
        
        x, y, w, h = bbox[:4]
        
        # Completely outside
        if (x + w <= 0) or (x >= img_width) or (y + h <= 0) or (y >= img_height):
            return 'completely_outside'
        # Partially outside (but not completely outside)
        elif (x < 0 and x + w > 0) or (y < 0 and y + h > 0) or \
             (x < img_width and x + w > img_width) or (y < img_height and y + h > img_height):
            return 'partially_outside'
        # Completely inside
        else:
            return 'inside'
    
    # Process each sequence to create sequence-level dataframe
    sequence_data = []
    
    for sequence in sequences_discovered:
        sequence_id = sequence['sequence_id']
        tracking_ids = sequence['tracking_ids']
        image_ids = sequence['sequence_images']
        
        # Get all annotations for this sequence
        seq_annotations = annotations_df[
            annotations_df['tracking_id'].isin(tracking_ids)
        ].copy() if annotations_df is not None else pd.DataFrame()
        
        # Create mapping of image_id to frame position in sequence (1-based)
        image_to_frame_map = {}
        for idx, image_id in enumerate(image_ids):
            image_to_frame_map[image_id] = idx + 1
        
        if not seq_annotations.empty:
            # Map category_id to category names
            seq_annotations['category'] = seq_annotations['category_id'].map(category_names)
            
            # Get unique categories in this sequence
            categories = sorted(seq_annotations['category'].unique())
            
            # Get category distribution (category_counts)
            category_counts = seq_annotations['category'].value_counts().to_dict()
            
            # Initialize analysis dictionaries
            problematic_category_counts = {}
            category_specific_issues = {}
            frames_completely_outside = {}
            frames_partially_outside = {}
            
            # Analyze each annotation
            for _, ann in seq_annotations.iterrows():
                if 'bbox' in ann and ann['bbox'] is not None and len(ann['bbox']) >= 4:
                    category = ann['category']
                    image_id = ann.get('image_id')
                    frame_position = image_to_frame_map.get(image_id)
                    
                    # Get image dimensions
                    img_dims = image_dimensions.get(image_id, {'width': 0, 'height': 0})
                    img_width = img_dims['width']
                    img_height = img_dims['height']
                    
                    # Analyze bbox for coordinate issues (actual problems)
                    issues = analyze_bbox_issues(ann['bbox'])
                    
                    if issues:
                        # Count annotations with coordinate issues by category
                        if category not in problematic_category_counts:
                            problematic_category_counts[category] = 0
                        problematic_category_counts[category] += 1
                        
                        # Collect issue types by category
                        if category not in category_specific_issues:
                            category_specific_issues[category] = defaultdict(int)
                        for issue_type in issues:
                            category_specific_issues[category][issue_type] += 1
                    
                    # Analyze bbox visibility status (separate from issues)
                    visibility = analyze_bbox_visibility(ann['bbox'], img_width, img_height)
                    
                    if visibility == 'completely_outside':
                        if category not in frames_completely_outside:
                            frames_completely_outside[category] = set()
                        if frame_position is not None:
                            frames_completely_outside[category].add(frame_position)
                    
                    elif visibility == 'partially_outside':
                        if category not in frames_partially_outside:
                            frames_partially_outside[category] = set()
                        if frame_position is not None:
                            frames_partially_outside[category].add(frame_position)
            
            # Format the category_specific_issues dict
            formatted_category_issues = {}
            for category, issue_counts in category_specific_issues.items():
                issue_strings = []
                for issue_type, count in issue_counts.items():
                    issue_strings.append(f"{issue_type}({count})")
                formatted_category_issues[category] = ', '.join(issue_strings)
            
            # Format the visibility frames dicts (convert sets to sorted comma-separated strings)
            formatted_frames_completely_outside = {}
            for category, frame_set in frames_completely_outside.items():
                formatted_frames_completely_outside[category] = ','.join(map(str, sorted(frame_set)))
            
            formatted_frames_partially_outside = {}
            for category, frame_set in frames_partially_outside.items():
                formatted_frames_partially_outside[category] = ','.join(map(str, sorted(frame_set)))
            
        else:
            categories = []
            category_counts = {}
            problematic_category_counts = {}
            formatted_category_issues = {}
            formatted_frames_completely_outside = {}
            formatted_frames_partially_outside = {}
        
        # Create sequence-level record
        sequence_data.append({
            'sequence_id': sequence_id,
            'num_tracking_ids': len(tracking_ids),
            'tracking_ids': tracking_ids,
            'image_ids': image_ids,
            'categories': categories,
            'num_categories': len(categories),
            'seed_tracking_id': sequence.get('seed_tracking_id', tracking_ids[0] if tracking_ids else None),
            'total_frames': len(image_ids),
            'category_counts': category_counts,
            'problematic_category_counts': problematic_category_counts,
            'category_specific_issues': formatted_category_issues,
            'frames_completely_outside': formatted_frames_completely_outside,
            'frames_partially_outside': formatted_frames_partially_outside
        })
    
    # Create DataFrame
    if sequence_data:
        df = pd.DataFrame(sequence_data)
        
        # Sort by total frames (most first), then by number of problematic categories (most first)
        df['total_problematic_annotations'] = df['problematic_category_counts'].apply(lambda x: sum(x.values()) if x else 0)
        df = df.sort_values(['total_frames', 'total_problematic_annotations'], ascending=[False, False])
        
        print(f"Created comprehensive dataframe: {len(df)} rows (sequences)")
        print(f"Total sequences: {len(df)}")
        
        return df
    else:
        logger.warning("No data to create comprehensive dataframe")
        return pd.DataFrame()


def export_comprehensive_analysis(df: pd.DataFrame, 
                                output_dir: Path = None, 
                                split_name: str = "") -> List[Path]:
    """
    Export comprehensive analysis DataFrame to CSV files.
    
    Args:
        df: Comprehensive DataFrame from create_comprehensive_sequence_dataframe
        output_dir: Output directory for CSV files
        split_name: Name of the data split
        
    Returns:
        List of paths to saved CSV files
    """
    if df is None or df.empty:
        logger.error("No data to export")
        return []
    
    # Set default output directory
    if output_dir is None:
        output_dir = Path('data/processed/comprehensive_analysis')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEXPORTING COMPREHENSIVE ANALYSIS")
    print("-" * 40)
    print(f"Output directory: {output_dir}")
    
    saved_files = []
    
    # Main comprehensive file
    main_csv_path = output_dir / f'{split_name.lower()}_comprehensive_sequences.csv'
    df.to_csv(main_csv_path, index=False)
    print(f"Saved comprehensive analysis: {main_csv_path}")
    print(f"  Records: {len(df)}")
    saved_files.append(main_csv_path)
    
    # Create summary file
    summary_data = []
    for _, row in df.iterrows():
        summary_data.append({
            'sequence_id': row['sequence_id'],
            'total_frames': row['total_frames'],
            'num_tracking_ids': row['num_tracking_ids'],
            'categories': ', '.join(row['categories']) if isinstance(row['categories'], list) else row['categories'],
            'has_problems': bool(row['problematic_category_counts']),
            'total_problematic_annotations': row.get('total_problematic_annotations', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_dir / f'{split_name.lower()}_sequence_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved sequence summary: {summary_csv_path}")
    print(f"  Sequences: {len(summary_df)}")
    saved_files.append(summary_csv_path)
    
    # Create README
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write("# Comprehensive Sequence Analysis\n\n")
        f.write("This directory contains comprehensive analysis with original create_sequence_dataframe format plus problematic bbox analysis.\n\n")
        f.write("## Files:\n")
        for file_path in saved_files:
            f.write(f"- `{file_path.name}`: {file_path.name.replace('_', ' ').replace('.csv', '').title()}\n")
        f.write("\n## Granularity:\n")
        f.write("Each row represents one sequence_id with comprehensive problematic analysis by category.\n\n")
        f.write("## Key Columns:\n")
        f.write("- Basic info: sequence_id, num_tracking_ids, total_frames, categories, category_counts\n")
        f.write("- Problematic analysis: problematic_category_counts, category_specific_issues\n")
        f.write("- Visibility analysis: frames_completely_outside, frames_partially_outside\n")
        f.write("\n## Important Note:\n")
        f.write("- Issues are coordinate problems (negative_x, negative_y, negative_width, negative_height)\n")
        f.write("- Visibility status (completely/partially outside) is separate analysis, not an issue\n")
        f.write("- problematic_category_counts: Count of unique annotations with coordinate issues\n")
        f.write("- category_specific_issues: Count of individual issue types (can be > annotations since one annotation can have multiple issues)\n")
    
    print(f"Created README: {readme_path}")
    
    return saved_files

