"""
JSON Structure Analysis Module

This module provides functions for loading, analyzing, and validating JSON annotation files
from the NightOwls dataset. It handles the core data structure exploration and creates
pandas DataFrames for further analysis.

Functions:
    load_json_safe: Safely load JSON files with comprehensive error handling
    explore_json_structure: Analyze and display JSON data structure
    create_analysis_dataframes: Convert JSON data to pandas DataFrames
    analyze_annotation_distribution: Analyze category distribution in annotations
    validate_data_consistency: Validate data consistency across splits
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_safe(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Safely load JSON file with comprehensive error handling.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing JSON data, or None if loading fails
        
    Raises:
        Logs errors but does not raise exceptions
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded: {file_path.name}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return None
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        return None


def explore_json_structure(data: Dict[str, Any], name: str, max_examples: int = 3) -> None:
    """
    Comprehensive exploration and display of JSON data structure.
    
    Args:
        data: Dictionary containing JSON data
        name: Name identifier for the dataset (e.g., "TRAINING", "VALIDATION")
        max_examples: Maximum number of examples to display per section
    """
    if not data:
        logger.warning(f"No data available for analysis: {name}")
        return
    
    print(f"\n{name.upper()} JSON STRUCTURE ANALYSIS")
    print("=" * 60)
    
    print(f"Top-level keys: {list(data.keys())}")
    
    # Explore each top-level section
    for key in data.keys():
        print(f"\n{key.upper()} SECTION:")
        value = data[key]
        
        if isinstance(value, list):
            print(f"  Type: List with {len(value):,} items")
            if len(value) > 0:
                first_item = value[0]
                print(f"  First item type: {type(first_item).__name__}")
                
                if isinstance(first_item, dict):
                    print(f"  Available fields: {list(first_item.keys())}")
                    
                    # Show detailed examples
                    for i, item in enumerate(value[:max_examples]):
                        print(f"\n  Example {i+1}:")
                        for field, field_value in item.items():
                            if isinstance(field_value, (str, int, float, bool, type(None))):
                                # Truncate long strings
                                if isinstance(field_value, str) and len(field_value) > 50:
                                    display_value = field_value[:50] + "..."
                                else:
                                    display_value = field_value
                                print(f"    {field}: {display_value}")
                            elif isinstance(field_value, list):
                                preview = field_value[:2] if len(field_value) > 0 else []
                                print(f"    {field}: [list with {len(field_value)} items] -> {preview}...")
                            elif isinstance(field_value, dict):
                                print(f"    {field}: [dict with keys: {list(field_value.keys())}]")
                            else:
                                print(f"    {field}: {type(field_value).__name__}")
                else:
                    # Handle non-dict items in list
                    sample_items = value[:5]
                    print(f"  Sample items: {sample_items}")
        elif isinstance(value, dict):
            print(f"  Type: Dictionary with {len(value)} keys")
            for k, v in value.items():
                if isinstance(v, (str, int, float, bool)):
                    print(f"    {k}: {v}")
                else:
                    print(f"    {k}: {type(v).__name__}")
        else:
            print(f"  Type: {type(value).__name__}, Value: {value}")


def create_analysis_dataframes(data: Dict[str, Any], split_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Create and analyze pandas DataFrames from JSON data.
    
    Args:
        data: Dictionary containing JSON data
        split_name: Name of the data split (e.g., "training", "validation")
        
    Returns:
        Tuple of (images_df, annotations_df) or (None, None) if data is invalid
    """
    if not data:
        logger.error(f"No data available for {split_name}")
        return None, None
    
    print(f"\n{split_name.upper()} DATAFRAMES ANALYSIS")
    print("-" * 40)
    
    # Create DataFrames
    images_df = pd.DataFrame(data.get('images', []))
    annotations_df = pd.DataFrame(data.get('annotations', []))
    
    # Images DataFrame analysis
    if not images_df.empty:
        print(f"Images DataFrame: {images_df.shape[0]:,} rows × {images_df.shape[1]} columns")
        print(f"  Columns: {list(images_df.columns)}")
        
        # Check for missing values
        missing_counts = images_df.isnull().sum()
        if missing_counts.any():
            missing_fields = {col: count for col, count in missing_counts.items() if count > 0}
            print(f"  Missing values: {missing_fields}")
        else:
            print("  Missing values: None")
    else:
        logger.warning("Images DataFrame is empty")
    
    # Annotations DataFrame analysis
    if not annotations_df.empty:
        print(f"Annotations DataFrame: {annotations_df.shape[0]:,} rows × {annotations_df.shape[1]} columns")
        print(f"  Columns: {list(annotations_df.columns)}")
        
        # Check for missing values
        missing_counts = annotations_df.isnull().sum()
        if missing_counts.any():
            missing_fields = {col: count for col, count in missing_counts.items() if count > 0}
            print(f"  Missing values: {missing_fields}")
        else:
            print("  Missing values: None")
    else:
        logger.warning("Annotations DataFrame is empty")
    
    return images_df, annotations_df


def analyze_annotation_distribution(annotations_df: pd.DataFrame, split_name: str) -> Optional[pd.Series]:
    """
    Analyze the category distribution in annotation data.
    
    Args:
        annotations_df: DataFrame containing annotation data
        split_name: Name of the data split
        
    Returns:
        Series with category counts, or None if data is invalid
    """
    if annotations_df is None or annotations_df.empty:
        logger.error(f"No annotation data for {split_name}")
        return None
    
    print(f"\n{split_name.upper()} ANNOTATION DISTRIBUTION")
    print("-" * 40)
    
    # Category distribution
    category_counts = annotations_df['category_id'].value_counts().sort_index()
    total_annotations = len(annotations_df)
    
    print(f"Total annotations: {total_annotations:,}")
    print("\nCategory breakdown:")
    
    # Category name mapping
    category_names = {1: 'Pedestrian', 2: 'Bicycledriver', 3: 'Motorbikedriver', 4: 'Ignore'}
    
    for cat_id, count in category_counts.items():
        percentage = (count / total_annotations) * 100
        cat_name = category_names.get(cat_id, f'Category_{cat_id}')
        print(f"  {cat_name} (ID {cat_id}): {count:,} ({percentage:.1f}%)")
    
    # Ignore flag distribution
    if 'ignore' in annotations_df.columns:
        ignore_counts = annotations_df['ignore'].value_counts()
        print(f"\nIgnore flag distribution:")
        for ignore_val, count in ignore_counts.items():
            percentage = (count / total_annotations) * 100
            status = "Ignore" if ignore_val == 1 else "Use"
            print(f"  {status} (ignore={ignore_val}): {count:,} ({percentage:.1f}%)")
    
    return category_counts


def validate_data_consistency(train_data: Optional[Dict[str, Any]], 
                            val_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate data consistency between training and validation sets.
    
    Args:
        train_data: Training dataset JSON data
        val_data: Validation dataset JSON data
        
    Returns:
        Dictionary containing validation results
    """
    print(f"\nDATA CONSISTENCY VALIDATION")
    print("-" * 40)
    
    validation_results = {
        'categories_consistent': False,
        'category_id_4_consistent': False,
        'issues_found': []
    }
    
    if not train_data or not val_data:
        validation_results['issues_found'].append("Missing training or validation data")
        return validation_results
    
    # Check category consistency
    train_categories = {cat['id']: cat['name'] for cat in train_data.get('categories', [])}
    val_categories = {cat['id']: cat['name'] for cat in val_data.get('categories', [])}
    
    if train_categories == val_categories:
        print("Categories: Consistent between training and validation sets")
        validation_results['categories_consistent'] = True
    else:
        print("Categories: Inconsistent between splits")
        train_only = set(train_categories.values()) - set(val_categories.values())
        val_only = set(val_categories.values()) - set(train_categories.values())
        if train_only:
            print(f"  Training only: {train_only}")
            validation_results['issues_found'].append(f"Training-only categories: {train_only}")
        if val_only:
            print(f"  Validation only: {val_only}")
            validation_results['issues_found'].append(f"Validation-only categories: {val_only}")
    
    # Check category_id = 4 consistency with ignore flag
    def check_category_4_consistency(data: Dict[str, Any], split_name: str) -> bool:
        """Check if all category_id=4 annotations have ignore=1"""
        annotations = data.get('annotations', [])
        category_4_annotations = [ann for ann in annotations if ann.get('category_id') == 4]
        
        if not category_4_annotations:
            return True
        
        all_ignored = all(ann.get('ignore') == 1 for ann in category_4_annotations)
        print(f"{split_name} category_id=4 consistency: {len(category_4_annotations):,} annotations, all ignored: {all_ignored}")
        return all_ignored
    
    train_consistent = check_category_4_consistency(train_data, "Training")
    val_consistent = check_category_4_consistency(val_data, "Validation")
    
    if train_consistent and val_consistent:
        print("Category ID 4 consistency: All category_id=4 have ignore=1 in both splits")
        validation_results['category_id_4_consistent'] = True
    else:
        validation_results['issues_found'].append("Category ID 4 inconsistency found")
    
    # Summary
    if not validation_results['issues_found']:
        print("\nValidation Summary: All consistency checks passed")
    else:
        print(f"\nValidation Summary: {len(validation_results['issues_found'])} issues found")
        for issue in validation_results['issues_found']:
            print(f"  - {issue}")
    
    return validation_results


def get_dataset_summary(images_df: pd.DataFrame, annotations_df: pd.DataFrame, 
                       split_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the dataset.
    
    Args:
        images_df: DataFrame containing image data
        annotations_df: DataFrame containing annotation data
        split_name: Name of the data split
        
    Returns:
        Dictionary containing dataset summary statistics
    """
    summary = {
        'split_name': split_name,
        'total_images': len(images_df) if images_df is not None else 0,
        'total_annotations': len(annotations_df) if annotations_df is not None else 0,
        'avg_annotations_per_image': 0,
        'unique_categories': 0,
        'category_distribution': {},
        'missing_data_fields': []
    }
    
    if images_df is not None and not images_df.empty:
        # Check for missing data
        missing_fields = images_df.isnull().sum()
        summary['missing_data_fields'] = [col for col, count in missing_fields.items() if count > 0]
    
    if annotations_df is not None and not annotations_df.empty:
        summary['avg_annotations_per_image'] = len(annotations_df) / len(images_df) if len(images_df) > 0 else 0
        summary['unique_categories'] = annotations_df['category_id'].nunique()
        
        # Category distribution
        category_counts = annotations_df['category_id'].value_counts()
        category_names = {1: 'Pedestrian', 2: 'Bicycledriver', 3: 'Motorbikedriver', 4: 'Ignore'}
        summary['category_distribution'] = {
            category_names.get(cat_id, f'Category_{cat_id}'): count 
            for cat_id, count in category_counts.items()
        }
    
    return summary