"""
Data Exploration Module for NightOwls Dataset

This module provides comprehensive tools for exploring and analyzing the NightOwls 
pedestrian detection dataset, including JSON structure analysis, bounding box quality 
assessment, sequence discovery, and advanced visualization capabilities.

Modules:
    json_analysis: Core JSON structure analysis and DataFrame creation
    sequence_discovery: Algorithms for discovering video sequences
    bbox_analysis: Bounding box analysis and quality assessment
    orphan_analysis: Analysis of images without annotations
    comprehensive_export: Unified data export and CSV generation
    visualization: Advanced visualization functions
"""

from .json_analysis import (
    load_json_safe,
    explore_json_structure,
    create_analysis_dataframes,
    analyze_annotation_distribution,
    validate_data_consistency,
    get_dataset_summary
)

from .sequence_discovery import (
    discover_sequences_from_tracking_id,
    discover_all_sequences_comprehensive,
    analyze_discovered_sequences_simple
)

from .bbox_analysis import (
    analyze_bounding_boxes,
    investigate_negative_bbox_values,
    analyze_bbox_coordinates_distribution
)

from .orphan_analysis import (
    analyze_orphan_images_comprehensive,
    get_orphan_summary
)

from .comprehensive_export import (
    create_comprehensive_sequence_dataframe,
    export_comprehensive_analysis
)

__version__ = "1.0.0"
__author__ = "NightOwls Analysis Team"

__all__ = [
    # JSON Analysis
    'load_json_safe',
    'explore_json_structure', 
    'create_analysis_dataframes',
    'analyze_annotation_distribution',
    'validate_data_consistency',
    'get_dataset_summary',
    
    # Sequence Discovery
    'discover_sequences_from_tracking_id',
    'discover_all_sequences_comprehensive',
    'analyze_discovered_sequences_simple',
    
    # Bounding Box Analysis
    'analyze_bounding_boxes',
    'investigate_negative_bbox_values',
    'analyze_bbox_coordinates_distribution',
    
    # Orphan Images Analysis
    'analyze_orphan_images_comprehensive',
    'get_orphan_summary',
    
    # Comprehensive Export
    'create_comprehensive_sequence_dataframe',
    'export_comprehensive_analysis',
]