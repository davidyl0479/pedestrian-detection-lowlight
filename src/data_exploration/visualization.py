"""
Visualization Module

This module provides comprehensive visualization functions for the NightOwls dataset analysis.
It includes bbox analysis visualization and an enhanced sequence visualization function that
combines multiple viewing modes and problem highlighting capabilities.

Functions:
    visualize_images_by_size_and_category: Visualize images filtered by bbox size and category
    visualize_sequence_comprehensive: Enhanced comprehensive sequence visualization
    create_visualization_legends: Utility functions for consistent legends and color schemes
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_images_by_size_and_category(
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    images_dir: Path,
    split_name: str,
    size_filter: str,
    category_filter: Optional[Union[int, str]] = None,
    num_samples: int = 6,
    linewidth: float = 1.0,
) -> None:
    """
    Display sample images with bounding boxes filtered by size and category.

    Args:
        images_df: DataFrame with image information
        annotations_df: DataFrame with annotation information
        images_dir: Path to images directory
        split_name: Name of the split (e.g., "TRAINING", "VALIDATION")
        size_filter: Size category to filter ('0-1K', '1-2K', '2-3K', '3-4K', '4-5K', '5-10K', '10-20K', '20K+')
        category_filter: Category to filter (int ID, str name, or None for any category)
        num_samples: Number of sample images to display
        linewidth: Thickness of bounding box lines
    """
    print(
        f"\n{split_name.upper()} - {size_filter.upper()} BOUNDING BOXES VISUALIZATION"
    )
    if category_filter:
        print(f"Filtered by category: {category_filter}")
    print("-" * 60)

    if images_df is None or annotations_df is None:
        logger.error("Missing data for visualization")
        return

    # Create bbox analysis with size intervals
    bbox_data = []
    for _, row in annotations_df.iterrows():
        if "bbox" in row and row["bbox"] is not None and len(row["bbox"]) >= 4:
            x, y, w, h = row["bbox"][:4]
            if w > 0 and h > 0:  # Only valid bboxes
                bbox_data.append(
                    {
                        "image_id": row.get("image_id"),
                        "annotation_id": row.get("id"),
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "area": w * h,
                        "category_id": row.get("category_id", -1),
                        "ignore": row.get("ignore", 0),
                    }
                )

    if not bbox_data:
        logger.warning("No valid bounding boxes found")
        return

    bbox_df = pd.DataFrame(bbox_data)

    # Add size categories
    bbox_df["size_category"] = pd.cut(
        bbox_df["area"],
        bins=[0, 1000, 2000, 3000, 4000, 5000, 10000, 20000, float("inf")],
        labels=["0-1K", "1-2K", "2-3K", "3-4K", "4-5K", "5-10K", "10-20K", "20K+"],
    )

    # Filter by size category
    size_filtered_bboxes = bbox_df[bbox_df["size_category"] == size_filter]

    if len(size_filtered_bboxes) == 0:
        print(f"No {size_filter} bounding boxes found")
        return

    # Filter by category if specified
    if category_filter is not None:
        # Convert category name to ID if needed
        category_name_to_id = {
            "pedestrian": 1,
            "bicycle": 2,
            "motorbike": 3,
            "ignore": 4,
        }

        if isinstance(category_filter, str):
            category_id = category_name_to_id.get(category_filter.lower())
            if category_id is None:
                logger.error(
                    f"Unknown category: {category_filter}. Use: pedestrian, bicycle, motorbike, or ignore"
                )
                return
            category_name = category_filter.capitalize()
        else:
            category_id = category_filter
            category_names = {
                1: "Pedestrian",
                2: "Bicycle",
                3: "Motorbike",
                4: "Ignore",
            }
            category_name = category_names.get(category_id, f"Category_{category_id}")

        size_filtered_bboxes = size_filtered_bboxes[
            size_filtered_bboxes["category_id"] == category_id
        ]

        if len(size_filtered_bboxes) == 0:
            print(f"No {size_filter} {category_name} bounding boxes found")
            return

        print(
            f"Found {len(size_filtered_bboxes)} {size_filter} {category_name} bounding boxes"
        )
    else:
        print(f"Found {len(size_filtered_bboxes)} {size_filter} bounding boxes")

    # Get unique images that contain the filtered bounding boxes
    target_image_ids = size_filtered_bboxes["image_id"].unique()
    target_images = images_df[images_df["id"].isin(target_image_ids)]

    if len(target_images) == 0:
        print(f"No images found with {size_filter} bounding boxes")
        return

    # Sample images to display
    sample_images = target_images.sample(min(num_samples, len(target_images)))

    # Display size range for reference
    size_areas = size_filtered_bboxes["area"]
    print(
        f"{size_filter} area range: {size_areas.min():.0f} - {size_areas.max():.0f} pixels²"
    )
    print(f"Average area: {size_areas.mean():.0f} pixels²")

    # Create visualization
    num_cols = 3
    num_rows = (len(sample_images) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))

    # Handle different cases for axes array
    if len(sample_images) == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten() if len(sample_images) > 1 else axes

    # Color scheme
    category_colors = {1: "red", 2: "blue", 3: "green", 4: "gray"}
    category_names = {1: "Pedestrian", 2: "Bicycle", 3: "Motorbike", 4: "Ignore"}

    # Size-specific colors for highlighting
    size_highlight_colors = {
        "0-1K": "orange",
        "1-2K": "purple",
        "2-3K": "cyan",
        "3-4K": "yellow",
        "4-5K": "magenta",
        "5-10K": "lime",
        "10-20K": "pink",
        "20K+": "gold",
    }

    for idx, (_, img_info) in enumerate(sample_images.iterrows()):
        ax = axes_flat[idx]

        img_path = images_dir / img_info["file_name"]

        if not img_path.exists():
            logger.warning(f"Image not found: {img_info['file_name']}")
            ax.text(
                0.5,
                0.5,
                f"Image not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        try:
            # Load and display image
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax.imshow(img_rgb)
            ax.set_title(
                f"Image ID: {img_info['id']}\n{img_info['file_name']}", fontsize=10
            )
            ax.axis("off")

            # Get all annotations for this image
            img_annotations = annotations_df[
                annotations_df["image_id"] == img_info["id"]
            ]

            # Get target size bboxes for this image
            img_target_bboxes = size_filtered_bboxes[
                size_filtered_bboxes["image_id"] == img_info["id"]
            ]

            target_bbox_count = 0
            other_bbox_count = 0

            for _, ann in img_annotations.iterrows():
                if "bbox" in ann and ann["bbox"] is not None and len(ann["bbox"]) >= 4:
                    x, y, w, h = ann["bbox"][:4]

                    # Skip invalid bboxes
                    if w <= 0 or h <= 0:
                        continue

                    ann_category_id = ann.get("category_id", 1)
                    ignore_flag = ann.get("ignore", 0)
                    ann_id = ann.get("id")

                    # Check if this annotation is one of our target size/category
                    is_target_bbox = ann_id in img_target_bboxes["annotation_id"].values

                    if is_target_bbox:
                        # Highlight target bboxes with size-specific color
                        color = size_highlight_colors.get(size_filter, "orange")
                        linestyle = "-"
                        current_linewidth = linewidth + 1
                        alpha = 1.0
                        target_bbox_count += 1

                        # Get actual area for this bbox
                        bbox_area = w * h

                        # Draw the target bbox
                        rect = Rectangle(
                            (x, y),
                            w,
                            h,
                            linewidth=current_linewidth,
                            edgecolor=color,
                            facecolor="none",
                            linestyle=linestyle,
                            alpha=alpha,
                        )
                        ax.add_patch(rect)

                        # Add detailed label
                        cat_name = category_names.get(
                            ann_category_id, f"Cat{ann_category_id}"
                        )
                        ignore_text = " (ignore)" if ignore_flag == 1 else ""
                        label = f"{size_filter}: {cat_name}{ignore_text}\nArea: {bbox_area:.0f}px²"

                        ax.text(
                            x,
                            y - 10,
                            label,
                            fontsize=8,
                            color=color,
                            weight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor=color,
                                alpha=0.9,
                                linewidth=1,
                            ),
                        )

                    else:
                        # Draw other bboxes with reduced opacity
                        color = category_colors.get(ann_category_id, "yellow")
                        linestyle = "--" if ignore_flag == 1 else "-"
                        current_linewidth = linewidth
                        alpha = 0.3  # Dim other bboxes

                        rect = Rectangle(
                            (x, y),
                            w,
                            h,
                            linewidth=current_linewidth,
                            edgecolor=color,
                            facecolor="none",
                            linestyle=linestyle,
                            alpha=alpha,
                        )
                        ax.add_patch(rect)

                        other_bbox_count += 1

            # Add image info
            img_height, img_width = img_rgb.shape[:2]
            ax.text(
                0.02,
                0.98,
                f"Size: {img_width}×{img_height}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7),
                verticalalignment="top",
            )

            # Add count info as xlabel
            xlabel_text = (
                f"{size_filter}: {target_bbox_count} | Others: {other_bbox_count}"
            )
            ax.set_xlabel(xlabel_text, fontsize=9, weight="bold")

        except Exception as e:
            logger.error(f"Error loading {img_info['file_name']}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error loading image",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

    # Hide any unused subplots
    for idx in range(len(sample_images), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.show()

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=size_highlight_colors.get(size_filter, "orange"),
            lw=linewidth + 1,
            label=f"{size_filter} (Target)",
        ),
        Line2D(
            [0], [0], color="red", lw=linewidth, alpha=0.3, label="Other Pedestrian"
        ),
        Line2D([0], [0], color="blue", lw=linewidth, alpha=0.3, label="Other Bicycle"),
        Line2D(
            [0], [0], color="green", lw=linewidth, alpha=0.3, label="Other Motorbike"
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=linewidth,
            linestyle="--",
            alpha=0.3,
            label="Other Ignore",
        ),
    ]

    plt.figure(figsize=(12, 1))
    plt.legend(handles=legend_elements, loc="center", ncol=len(legend_elements))
    plt.axis("off")
    plt.title(f"Bounding Box Legend - Highlighting {size_filter} Objects")
    plt.show()


def visualize_sequence_comprehensive(
    sequence_id: str,
    annotations_df: pd.DataFrame,
    images_df: pd.DataFrame,
    images_dir: Path,
    sequences_data: Optional[List[Dict[str, Any]]] = None,
    negative_issues: Optional[List[Dict[str, Any]]] = None,
    split_name: str = "",
    initial_frame: Optional[int] = None,
    final_frame: Optional[int] = None,
    max_frames: int = 16,
    show_coordinates: bool = True,
    highlight_tracking_id: Optional[int] = None,
    highlight_category: Optional[Union[int, str]] = None,
    highlight_problems: bool = True,
    show_other_objects: bool = True,
) -> None:
    """
    Enhanced comprehensive sequence visualization combining multiple viewing modes.

    This function builds upon the original visualize_sequence_comprehensive and incorporates
    functionality from visualize_negative_bbox_images_by_category and visualize_tracking_sequence.

    Args:
        sequence_id: Sequence ID to visualize
        annotations_df: DataFrame with annotations
        images_df: DataFrame with image information
        images_dir: Path to images directory
        sequences_data: Optional list of sequence data for metadata lookup
        negative_issues: Optional list of negative bbox issues for problem highlighting
        split_name: Name of the split (e.g., "TRAINING", "VALIDATION")
        initial_frame: Starting frame number (1-based, None for beginning)
        final_frame: Ending frame number (1-based, None for end)
        max_frames: Maximum frames to show at once
        show_coordinates: Show bbox coordinates for problematic ones
        highlight_tracking_id: Optional tracking ID to focus on
        highlight_category: Optional category to focus on (int ID or str name)
        highlight_problems: Whether to highlight problematic bboxes
        show_other_objects: Show other objects with low opacity
    """

    print(f"\n{split_name.upper()} COMPREHENSIVE SEQUENCE VISUALIZATION")
    print(f"Sequence: {sequence_id}")
    print("-" * 60)

    # Find sequence information
    sequence_info = None
    if sequences_data:
        for seq in sequences_data:
            if seq.get("sequence_id") == sequence_id:
                sequence_info = seq
                break

    if sequence_info is None:
        # Try to construct sequence info from sequence_id
        if sequence_id.startswith("seq_"):
            try:
                seed_tracking_id = int(sequence_id.replace("seq_", ""))
                # Get images for this tracking ID
                seed_images = annotations_df[
                    annotations_df["tracking_id"] == seed_tracking_id
                ]["image_id"].unique()
                if len(seed_images) > 0:
                    # Get all tracking IDs in those images
                    related_tracking_ids = annotations_df[
                        annotations_df["image_id"].isin(seed_images)
                    ]["tracking_id"].unique()
                    # Get all images for those tracking IDs
                    all_images = annotations_df[
                        annotations_df["tracking_id"].isin(related_tracking_ids)
                    ]["image_id"].unique()

                    sequence_info = {
                        "sequence_id": sequence_id,
                        "tracking_ids": list(related_tracking_ids),
                        "sequence_images": list(all_images),
                        "total_frames": len(all_images),
                    }
            except ValueError:
                pass

    if sequence_info is None:
        logger.error(f"Sequence {sequence_id} not found")
        return

    image_ids = sequence_info["sequence_images"]
    tracking_ids = sequence_info["tracking_ids"]

    print(
        f"Sequence contains {len(image_ids)} frames with {len(tracking_ids)} tracking IDs"
    )
    print(f"Tracking IDs: {tracking_ids}")

    # Get image information and sort by filename for temporal order
    sequence_images = images_df[images_df["id"].isin(image_ids)].copy()
    sequence_images = sequence_images.sort_values("file_name")

    # Apply frame range filtering
    total_frames = len(sequence_images)
    if initial_frame is not None:
        initial_frame = max(1, min(initial_frame, total_frames))
    else:
        initial_frame = 1

    if final_frame is not None:
        final_frame = max(initial_frame, min(final_frame, total_frames))
    else:
        final_frame = total_frames

    # Select frame range
    frame_slice = sequence_images.iloc[initial_frame - 1 : final_frame]

    print(f"Showing frames {initial_frame}-{final_frame} of {total_frames}")

    # Further limit if too many frames
    if len(frame_slice) > max_frames:
        print(f"Limiting to first {max_frames} frames of range")
        frame_slice = frame_slice.head(max_frames)

    # Convert category filter if needed
    category_id_filter = None
    if highlight_category is not None:
        category_name_to_id = {
            "pedestrian": 1,
            "bicycle": 2,
            "motorbike": 3,
            "ignore": 4,
        }

        if isinstance(highlight_category, str):
            category_id_filter = category_name_to_id.get(highlight_category.lower())
            if category_id_filter is None:
                logger.warning(f"Unknown category: {highlight_category}")
        else:
            category_id_filter = highlight_category

    # Helper function to check bbox problems
    def check_bbox_problems(image_id, img_width, img_height):
        """Check if image has problematic bboxes and which tracking IDs are affected."""
        problematic_tracking_ids = set()

        # Check negative issues
        if negative_issues:
            for issue in negative_issues:
                if issue.get("image_id") == image_id:
                    tracking_id = issue.get("tracking_id")
                    if tracking_id:
                        problematic_tracking_ids.add(tracking_id)

        # Check annotations for problems
        img_annotations = annotations_df[annotations_df["image_id"] == image_id]
        for _, ann in img_annotations.iterrows():
            if "bbox" in ann and ann["bbox"] is not None and len(ann["bbox"]) >= 4:
                x, y, w, h = ann["bbox"][:4]
                tracking_id = ann.get("tracking_id")

                # Check for problematic coordinates
                has_negative = x < 0 or y < 0 or w < 0 or h < 0

                # Check if bbox goes outside image bounds
                completely_outside = (
                    (x + w <= 0)
                    or (x >= img_width)
                    or (y + h <= 0)
                    or (y >= img_height)
                )

                partially_outside = (
                    (x < 0 and x + w > 0)
                    or (y < 0 and y + h > 0)
                    or (x < img_width and x + w > img_width)
                    or (y < img_height and y + h > img_height)
                )

                if has_negative or completely_outside or partially_outside:
                    if tracking_id:
                        problematic_tracking_ids.add(tracking_id)

        return problematic_tracking_ids

    # Set up visualization
    n_frames = len(frame_slice)
    cols = min(4, n_frames)
    rows = (n_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(
        f"{split_name} - Sequence {sequence_id} (Frames {initial_frame}-{final_frame})",
        fontsize=14,
    )

    # Adjust spacing
    plt.subplots_adjust(top=1.5, bottom=0.1, hspace=0.4, wspace=0.3)

    # Handle axes array
    if n_frames == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten() if n_frames > 1 else axes

    # Color scheme
    category_colors = {1: "red", 2: "blue", 3: "green", 4: "gray"}
    category_names = {1: "Pedestrian", 2: "Bicycle", 3: "Motorbike", 4: "Ignore"}

    for idx, (_, img_info) in enumerate(frame_slice.iterrows()):
        ax = axes_flat[idx]
        img_path = images_dir / img_info["file_name"]

        # Frame number for display
        frame_number = initial_frame + idx

        if not img_path.exists():
            ax.text(
                0.5,
                0.5,
                f"Image not found:\n{img_info['file_name']}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Frame {frame_number} - Error", fontsize=10)
            ax.axis("off")
            continue

        try:
            # Load and display image
            img = cv2.imread(str(img_path))
            if img is None:
                ax.text(
                    0.5,
                    0.5,
                    f"Cannot load:\n{img_info['file_name']}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width = img_rgb.shape[:2]
            ax.imshow(img_rgb)

            # Check for problematic tracking IDs in this frame
            problematic_tracking_ids = check_bbox_problems(
                img_info["id"], img_width, img_height
            )

            # Get all annotations for this image
            img_annotations = annotations_df[
                annotations_df["image_id"] == img_info["id"]
            ]

            # Collect problematic bbox info for bottom display
            problematic_info = []

            for _, ann in img_annotations.iterrows():
                if "bbox" in ann and ann["bbox"] is not None and len(ann["bbox"]) >= 4:
                    x, y, w, h = ann["bbox"][:4]
                    category_id = ann.get("category_id", 1)
                    tracking_id = ann.get("tracking_id")
                    ignore_flag = ann.get("ignore", 0)

                    # Determine if this bbox is problematic
                    has_negative = x < 0 or y < 0 or w < 0 or h < 0
                    completely_outside = (
                        (x + w <= 0)
                        or (x >= img_width)
                        or (y + h <= 0)
                        or (y >= img_height)
                    )
                    partially_outside = (
                        (x < 0 and x + w > 0)
                        or (y < 0 and y + h > 0)
                        or (x < img_width and x + w > img_width)
                        or (y < img_height and y + h > img_height)
                    )

                    is_problematic = (
                        has_negative or completely_outside or partially_outside
                    )

                    # Determine styling based on filters and problems
                    should_highlight = False
                    color = category_colors.get(category_id, "yellow")
                    linewidth = 1
                    alpha = 1.0
                    linestyle = "--" if ignore_flag == 1 else "-"

                    # Apply highlighting logic
                    if highlight_problems and is_problematic:
                        # Problematic bboxes - Orange color, highest priority
                        color = "orange"
                        linewidth = 3
                        alpha = 1.0
                        should_highlight = True

                        # Collect info for bottom display
                        if show_coordinates:
                            category_name = category_names.get(
                                category_id, f"Cat{category_id}"
                            )
                            if completely_outside:
                                problem_type = "completely_outside"
                            elif partially_outside:
                                problem_type = "partially_outside"
                            else:
                                problem_type = "negative_coords"

                            info_text = f"ID:{tracking_id} {category_name} {problem_type} [{x:.1f},{y:.1f},{w:.1f},{h:.1f}]"
                            problematic_info.append(info_text)

                    elif (
                        highlight_tracking_id is not None
                        and tracking_id == highlight_tracking_id
                    ):
                        # Highlighted tracking ID
                        color = "lime"
                        linewidth = 2
                        alpha = 1.0
                        should_highlight = True

                    elif (
                        category_id_filter is not None
                        and category_id == category_id_filter
                    ):
                        # Highlighted category
                        color = "cyan"
                        linewidth = 2
                        alpha = 1.0
                        should_highlight = True

                    elif tracking_id in tracking_ids:
                        # Normal bboxes from this sequence
                        if show_other_objects:
                            alpha = 0.7
                        else:
                            continue  # Skip if not showing other objects
                    else:
                        # Other tracking IDs (dim)
                        if show_other_objects:
                            alpha = 0.3
                        else:
                            continue  # Skip if not showing other objects

                    # Draw bounding box
                    rect = Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=linewidth,
                        edgecolor=color,
                        facecolor="none",
                        linestyle=linestyle,
                        alpha=alpha,
                    )
                    ax.add_patch(rect)

            # Display problematic info below the image
            if problematic_info and show_coordinates:
                y_offset = -0.15
                for info_text in problematic_info[:3]:  # Limit to 3 entries
                    # Extract category from info text to determine color
                    category_color = "black"  # default
                    if "Pedestrian" in info_text:
                        category_color = "red"
                    elif "Bicycle" in info_text:
                        category_color = "blue"
                    elif "Motorbike" in info_text:
                        category_color = "green"
                    elif "Ignore" in info_text:
                        category_color = "gray"

                    ax.text(
                        0.5,
                        y_offset,
                        info_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        ha="center",
                        va="top",
                        color=category_color,
                        weight="bold",
                    )
                    y_offset -= 0.05

            # Set title with frame number and problem indicator
            problem_indicator = " ⚠" if problematic_tracking_ids else ""
            title = f"Frame {frame_number}{problem_indicator}\n{img_info['file_name']}"
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error loading frame {frame_number}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Frame {frame_number} - Error", fontsize=10)
            ax.axis("off")

    # Hide unused subplots
    for idx in range(n_frames, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.show()

    # Create comprehensive legend
    create_visualization_legends(
        highlight_problems=highlight_problems,
        highlight_tracking_id=highlight_tracking_id,
        highlight_category=highlight_category,
        show_other_objects=show_other_objects,
    )

    # Print summary
    total_problematic_frames = 0
    for _, img_info in frame_slice.iterrows():
        img_height = img_info.get("height", 0)
        img_width = img_info.get("width", 0)
        if img_width > 0 and img_height > 0:
            problematic_ids = check_bbox_problems(img_info["id"], img_width, img_height)
            if problematic_ids:
                total_problematic_frames += 1

    print(f"\nVisualization summary:")
    print(f"  Sequence: {sequence_id}")
    print(f"  Frames shown: {initial_frame}-{final_frame} of {total_frames}")
    print(f"  Problematic frames: {total_problematic_frames}")
    if highlight_tracking_id:
        print(f"  Highlighted tracking ID: {highlight_tracking_id}")
    if highlight_category:
        print(f"  Highlighted category: {highlight_category}")


def create_visualization_legends(
    highlight_problems: bool = True,
    highlight_tracking_id: Optional[int] = None,
    highlight_category: Optional[Union[int, str]] = None,
    show_other_objects: bool = True,
) -> None:
    """
    Create comprehensive visualization legends based on current highlighting modes.

    Args:
        highlight_problems: Whether problems are being highlighted
        highlight_tracking_id: Optional highlighted tracking ID
        highlight_category: Optional highlighted category
        show_other_objects: Whether other objects are shown
    """
    legend_elements = []

    if highlight_problems:
        legend_elements.append(
            Line2D([0], [0], color="orange", lw=3, label="⚠ Problematic BBox")
        )

    if highlight_tracking_id is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="lime",
                lw=2,
                label=f"Tracking ID: {highlight_tracking_id}",
            )
        )

    if highlight_category is not None:
        legend_elements.append(
            Line2D(
                [0], [0], color="cyan", lw=2, label=f"Category: {highlight_category}"
            )
        )

    # Standard category colors
    legend_elements.extend(
        [
            Line2D([0], [0], color="red", lw=1, alpha=0.7, label="Pedestrian"),
            Line2D([0], [0], color="blue", lw=1, alpha=0.7, label="Bicycle"),
            Line2D([0], [0], color="green", lw=1, alpha=0.7, label="Motorbike"),
            Line2D(
                [0], [0], color="gray", lw=1, linestyle="--", alpha=0.7, label="Ignore"
            ),
        ]
    )

    if show_other_objects:
        legend_elements.append(
            Line2D([0], [0], color="gray", lw=1, alpha=0.3, label="Other Objects")
        )

    plt.figure(figsize=(12, 1.5))
    plt.legend(handles=legend_elements, loc="center", ncol=min(6, len(legend_elements)))
    plt.axis("off")
    plt.title("Comprehensive Visualization Legend")
    plt.show()
