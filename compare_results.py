import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely.geometry import Polygon, MultiPolygon
from shapely.measurement import hausdorff_distance
import pandas as pd


def intersection_over_union(poly1, poly2):
    """Computes IoU between two polygons."""
    intersection = poly1.intersection(poly2)

    # If intersection is None (no valid intersection), return 0
    if intersection is None or intersection.is_empty:
        print("No intersection found")
        return 0.0

    union = poly1.union(poly2).area
    return intersection.area / union if union > 0 else 0


def get_shapefiles_from_folder(folder_path):
    """Returns a list of all .shp files in the given folder, ignoring other file types."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".shp")]


def compute_scores(reference_path, folder_path):
    """
    Computes IoU scores and Hausdorff distances for all shapefiles in a folder against the reference,
    matching polygons by perceel_id and skipping shapefiles where required columns are missing.

    Parameters:
        reference_path (str): Path to the reference shapefile.
        folder_path (str): Path to the folder containing test shapefiles.

    Returns:
        dict: A dictionary with filenames as keys, and a list of tuples (IoU, Hausdorff distances) as values.
    """
    reference_gdf = gpd.read_file(reference_path)
    test_shapefiles = get_shapefiles_from_folder(folder_path)

    scores = {}

    for test_path in test_shapefiles:
        test_gdf = gpd.read_file(test_path)

        # Merge reference and test GeoDataFrames based on perceel_id, specifying suffixes
        merged_gdf = reference_gdf.merge(test_gdf, on="perceel_id", how="inner", suffixes=('_ref', '_test'))

        for idx, row in merged_gdf.iterrows():
            reference_geom = row['geometry_ref']  # reference geometry (with suffix '_ref')
            test_geom = row['geometry_test']  # test geometry (with suffix '_test')

            iou_score = intersection_over_union(reference_geom, test_geom)

            # Check if 'score_haus_test' exists in the test GeoDataFrame and use it if available
            if 'score_haus_test' in row and pd.notna(row['score_haus_test']):
                hausdorff_dist = row['score_haus_test']

            else:
                hausdorff_dist = hausdorff_distance(reference_geom, test_geom)

            filename = os.path.basename(test_path)
            if filename not in scores:
                scores[filename] = []
            scores[filename].append((iou_score, hausdorff_dist))

    return scores



def plot_comparison(reference_path, folder_path):
    """
    Creates two separate bar charts: one for IoU scores and one for Hausdorff distances.
    The Hausdorff distance plot is zoomed in for better comparison.

    Parameters:
        reference_path (str): Path to the reference shapefile.
        folder_path (str): Path to the folder containing test shapefiles.
    """
    scores = compute_scores(reference_path, folder_path)

    if not scores:
        print("No shapefiles found in the folder or no valid scores.")
        return

    # Prepare data for the plots: aggregate IoU and Hausdorff distances for each shapefile
    filenames = []
    iou_scores = []
    hausdorff_distances = []

    for filename, score_list in scores.items():
        filenames.append(filename)
        # Aggregate the IoU and Hausdorff distance scores
        avg_iou = np.mean([score[0] for score in score_list])  # Average IoU
        avg_hausdorff = np.mean([score[1] for score in score_list])  # Average Hausdorff distance
        iou_scores.append(avg_iou)
        hausdorff_distances.append(avg_hausdorff)

    # Sort the scores in descending order based on IoU score
    sorted_indices = np.argsort(iou_scores)[::-1]  # Sort in descending order
    sorted_filenames = [filenames[i] for i in sorted_indices]
    sorted_iou_scores = [iou_scores[i] for i in sorted_indices]
    sorted_indices_haus = np.argsort(hausdorff_distances)[::1]  # Sort in descending order
    sorted_filenames_haus = [filenames[i] for i in sorted_indices_haus]
    sorted_hausdorff_distances = [hausdorff_distances[i] for i in sorted_indices_haus]

    # Plot IoU Scores
    fig, ax = plt.subplots(figsize=(12, 8))
    indices = np.arange(len(sorted_filenames))
    ax.bar(indices, sorted_iou_scores, color='tab:blue', alpha=0.7)

    for i, score in enumerate(sorted_iou_scores):
        ax.text(i, score + 0.02, f"{score:.2f}", ha="center", fontsize=6)

    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU Score")
    ax.set_title("IoU Score Comparison")
    ax.set_xticks(indices)
    ax.set_xticklabels(sorted_filenames, rotation=30, ha="right", fontsize=8)
    plt.subplots_adjust(bottom=0.4)

    # Plot Hausdorff Distances
    fig, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(indices, sorted_hausdorff_distances, color='tab:orange', alpha=0.7)

    for i, score in enumerate(sorted_hausdorff_distances):
        ax2.text(i, score + 0.02, f"{score:.2f}", ha="center", fontsize=6)

    # Zoom in by setting the y-axis range to a smaller range (e.g., from min to max with some margin)
    y_min = min(sorted_hausdorff_distances) - 0.1
    y_max = max(sorted_hausdorff_distances) + 0.1
    ax2.set_ylim(y_min, y_max)

    ax2.set_ylabel("Hausdorff Distance")
    ax2.set_title("Hausdorff Distance Comparison")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(sorted_filenames_haus, rotation=30, ha="right", fontsize=8)

    # Show the plots
    plt.tight_layout()
    plt.show()




reference_shapefile = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\bgt_geometry\bgt_geometry2.shp"
test_folder = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\werkmap"

plot_comparison(reference_shapefile, test_folder)