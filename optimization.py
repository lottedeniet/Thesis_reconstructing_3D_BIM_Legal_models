import os
import json
import numpy as np
import requests
import geopandas as gp
import pandas as pd
from matplotlib import pyplot as plt
from shapely.affinity import translate, scale, rotate
from shapely.geometry import shape
from shapely import geometry as geom
from shapely.geometry import Polygon, MultiPolygon, Point
import re
import math
import time
from shapely.geometry.linestring import LineString
from shapely.measurement import hausdorff_distance, frechet_distance
from shapely.ops import nearest_points
import itertools
from joblib import Parallel, delayed

kad_path = r'C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\original_data\aanvullende_data\Hilversum\Percelen_aktes_Hilversum.shp'
json_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\original_data\gevectoriseerde_set\hilversum_set\observations\snapshots\latest"
out_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\werkmap\output"
api_bgt = 'https://api.pdok.nl/lv/bgt/ogc/v1/collections/pand/items'
files = os.listdir(json_path)


if not os.path.exists(out_path):
    os.mkdir(out_path)

# existing functions
def calcScale(bbxobj, pix):
    """
    Original method to calculate scale from bounding box
    :param bbxobj:
    :param pix:
    :return:
    """
    # get the length of the xbbox and ybbox of the kad perceel
    xlen = float((bbxobj.maxx - bbxobj.minx).iloc[0])
    ylen = float((bbxobj.maxy - bbxobj.miny).iloc[0])
    # if the building is drawn horizontally
    if xlen < ylen:
        # why not the other way around? if the x is smaller, the page is horizontal so the pixel approx is for the x right?
        return ylen / float(pix)
    else:
        return xlen / float(pix)

def egetGeometry(plist, pdict):
    gl = []
    for p in plist:
        gl.append(pdict[p])
    return geom.Polygon(gl)

def addValue(cat, clist, room):
    try:
        clist.append(data['text'][data['rooms'][room][cat]]['value'])
    except:
        clist.append('')


# new functions

def rotate_geom_azimuth(group):
    geometry_columns = [col for col in group.columns if 'geom_akte_bg' in col]
    ref_columns = [col for col in group.columns if 'bgt_outline' in col]

    for index, row in group.iterrows():
        for col in ref_columns:
            geometry_ref = row[col]
            area = geometry_ref.minimum_rotated_rectangle
            line = area.boundary

            coords = [c for c in line.coords]
            segments = [LineString([a, b]) for a, b in zip(coords, coords[1:])]
            longest_segment = max(segments, key=lambda x: x.length)
            p1, p2 = [c for c in longest_segment.coords]
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) + 90
            print(angle)
            for col in geometry_columns:
                geometry_pol = row[col]
                group.at[index, col] = rotate(geometry_pol, angle, origin='centroid')

    return group


def translate_polygon(geometries, translation_vector):
    if isinstance(translation_vector[0], pd.Series):
        dx, dy = translation_vector[0].iloc[0], translation_vector[1].iloc[0]  # Extract scalars
    else:
        dx, dy = translation_vector
    return geometries.apply(lambda geom: translate(geom, xoff=dx, yoff=dy))


def goodness_of_fit(polygon, reference):
    if isinstance(polygon, (Polygon, MultiPolygon)) and isinstance(reference, (Polygon, MultiPolygon)):
        if polygon.is_empty or reference.is_empty:
            return 0
        else:
            c = polygon.intersection(reference).area
            a = polygon.area
            b = reference.area
            g_o_f = (c / b) * (c / a)
            return g_o_f
    else:
        return 0


def intersection_over_union(polygon, reference):
    if isinstance(polygon, (Polygon, MultiPolygon)) and isinstance(reference, (Polygon, MultiPolygon)):
        if polygon.is_empty or reference.is_empty:
            return 0
        else:
            intersection = polygon.intersection(reference).area
            union = polygon.area + reference.area - intersection
            return intersection/union
    else:
        return 0

def calc_hausdorff(polygon, reference):
    """"
    Greatest distance between any point in A and the closest point in B
    """
    if isinstance(polygon, (Polygon, MultiPolygon)) and isinstance(reference, (Polygon, MultiPolygon)):
        return hausdorff_distance(polygon, reference)
    return float('inf')




def extract_boundary_points(geometry):
    """
    Extracts boundary points from a Polygon or MultiPolygon.
    """
    if isinstance(geometry, Polygon):
        return np.array(geometry.exterior.coords)
    elif isinstance(geometry, MultiPolygon):
        return np.vstack([np.array(poly.exterior.coords) for poly in geometry.geoms])
    return np.array([])


def averaged_hausdorff_distance(polygon, reference):
    """
    Computes the Averaged Hausdorff Distance (AHD) between two geometries.

    Parameters:
    - polygon: Shapely Polygon or MultiPolygon (aligned geometry)
    - reference: Shapely Polygon or MultiPolygon (reference geometry)

    Returns:
    - Averaged Hausdorff Distance (float)
    """
    if not isinstance(polygon, (Polygon, MultiPolygon)) or not isinstance(reference, (Polygon, MultiPolygon)):
        return float('inf')

    # Extract boundary points
    polygon_coords = extract_boundary_points(polygon)
    reference_coords = extract_boundary_points(reference)

    if len(polygon_coords) == 0 or len(reference_coords) == 0:
        return float('inf')

    # Compute forward distances (polygon → reference)
    forward_distances = [reference.distance(Point(p)) for p in polygon_coords]

    # Compute backward distances (reference → polygon)
    backward_distances = [polygon.distance(Point(r)) for r in reference_coords]

    # Compute the averaged Hausdorff distance
    avg_hausdorff = (np.mean(forward_distances) + np.mean(backward_distances)) / 2

    return avg_hausdorff


def calc_frechet(polygon, reference):
    """"
    Greatest distance between any point in A and the closest point in B
    """
    if isinstance(polygon, (Polygon)) and isinstance(reference, (Polygon)):
        return frechet_distance(polygon, reference)
    return float('inf')

import numpy as np
from shapely.affinity import rotate, scale, translate


def line_slope(line):
    """Returns the slope of a line (None if vertical)."""
    (x1, y1), (x2, y2) = line
    return (y2 - y1) / (x2 - x1) if x2 != x1 else None  # Avoid division by zero


def angle_between_lines(line1, line2):
    """Computes the angle between two lines."""
    vec1 = np.array(line1[1]) - np.array(line1[0])
    vec2 = np.array(line2[1]) - np.array(line2[0])
    angle = np.degrees(np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0]))
    return abs(angle)


def get_polygon_edges(polygon):
    """Extracts edges (as line segments) from a polygon."""
    edges = []

    if isinstance(polygon, MultiPolygon):
        # Process each polygon separately
        for poly in polygon.geoms:
            edges.extend(get_polygon_edges(poly))  # Recursively get edges
    elif isinstance(polygon, Polygon):
        coords = list(polygon.exterior.coords)  # Extract boundary coordinates
        edges = [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]  # Create edge pairs

    return edges


def compute_edge_angle(edge):
    """Returns the angle of an edge in degrees."""
    vec = np.array(edge[1]) - np.array(edge[0])
    return np.degrees(np.arctan2(vec[1], vec[0]))


def compute_neighbor_angle(edge, neighbor):
    """Computes the angle between an edge and its neighbor."""
    vec1 = np.array(edge[1]) - np.array(edge[0])
    vec2 = np.array(neighbor[1]) - np.array(neighbor[0])

    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if norm_product == 0:
        return 0  # Avoid division by zero

    angle = np.degrees(np.arccos(np.clip(dot_product / norm_product, -1, 1)))
    return angle





def find_best_matching_edges(aligned_edges, reference_edges,

                             length_threshold=0.05,
                             angle_continuity_threshold=1):
    """
    Finds multiple matching edge pairs between two polygons based on:
    - Slope (edge direction)
    - Length similarity
    - Angle continuity with neighboring edges

    Returns:
    - List of (aligned_edge, reference_edge, score) tuples.
    """
    matches = []

    for i, a_edge in enumerate(aligned_edges):
        angle_a = compute_edge_angle(a_edge)
        a_length = np.linalg.norm(np.array(a_edge[1]) - np.array(a_edge[0]))

        # Find neighboring edges (next edge in the polygon)
        a_neighbor = aligned_edges[(i + 1) % len(aligned_edges)]
        angle_a_neighbor = compute_neighbor_angle(a_edge, a_neighbor)

        for j, r_edge in enumerate(reference_edges):
            angle_r = compute_edge_angle(r_edge)
            r_length = np.linalg.norm(np.array(r_edge[1]) - np.array(r_edge[0]))

            # **1. Check Slope Difference**
            # slope_diff = abs(angle_r - angle_a) % 180
            # if slope_diff > slope_threshold:
            #     continue

            # **2. Check Length Similarity**
            length_diff = abs(a_length - r_length) / max(a_length, r_length)
            if length_diff > length_threshold:
                continue

            # # **3. Check Neighbor Angle Similarity**
            # r_neighbor = reference_edges[(j + 1) % len(reference_edges)]
            # angle_r_neighbor = compute_neighbor_angle(r_edge, r_neighbor)
            #
            # angle_diff = abs(angle_r_neighbor - angle_a_neighbor)
            # if angle_diff > angle_continuity_threshold:
            #     continue

            # **Compute Matching Score (lower is better)**
            total_diff = (length_diff * 10)

            matches.append((a_edge, r_edge, total_diff))

    # Sort matches by best score
    matches.sort(key=lambda x: x[2])

    return matches[:1]  # Return top 3 matches


# import shapely.affinity as affinity
# import shapely.geometry as sg
#
#
#
#
#
# def compute_turning_function(edges):
#     angles = []
#     total_length = 0
#     lengths = []
#
#     for i in range(len(edges)):
#         edge = edges[i]
#         next_edge = edges[(i + 1) % len(edges)]
#
#         vec1 = np.array(edge[1]) - np.array(edge[0])
#         vec2 = np.array(next_edge[1]) - np.array(next_edge[0])
#
#         angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
#         angle = np.degrees(angle) % 360
#         if angle > 180:
#             angle -= 360  # Normalize to [-180, 180]
#
#         length = np.linalg.norm(vec1)
#         total_length += length
#         angles.append(angle)
#         lengths.append(total_length)
#
#     return np.array(lengths) / total_length, np.cumsum(angles)
#
#
# def resample_turning_function(lengths, angles, target_length):
#     target_x = np.linspace(0, 1, target_length)
#     resampled_angles = np.interp(target_x, lengths, angles)
#     return resampled_angles
#
#
# def edge_match_transform(pand, reference_column, aligned_column, bgt_outline):
#     transformed_geometries = []
#
#     for idx, row in pand.iterrows():
#         aligned_geom = row[aligned_column]
#         reference_geom = row[bgt_outline]
#
#         if not isinstance(aligned_geom, Polygon) or not isinstance(reference_geom, Polygon):
#             transformed_geometries.append(None)
#             continue
#
#         aligned_edges = list(zip(aligned_geom.exterior.coords[:-1], aligned_geom.exterior.coords[1:]))
#         reference_edges = list(zip(reference_geom.exterior.coords[:-1], reference_geom.exterior.coords[1:]))
#
#         ref_lengths, ref_turning = compute_turning_function(reference_edges)
#         aligned_lengths, aligned_turning = compute_turning_function(aligned_edges)
#
#         target_length = max(len(ref_turning), len(aligned_turning))
#         ref_turning_resampled = resample_turning_function(ref_lengths, ref_turning, target_length)
#         aligned_turning_resampled = resample_turning_function(aligned_lengths, aligned_turning, target_length)
#
#         best_score = float('inf')
#         best_transform = None
#
#         for angle in np.arange(0, 360, 1):  # Try rotating in 1-degree steps
#             rotated_geom = rotate(aligned_geom, angle, origin='centroid')
#             rotated_edges = list(zip(rotated_geom.exterior.coords[:-1], rotated_geom.exterior.coords[1:]))
#             _, rotated_turning = compute_turning_function(rotated_edges)
#             rotated_turning_resampled = resample_turning_function(aligned_lengths, rotated_turning, target_length)
#
#             score = np.sum((rotated_turning_resampled - ref_turning_resampled) ** 2)
#
#             if score < best_score:
#                 best_score = score
#                 best_transform = rotated_geom
#
#         transformed_geometries.append(best_transform)
#
#     pand["optimized_geometry"] = transformed_geometries
#     return pand
#
#
# def match_turning_function(aligned_geom, aligned_turning, ref_turning):
#     """Finds the best transformation to match the turning function."""
#     best_score = float("inf")
#     best_transform = None
#
#     for angle in np.linspace(-180, 180, 360):  # Try rotations in 1-degree increments
#         rotated = affinity.rotate(aligned_geom, angle, origin='centroid')
#         rotated_turning = compute_turning_function(rotated)
#         score = np.sum((rotated_turning - ref_turning) ** 2)
#
#         if score < best_score:
#             best_score = score
#             best_transform = {'rotation': angle}
#
#     return best_transform
#
#
# def apply_transformation(geom, transform):
#     """Applies the best transformation to the geometry."""
#     return affinity.rotate(geom, transform['rotation'], origin='centroid')


def edge_match_transform(pand, reference_column, aligned_column, bgt_outline):
    """
    Performs edge matching on each polygon in `pand`, aligning `aligned_column` to `reference_column`.

    Parameters:
    - pand (GeoDataFrame): Data containing polygons to align.
    - reference_column (str): Column name for reference polygons.
    - aligned_column (str): Column name for polygons to be aligned.
    - apply_scaling (bool): Apply scaling (default=True).
    - apply_rotation (bool): Apply rotation (default=True).
    - apply_translation (bool): Apply translation (default=True).

    Returns:
    - Updated GeoDataFrame with `optimized_geometry` column storing transformed geometries.
    """
    optimized_geometries = []

    for idx, row in pand.iterrows():
        aligned_poly = row[aligned_column]
        reference_poly = row[bgt_outline]

        # Skip if geometries are missing
        if aligned_poly is None or reference_poly is None:
            optimized_geometries.append(None)
            continue

        aligned_edges = get_polygon_edges(aligned_poly)
        reference_edges = get_polygon_edges(reference_poly)

        best_matches = find_best_matching_edges(aligned_edges, reference_edges)

        if not best_matches:
            # Use grid search if no matches found
            geometries = grid_search(pand, reference_column, aligned_column, bgt_outline, alpha=0.2, buffer=1,
                                     angle_step=1, scale_step=0.05, scale_range=(0.8, 1.2),
                                     translation_step=0.2, max_translation=1)
            optimized_geometries.append(geometries[0])
        else:
            translations = []
            rotations = []
            scales = []

            for best_aligned_edge, best_reference_edge, _ in best_matches:
                # Compute translation
                translation_x = best_reference_edge[0][0] - best_aligned_edge[0][0]
                translation_y = best_reference_edge[0][1] - best_aligned_edge[0][1]
                translations.append((translation_x, translation_y))

                # Compute rotation
                a_vec = np.array(best_aligned_edge[1]) - np.array(best_aligned_edge[0])
                r_vec = np.array(best_reference_edge[1]) - np.array(best_reference_edge[0])
                angle_a = np.degrees(np.arctan2(a_vec[1], a_vec[0]))
                angle_r = np.degrees(np.arctan2(r_vec[1], r_vec[0]))
                rotations.append(angle_r - angle_a)

                # Compute scaling
                aligned_length = np.linalg.norm(a_vec)
                reference_length = np.linalg.norm(r_vec)
                scales.append(reference_length / aligned_length if aligned_length > 0 else 1)

            # **Apply Averaged Transformation**
            avg_translation = np.mean(translations, axis=0)
            avg_rotation = np.mean(rotations)
            avg_scale = np.mean(scales)

            # Apply transformations
            aligned_poly = translate(aligned_poly, xoff=avg_translation[0], yoff=avg_translation[1])
            aligned_poly = rotate(aligned_poly,  avg_rotation,
                                  origin=best_matches[0][1][0])  # Rotate around first ref point
            aligned_poly = scale(aligned_poly, xfact=avg_scale, yfact=avg_scale, origin=best_matches[0][1][0])

            optimized_geometries.append(aligned_poly)

    # Store transformed geometries in the GeoDataFrame
    print(optimized_geometries)
    pand['optimized_geometry'] = optimized_geometries
    return pand



def grid_search(pand, reference_column, aligned_column, bgt_outline, alpha,
                buffer=0.5,
                angle_step=1,
                scale_step=0.05, scale_range=(0.8, 1.2),
                translation_step=0.2, max_translation=2.0):
    """
    Performs a grid search over rotation angles, scales, and translations to optimize Goodness of Fit (GoF).
    Each transformation (rotation, scaling, translation) can be toggled on or off.

    Parameters:
    - pand: DataFrame
    - reference_column: Column name containing the reference geometries
    - aligned_column: Column name containing the geometries to transform
    - bgt_outline: Column name containing the BGT outline geometries
    - alpha: Ratio of Hausdorff vs GoF metrics
    - buffer: Buffer size around the BGT geometry (default: 0.5 meters)
    - apply_rotation: Whether to apply rotation (default: True)
    - angle_step: Step size for rotation angles (default: 0.3°)
    - apply_scaling: Whether to apply scaling (default: True)
    - scale_step: Step size for scaling factor (default: 0.05)
    - scale_range: Range of scaling factors (default: (0.8, 1.2))
    - apply_translation: Whether to apply translation (default: True)
    - translation_step: Step size for translation (default: 0.2 meters)
    - max_translation: Maximum translation distance in meters (default: 2.0 meters)

    Returns:
    - DataFrame with the best transformation applied
    """

    best_score = -np.inf
    best_geometries = None
    best_scale = 1.0
    best_angle = 0
    best_translation = (0, 0)
    best_gof = 0
    best_haus = 0
    if pand['perceel_id'].iloc[0] == 'HVS00N9252':
        bgt_geometry = pand[bgt_outline].iloc[0].buffer(buffer + 4)
    else:
        bgt_geometry = pand[bgt_outline].iloc[0].buffer(buffer)
    if isinstance(pand[aligned_column].iloc[0], MultiPolygon):
        apply_translation = True
    else:
        apply_translation = False

    angles = np.arange(-180, 180, angle_step)
    scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
    translations = np.arange(-max_translation, max_translation + translation_step,
                             translation_step) if apply_translation else np.array([0])

    # Generate all transformation combinations
    transform_params = list(itertools.product(scales, angles, translations, translations))

    def apply_transformations(scale_factor, angle, dx, dy):
        """ Apply scale, rotation, and translation if applicable. """
        transformed_geometries = np.array(pand[aligned_column])  # Convert to NumPy array

        # Apply transformations
        transformed_geometries = np.array([scale(g, xfact=scale_factor, yfact=scale_factor) for g in transformed_geometries])
        transformed_geometries = np.array([rotate(g, angle, origin='centroid') for g in transformed_geometries])
        if apply_translation:
            transformed_geometries = np.array([translate(g, xoff=dx, yoff=dy) for g in transformed_geometries])

        # Keep only valid geometries (inside BGT boundary)
        valid_mask = np.array([g.within(bgt_geometry) for g in transformed_geometries])
        if not valid_mask.any():
            return None, None, None  # Skip invalid cases

        transformed_geometries = transformed_geometries[valid_mask]

        # Compute scores in parallel
        gof_scores = np.array(Parallel(n_jobs=-1)(delayed(goodness_of_fit)(g, pand[bgt_outline].iloc[0]) for g in transformed_geometries))
        hausdorff_values = np.array(Parallel(n_jobs=-1)(delayed(averaged_hausdorff_distance)(g, pand[bgt_outline].iloc[0]) for g in transformed_geometries))

        mean_gof = np.mean(gof_scores)
        mean_hausdorff = np.mean(hausdorff_values)
        score, score_gof, score_haus = combined_score(mean_gof, mean_hausdorff, alpha)

        return transformed_geometries, score, (score_gof, score_haus, scale_factor, angle, (dx, dy))

    # Run grid search in parallel
    results = Parallel(n_jobs=-1)(delayed(apply_transformations)(s, a, dx, dy) for s, a, dx, dy in transform_params)

    # Find the best transformation
    best_score = -np.inf
    best_geometries = None
    best_params = None

    for transformed_geometries, score, params in results:
        if transformed_geometries is not None and score > best_score:
            best_score = score
            best_geometries = transformed_geometries
            best_params = params  # (score_gof, score_haus, scale, angle, translation)

    # Store results in DataFrame
    # pand['optimized_geometry'] = best_geometries
    pand["optimized_geometry"] = best_geometries
    pand['score_gof'] = best_params[0] if best_params else None
    pand['score_haus'] = best_params[1] if best_params else None
    pand['score'] = best_score
    pand['angle'] = best_params[3] if best_params else None
    pand['scale'] = best_params[2] if best_params else None
    pand['translation_x'] = best_params[4][0] if best_params else None
    pand['translation_y'] = best_params[4][1] if best_params else None
    if best_geometries is not None:
        best_geometries = [refine_alignment(pand["bgt_outline"].iloc[i], g)
                           for i, g in enumerate(best_geometries)]
    pand["optimized_geometry"] = best_geometries
    return pand


from scipy.spatial import KDTree
from shapely.geometry import Point
import numpy as np


def compute_vertex_angles(geometry):
    """Compute vertex angles for Polygons and MultiPolygons."""
    angles = []

    if isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            angles.extend(compute_vertex_angles(poly))  # Recursively compute angles for each polygon
    elif isinstance(geometry, Polygon):
        coords = np.array(geometry.exterior.coords)

        for i in range(1, len(coords) - 1):  # Skip first and last as they repeat
            v1 = coords[i - 1] - coords[i]
            v2 = coords[i + 1] - coords[i]

            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

            angle = np.arccos(np.clip(dot_product / norm_product, -1, 1))  # Clamp for numerical stability
            angles.append((tuple(coords[i]), np.degrees(angle)))  # Convert to degrees

    return angles


# from shapely.geometry import Polygon
# import numpy as np
# from scipy.spatial import KDTree
# from skimage.transform import estimate_transform
#
# def find_best_anchors(ref_vertices, aligned_vertices, distance_threshold, angle_threshold):
#     """Find three well-matching anchor points."""
#     ref_tree = KDTree([v[0] for v in ref_vertices])
#     matched_points = []
#
#     for v, angle in aligned_vertices:
#         dist, idx = ref_tree.query(v, k=1)
#         ref_angle = ref_vertices[idx][1]
#
#         if dist < distance_threshold and abs(ref_angle - angle) < angle_threshold:
#             matched_points.append((v, ref_vertices[idx][0]))  # (aligned_point, reference_point)
#
#     if len(matched_points) < 3:
#         return None  # Not enough matches
#
#     # Select 3 well-distributed matches
#     matched_points = np.array(matched_points)[:3]  # Take the first 3 (could improve selection)
#
#     return matched_points[:, 0], matched_points[:, 1]  # Separate aligned & reference points
#
# def transform_polygon(aligned_geom, matched_aligned, matched_ref):
#     """Compute transformation and apply it to the polygon."""
#     if len(matched_aligned) != 3 or len(matched_ref) != 3:
#         return aligned_geom  # Not enough points, return as is
#
#     # Estimate transformation (similarity: rotation, scaling, translation)
#     transform = estimate_transform('similarity', np.array(matched_aligned), np.array(matched_ref))
#
#     # Apply transformation to all polygon coordinates
#     transformed_coords = transform(np.array(aligned_geom.exterior.coords))
#
#     return Polygon(transformed_coords)
#
#
# def refine_alignment(reference_geom, aligned_geom, distance_threshold=2, angle_threshold=1):
#     """Refine alignment by finding best 3 anchor points and transforming the polygon."""
#     if not isinstance(reference_geom, Polygon) or not isinstance(aligned_geom, Polygon):
#         return aligned_geom  # Skip multipolygons
#
#     ref_vertices = compute_vertex_angles(reference_geom)
#     aligned_vertices = compute_vertex_angles(aligned_geom)
#
#     anchors = find_best_anchors(ref_vertices, aligned_vertices, distance_threshold, angle_threshold)
#
#     if anchors is None:
#         return aligned_geom  # Not enough matches, return original polygon
#
#     matched_aligned, matched_ref = anchors  # Safe unpacking now
#
#     return transform_polygon(aligned_geom, matched_aligned, matched_ref)


def refine_alignment(reference_geom, aligned_geom, distance_threshold=2, angle_threshold=1, visualize=True):
    """Align vertices within distance and angle constraints, with optional visualization."""

    # Skip MultiPolygons
    if reference_geom.geom_type == "MultiPolygon" or aligned_geom.geom_type == "MultiPolygon":
        print("Skipping MultiPolygons...")
        return aligned_geom

    ref_vertices = compute_vertex_angles(reference_geom)
    aligned_vertices = compute_vertex_angles(aligned_geom)

    adjusted_coords = []
    snap_lines = []  # Store lines for visualization

    for v, angle in aligned_vertices:
        # Find closest reference vertex manually
        closest_dist = float('inf')
        closest_vertex = None
        closest_angle = None

        for ref_v, ref_angle in ref_vertices:
            dist = np.linalg.norm(np.array(v) - np.array(ref_v))  # Euclidean distance

            if dist < closest_dist:
                closest_dist = dist
                closest_vertex = ref_v
                closest_angle = ref_angle

        # Check if closest vertex is within distance and angle thresholds
        if closest_dist < distance_threshold and abs(closest_angle - angle) < angle_threshold:
            adjusted_coords.append(closest_vertex)  # Snap to reference vertex
            snap_lines.append((v, closest_vertex))  # Store snap line
        else:
            adjusted_coords.append(v)  # Keep original vertex

    refined_polygon = Polygon(adjusted_coords)

    if visualize:
        plot_alignment(reference_geom, aligned_geom, refined_polygon, snap_lines)

    return refined_polygon  # Return refined geometry


def plot_alignment(reference_geom, aligned_geom, refined_geom, snap_lines):
    """Visualize reference, aligned, and adjusted geometries."""
    fig, ax = plt.subplots(figsize=(6, 6))

    def plot_polygon(polygon, color, label, marker='o'):
        """Helper function to plot polygon with vertices."""
        if not polygon or not polygon.is_valid:
            return  # Skip invalid polygons

        x, y = zip(*polygon.exterior.coords)
        ax.plot(x, y, color=color, linestyle='-', alpha=0.5, label=label)
        ax.scatter(x, y, color=color, s=40, edgecolor='k', marker=marker)  # Mark vertices

    # Plot original reference polygon (blue)
    plot_polygon(reference_geom, color='blue', label='Reference Polygon', marker='s')

    # Plot aligned polygon before refinement (red)
    plot_polygon(aligned_geom, color='red', label='Aligned Polygon', marker='x')

    # Plot refined polygon after snapping (green)
    plot_polygon(refined_geom, color='green', label='Refined Polygon', marker='o')

    # Draw lines between snapped points
    for v1, v2 in snap_lines:
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k--', alpha=0.7)  # Dashed line for snapping

    ax.legend()
    ax.set_title("Polygon Vertex Alignment")
    plt.show()


def combined_score(gof, hausdorff, alpha):
    """
    Combine GoF and Hausdorff into a single score.
    - `alpha` controls the weight:
        - alpha=0.5 means equal weighting
        - alpha > 0.5 favors GoF more
        - alpha < 0.5 favors minimizing Hausdorff more
    """
    hausdorff_normalized = 1 / (1 + hausdorff)
    combined_score = alpha * gof + (1 - alpha) * hausdorff_normalized
    return combined_score, gof, hausdorff_normalized

def get_edges(geometry):
    """
    Get the exterior edges of a polygon or multipolygon.
    """
    edges = []
    if isinstance(geometry, Polygon):
        # If it's a Polygon, get the exterior edges
        edges = [geometry.exterior.coords[i:i + 2] for i in range(len(geometry.exterior.coords) - 1)]
    elif isinstance(geometry, MultiPolygon):
        # If it's a MultiPolygon, iterate through each polygon
        for polygon in geometry.geoms:
            edges.extend([polygon.exterior.coords[i:i + 2] for i in range(len(polygon.exterior.coords) - 1)])
    return edges



# start code
start_time = time.time()
# options: text, area
scale_version = 'area'
# options: azimuth, arrow
rotate_version = 'azimuth'
# options: bbox, centroid
translation_version = 'centroid'
rotation_angles2 = [171.3,  180,  -43.5,  78.9,  6.8,  0.0,  121.6, 120,  22.2,7]
rotation_angles = {'HVS00N1878': 171.3, "HVS00N1882": 180, "HVS00N2359": -43.5, "HVS00N2643": 78.9, "HVS00N2848": 6.8, "HVS00N3211": 0.0, "HVS00N3723": 121.6, "HVS00N4216": 120, "HVS00N555": 22.2, "HVS00N9252":7}
alpha = 0.5
angles_list = []

kadpercelen = gp.read_file(kad_path)


perceel_list = []
# get all the 'Kadastrale aanduidingen' (Gemeente, Sectie, Perceelnr) in the deed files
for f in files:
    if f.endswith('.json'):
        parts = f.split('.')
        perceel_list.append(parts[0])


all_panden = []
for perceel in perceel_list:
    print(perceel)
    parts = perceel.split('_')

    # If there are multiple parcels (parts[4] and beyond)
    if len(parts) > 4:
        perceel_id = str(parts[1]) + str(parts[2]) + ''.join(parts[3:])
        # Group parcels based on their KAD_GEM and SECTIE and union their geometries
        perceel_ids = []
        all_geometries = []
        all_bgt_lokaal_ids = []
        all_bag_pnds = []

        for i in range(3, len(parts), 1):
            perceel_id = str(parts[1]) + str(parts[2]) + str(parts[i])
            perceel_ids.append(perceel_id)
            print(perceel_id)
            selection_perceel = kadpercelen[
                kadpercelen.KAD_GEM.eq(parts[1]) & kadpercelen.SECTIE.eq(parts[2]) & kadpercelen.PERCEELNUM.eq(
                    int(parts[i]))]
            bbx = selection_perceel.geometry.bounds
            bbox = f'{int(bbx.minx.iloc[0])},{int(bbx.miny.iloc[0])},{int(bbx.maxx.iloc[0])},{int(bbx.maxy.iloc[0])}'
            selection_perceel['perceel_id'] = perceel_id
            selection_perceel['perceel_id'] = selection_perceel['perceel_id'].astype(str)

            params = {'bbox': bbox, 'bbox-crs': 'http://www.opengis.net/def/crs/EPSG/0/28992',
                      'crs': 'http://www.opengis.net/def/crs/EPSG/0/28992'}
            response = requests.get(api_bgt, params=params)
            response_json = response.json()

            features = response_json.get('features', [])
            bgt_data = []
            for feature in features:
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                lokaal_id = properties.get('lokaal_id')
                bag_pnd = properties.get('bag_pnd', None)
                coordinates = geometry.get('coordinates', None)
                bgt_data.append({'bgt_lokaal_id': lokaal_id, 'bag_pnd': bag_pnd, 'geometry': coordinates})
                all_bgt_lokaal_ids.append(lokaal_id)
                all_bag_pnds.append(bag_pnd)

            for item in bgt_data:
                item['geometry'] = shape({'type': 'MultiPolygon', 'coordinates': item['geometry']})

            bgt_geom_all_temp = gp.GeoDataFrame(bgt_data, geometry='geometry', crs=28992)
            all_geometries.append(bgt_geom_all_temp)

        # After looping through all parcels, union the geometries
        if len(all_geometries) > 1:
            union_geom = all_geometries[0].unary_union
        else:
            union_geom = all_geometries[0].geometry.iloc[0]

        # Create the final GeoDataFrame for the unioned geometry
        combined_perceel_id = '_'.join(perceel_ids)
        bgt_geom_all = gp.GeoDataFrame(
            [{'perceel_id': perceel_id,
              'bgt_lokaal_id': ', '.join(set(all_bgt_lokaal_ids)),
              'bag_pnd': ', '.join(set(all_bag_pnds)),
              'geometry': union_geom}],
            geometry='geometry', crs=28992)


    else:
        # Handle single parcel case
        selection_perceel = kadpercelen[
            kadpercelen.KAD_GEM.eq(parts[1]) & kadpercelen.SECTIE.eq(parts[2]) & kadpercelen.PERCEELNUM.eq(
                int(parts[3]))]
        bbx = selection_perceel.geometry.bounds
        bbox = f'{int(bbx.minx.iloc[0])},{int(bbx.miny.iloc[0])},{int(bbx.maxx.iloc[0])},{int(bbx.maxy.iloc[0])}'
        selection_perceel['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
        selection_perceel['perceel_id'] = selection_perceel['perceel_id'].astype(str)

        # Connect to the BGT
        params = {'bbox': bbox, 'bbox-crs': 'http://www.opengis.net/def/crs/EPSG/0/28992',
                  'crs': 'http://www.opengis.net/def/crs/EPSG/0/28992'}
        response = requests.get(api_bgt, params=params)
        response_json = response.json()

        features = response_json.get('features', [])
        bgt_data = []
        for feature in features:
            properties = feature.get('properties', {})
            geometry = feature.get('geometry', {})
            lokaal_id = properties.get('lokaal_id')
            bag_pnd = properties.get('bag_pnd', None)
            coordinates = geometry.get('coordinates', None)
            bgt_data.append({'bgt_lokaal_id': lokaal_id, 'bag_pnd': bag_pnd, 'geometry': coordinates})

        # Convert to polygon geometries
        for item in bgt_data:
            item['geometry'] = shape({'type': 'MultiPolygon', 'coordinates': item['geometry']})

        bgt_geom_all = gp.GeoDataFrame(bgt_data, geometry='geometry', crs=28992)

    # now intersect with the actual geometry bc otherwise its too many polygons
    join_bgt = gp.sjoin(bgt_geom_all, selection_perceel, how='inner', predicate='intersects')
    join_bgt["overlap_area"] = join_bgt.geometry.intersection(selection_perceel.geometry.iloc[0]).area

    join_bgt = join_bgt[join_bgt["overlap_area"] >= 2]

    perceel_bgt = join_bgt.set_crs('epsg:28992', allow_override=True)
    if 'perceel_id_left' in join_bgt.columns and 'perceel_id_right' in join_bgt.columns:
        if (join_bgt['perceel_id_left'] == join_bgt['perceel_id_right']).all():
            perceel_bgt = join_bgt.drop(columns=['perceel_id_right']).rename(columns={'perceel_id_left': 'perceel_id'})

    perceel_bgt = perceel_bgt[['bgt_lokaal_id', 'bag_pnd', 'geometry', 'perceel_id']]
    # now get the rooms data
    with open(os.path.join(json_path, f'{perceel}.latest.json')) as f:
        data = json.load(f)

    roomIDs = []
    appartementsnummer = []
    ruimteomschrijving = []
    verdiepingsaanduiding = []
    geometry = []
    attachment = []
    room_polygons = []
    for r in data['rooms'].keys():
        roomIDs.append(r)
        addValue('appartementsnummer', appartementsnummer, r)
        addValue('ruimteomschrijving', ruimteomschrijving, r)
        addValue('verdiepingaanduiding', verdiepingsaanduiding, r)
        attachment.append(data['rooms'][r]['attachment'])


        if scale_version == 'area':
            scale_factor = 1

        pointDict = {}
        for pt in data['points'].keys():
            x, y = data['points'][pt]['position']
            pointDict[pt] = [
                x / scale_factor,
                # y is top down so first make bottom up
                (float(data['meta']['frontDimensions'][1]) - y) / scale_factor]

        geometry.append(egetGeometry(data['rooms'][r]['points'], pointDict))

    aktes_rooms = gp.GeoDataFrame(
        data=zip(roomIDs, verdiepingsaanduiding, appartementsnummer, ruimteomschrijving, attachment),
        geometry=geometry, crs="EPSG:28992",
        columns=['room', 'verdieping', 'appartement', 'ruimte', 'attachment'])

    aktes_rooms['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
    aktes_rooms['perceel_id'] = aktes_rooms['perceel_id'].astype(str)

    # join rooms with perceel through perceel ID
    pand_data = aktes_rooms.merge(perceel_bgt, on='perceel_id', how='left')
    # geometry_x = the akte geometry
    pand_data.rename(columns={'geometry_x': 'geom_akte_all', "geometry_y": "geom_bgt"}, inplace=True)
    pand_data = pand_data.set_geometry('geom_akte_all')


    # create a dataframe for the panden with the polygon outline of the BG
    rooms_bg = pand_data[pand_data['verdieping'].fillna('').str.lower().str.contains("begane grond")]
    pand_outline = rooms_bg.groupby('bgt_lokaal_id').agg({
        'geom_akte_all': lambda g: g.unary_union, 'perceel_id': 'first', 'bag_pnd': 'first', 'geom_bgt': 'first' })
    pand = gp.GeoDataFrame(pand_outline,geometry='geom_akte_all',crs=28992)

    pand.rename(columns={'geom_akte_all': 'geom_akte_bg'}, inplace=True)

    # join the dataframes
    combined_df = pand.merge(pand_data, left_on='bgt_lokaal_id', right_on='bgt_lokaal_id', how='left')
    combined_df.drop(columns=['geom_bgt_y'], inplace=True)

    outline = pand.groupby('perceel_id').agg({'geom_bgt': lambda g: g.unary_union})
    bgt_outline = gp.GeoDataFrame(outline, geometry='geom_bgt', crs=28992)
    bgt_outline = bgt_outline.rename_geometry('bgt_outline')
    pand = pand.merge(bgt_outline, on='perceel_id', how='left')

    pand = gp.GeoDataFrame(pand, geometry='geom_akte_bg')

    # rotate according to azimuth
    # pand_grouped = pand.groupby('perceel_id').apply(rotate_geom_azimuth)
    # pand = pand.merge(pand_grouped, on='bag_pnd', how='left')
    # pand = pand.loc[:, ~pand.columns.str.endswith('_x')]
    # pand.columns = pand.columns.str.replace('_y', '', regex=False)

    # scale according to ratio
    pand.set_geometry('bgt_outline', inplace=True)
    if len(pand.geometry.area) > 1:
        reference_area = pand.geometry.area.iloc[0]
    else:
        reference_area = pand.geometry.area
    pand.set_geometry('geom_akte_bg', inplace=True)
    if len(pand.geometry.area) > 1:
        akte_area = pand.geometry.area.iloc[0]
    else:
        akte_area = pand.geometry.area

    scale_factor = np.sqrt(reference_area / akte_area)

    # Extract scalar value from the Series
    if isinstance(scale_factor, pd.Series):
        scale_factor_value = scale_factor.iloc[0]  # Extract first element if it's a Series
    else:
        scale_factor_value = scale_factor  # Get first value if it's a single-element Series

    pand['geom_akte_bg_scaled'] = pand['geom_akte_bg'].apply(
        lambda g: scale(g, xfact=scale_factor_value, yfact=scale_factor_value, origin='centroid'))

    # pand_data['geom_akte_all_scaled'] = pand_data['geom_akte_all'].apply(
    #     lambda g: scale(g, xfact=scale_factor, yfact=scale_factor, origin='centroid'))

    pand.set_geometry('geom_akte_bg_scaled', inplace=True)
    pand['geom_akte_bg'] = pand['geom_akte_bg_scaled']
    # pand_data['geom_akte_all'] = pand_data['geom_akte_all_scaled']
    pand.drop(columns=['geom_akte_bg_scaled'], axis=1, inplace=True)
    # pand_data.drop(columns=['geom_akte_all_scaled'], axis=1, inplace=True)
    pand.set_geometry('geom_akte_bg', inplace=True)

    # allign centroids -> translation
    bgt_centroid = pand.bgt_outline.centroid.iloc[0]
    if len(pand.geometry.centroid) > 1:
        building_centroid = pand.geometry.centroid.iloc[0]
    else:
        building_centroid = pand.geometry.centroid

    translation_vector = (bgt_centroid.x - building_centroid.x,
                          bgt_centroid.y - building_centroid.y)

    pand.set_geometry('geom_akte_bg')
    pand['aligned_geometry'] = translate_polygon(pand.geometry, translation_vector)
    # still need to translate the rooms too
    # pand_data['geom_akte_all'] = translate_polygon(pand_data['geom_akte_all'], translation_vector)
    pand = gp.GeoDataFrame(pand, geometry='geom_bgt', crs='EPSG:28992')
    pand.plot()
    plt.show()

    pand = gp.GeoDataFrame(pand, geometry='aligned_geometry', crs='EPSG:28992')
    pand.plot()
    plt.show()
    # grid search optimization for rotation and scale
    pand = grid_search(pand, 'geom_akte_bg', "aligned_geometry", "bgt_outline", alpha=0.2, buffer=1,
                                     angle_step=1, scale_step=0.05, scale_range=(0.8, 1.2),
                                     translation_step=0.2, max_translation=1)

    # # test edge matching
    # aligned_geom = pand.loc[0, 'aligned_geometry']
    # reference_geom = pand.loc[0, 'geom_bgt']
    #
    # # Transform using edge matching
    # transformed_geom = edge_match_transform(aligned_geom, reference_geom)
    #
    # # Store back in DataFrame
    # pand.loc[0, 'optimized_geometry'] = transformed_geom

    pand = gp.GeoDataFrame(pand, geometry='optimized_geometry', crs='EPSG:28992')

    all_panden.append(pand)

panden = pd.concat(all_panden)


panden.to_csv(os.path.join("werkmap",'separate_pand.csv'), index=True)

import matplotlib.pyplot as plt

end_time = time.time()
# print("Executed time: ", end_time - start_time)
print(angles_list)

panden.set_geometry('optimized_geometry', crs='EPSG:28992')
print(panden.info())
panden2 = panden.drop(columns=['geom_akte_bg', 'bgt_outline', 'geom_bgt', 'aligned_geometry'])
# print("Average Hausdorff Score: ", panden2['score_haus'].sum()/len(panden2))
# print("Average GoF Score: ", panden2['score_gof'].sum()/len(panden2))
# print("Combined Score: ", panden2['score'].sum()/len(panden2))
os.makedirs("werkmap", exist_ok=True)

# Save the file in the "werkmap" folder
panden2.to_file(os.path.join("werkmap", f"snap_test_othermet.shp"), driver="ESRI Shapefile")

# panden2.to_file( "gof.shp", driver="ESRI Shapefile")



