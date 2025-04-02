import os
import json
from statistics import mode
import numpy as np
import requests
import geopandas as gp
import pandas as pd
from geopandas import GeoSeries
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, euclidean
from shapely.affinity import translate, scale, rotate
from shapely.geometry import shape
from shapely import geometry as geom
from shapely.geometry import Polygon, MultiPolygon, Point
import re
import time
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.measurement import hausdorff_distance
from shapely.ops import unary_union, snap, transform
import itertools
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from shapely.plotting import plot_polygon
from skimage.transform import estimate_transform
import seaborn as sns
from tqdm import tqdm

kad_path = r'C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\original_data\aanvullende_data\Hilversum\Percelen_aktes_Hilversum.shp'
json_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\original_data\gevectoriseerde_set\hilversum_set\observations\snapshots\latest"
out_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\werkmap\output"
api_bgt = 'https://api.pdok.nl/lv/bgt/ogc/v1/collections/pand/items'
files = os.listdir(json_path)


if not os.path.exists(out_path):
    os.mkdir(out_path)

# existing functions
def calcScale(bbxobj, pix) -> float:
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

floor_mapping = {
    "begane grond": 0,
    "kelder": -1,
    "eerste": 1,
    "tweede": 2,
    "derde": 3,
    "vierde": 4,
    "vijfde": 5,
    "zesde": 6,
    "zevende": 7,
    "achtste": 8,
    "negende": 9,
    "tiende": 10,
    "1e": 1,
    "2e": 2,
    "3e": 3,
    "4e": 4,
    "5e": 5,
    "6e": 6,
}


def map_floor(floor):
    """"map the floor text to an index number, if it does not find a mapping, it returns -999"""
    if pd.isna(floor) or not isinstance(floor, str):
        return -999

    # if the floor text contains one of the map values, it returns the corresponding index
    # for example "eerste verdieping" returns index 1
    floor = floor.lower().strip()
    for key in floor_mapping:
        if key in floor:
            return floor_mapping[key]

    # if the key is not in the mapping, but it contains a number, that number will be returned
    # for example "1e verdieping" returns index 1
    match = re.search(r'(\d+)', floor)
    if match:
        return int(match.group(1))

    return -999

def extrude_to_3d(geometry, floor_height=3, floor_index=0):
    """function to extrude geometry to 3D, with a predetermined floor height"""
    if geometry.is_empty:
        return None

    def to_3d(x, y, z=floor_index * floor_height):
        return (x, y, z)

    if isinstance(geometry, Polygon):
        return transform(lambda x, y: to_3d(x, y), geometry)

    elif isinstance(geometry, MultiPolygon):
        transformed_polygons = [transform(lambda x, y: to_3d(x, y), poly) for poly in geometry.geoms]
        return MultiPolygon(transformed_polygons)

    return None


def extract_multilines(geom):
    """ Extract exterior and interior geometries (LOOK AT) """
    lines = []

    if isinstance(geom, Polygon):
        lines.append(LineString(geom.exterior.coords))
        lines.extend(LineString(ring.coords) for ring in geom.interiors)

    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            lines.append(LineString(poly.exterior.coords))
            lines.extend(LineString(ring.coords) for ring in poly.interiors)

    return MultiLineString(lines)


def plot_geometries(ref_floor_geom, transformed_geometries, buffer_distance=0.2):
    """ plots the reference floor geometry, its buffered version, and the best transformed geometries. """
    fig, ax = plt.subplots(figsize=(8, 8))

    if isinstance(ref_floor_geom, gp.GeoSeries):
        geometries = ref_floor_geom.geometry
    else:
        geometries = [ref_floor_geom]

    # plot the lines instead of the polygons
    all_lines = []
    for geom in geometries:
        if isinstance(geom, Polygon):
            all_lines.append(extract_multilines(geom))
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                all_lines.append(extract_multilines(poly))

    buffered_lines = [line.buffer(buffer_distance) for multi_line in all_lines for line in multi_line.geoms]

    for geom in geometries:
        if isinstance(geom, Polygon):
            plot_polygon(geom, ax=ax, facecolor='none', add_points=False, edgecolor='blue', linewidth=1, label="Ground Floor")
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                plot_polygon(poly, ax=ax, facecolor='none', add_points=False, edgecolor='blue', linewidth=1, label="Ground Floor")

    for buffered_line in buffered_lines:
        plot_polygon(buffered_line, ax=ax, facecolor='none',add_points=False, edgecolor='blue', linestyle='dashed', linewidth=1.5,
                     label="Buffered Lines")

    if isinstance(transformed_geometries, Polygon):
        transformed_geometries = MultiPolygon([transformed_geometries])

    for poly in transformed_geometries.geoms:
        plot_polygon(poly, ax=ax, facecolor='none', edgecolor='red', add_points=False, linewidth=1, label="Transformed Geometry")

    ax.set_title("Shape Similarity")
    plt.show()


def shape_similarity_score(ref_geom, geom, buffer_distance=0.5):
    """
    Computes the shape similarity score based on the percentage of the floors boundary
    that falls within a buffered version of the reference floors boundary.

    Returns:
    float: Similarity score (0 to 1), where 1 means complete alignment.
    (LOOK AT: returns sometimes scores above 1, clean up types)
    """

    if isinstance(ref_geom, gp.GeoSeries):
        ref_geometries = ref_geom.geometry
    else:
        ref_geometries = [ref_geom]

    # extract the lines instead of the polygons
    all_ref_lines = []
    for rgeom in ref_geometries:
        if isinstance(rgeom, Polygon):
            all_ref_lines.append(extract_multilines(rgeom))
        elif isinstance(rgeom, MultiPolygon):
            for poly in rgeom.geoms:
                all_ref_lines.append(extract_multilines(poly))

    # buffer the lines of the floor below
    buffered_ref_lines = [line.buffer(buffer_distance) for multi_line in all_ref_lines for line in multi_line.geoms]

    if isinstance(geom, gp.GeoSeries):
        geometries = geom.geometry
    else:
        geometries = [geom]

    all_geom_lines = []
    for g in geometries:
        if isinstance(g, Polygon):
            all_geom_lines.append(extract_multilines(g))
        elif isinstance(g, MultiPolygon):
            for poly in g.geoms:
                all_geom_lines.append(extract_multilines(poly))


    total_length = sum(line.length for multi_line in all_geom_lines for line in multi_line.geoms)
    # compute the intersection of the floor lines that fall in the buffered lines
    intersected_length = sum(line.intersection(buffered_line).length
                             for buffered_line in buffered_ref_lines
                             for multi_line in all_geom_lines
                             for line in multi_line.geoms)

    if total_length == 0:
        return 0.0

    return intersected_length / total_length



def translate_polygon(geometries, translation_vector):
    if isinstance(translation_vector[0], pd.Series):
        dx, dy = translation_vector[0].iloc[0], translation_vector[1].iloc[0]
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
    greatest distance between any point in the polygon and the closest point in the reference
    """
    if isinstance(polygon, (Polygon, MultiPolygon)) and isinstance(reference, (Polygon, MultiPolygon)):
        return hausdorff_distance(polygon, reference)
    return float('inf')


def snap_floors_to_reference(best_geometries, below_floor_geom, threshold=0.2, simplification_tolerance=0.1):
    """
    snaps the floor geometry to the reference floor (either the ground floor, or the floor below the current floor) using Shapely's snap function, after simplifying.
    """

    if isinstance(below_floor_geom, Polygon):
        below_floor_geom = MultiPolygon([below_floor_geom])

    # the polygons are simplfied to remove any unnecessary vertices which could impact the snapping
    simplified_below = below_floor_geom.simplify(simplification_tolerance, preserve_topology=True)
    simplified_best = best_geometries.simplify(simplification_tolerance, preserve_topology=True)

    # the union of the floor below, to ensure we only snap the outside boundary, not the interior
    below_union = unary_union(simplified_below)

    # snap the best geometries to the reference floor geometry
    snapped_floors = snap(simplified_best, below_union, threshold)

    return snapped_floors

def extract_boundary_points(geometry):
    """
    extracts the boundary for polygons and multipolygons
    """
    if isinstance(geometry, Polygon):
        return np.array(geometry.exterior.coords)
    elif isinstance(geometry, MultiPolygon):
        return np.vstack([np.array(poly.exterior.coords) for poly in geometry.geoms])
    return np.array([])


def averaged_hausdorff_distance(polygon, reference):
    """
    computes the averaged Hausdorff distance between the polygon and its reference.
    """
    if not isinstance(polygon, (Polygon, MultiPolygon)) or not isinstance(reference, (Polygon, MultiPolygon)):
        return float('inf')

    # extract boundary points
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


def get_polygon_edges(polygon):
    """extracts edges (as line segments) from a polygon."""
    edges = []

    if isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            edges.extend(get_polygon_edges(poly))
    elif isinstance(polygon, Polygon):
        coords = list(polygon.exterior.coords)
        edges = [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]

    return edges


def grid_search_room(floor_geom, ground_floor_geom,
                translation_step=0.2):
    """
    performs a grid search over translations to optimize shape similarity.
    """
    ground_floor_geom = gp.GeoSeries(ground_floor_geom)
    boundary_points = extract_boundary_points(ground_floor_geom.unary_union)
    max_translation = np.max(pdist(boundary_points))/2
    print(max_translation)

    if isinstance(floor_geom, Polygon):
        floor_multipolygon = MultiPolygon([floor_geom])
    else:
        # If it's already a collection, ensure all are valid and make them a MultiPolygon
        floor_polygons = [geom for geom in floor_geom if geom.is_valid]
        floor_multipolygon = MultiPolygon(floor_polygons)

    # For ground floor or reference floor (same approach for multipolygon)
    if isinstance(ground_floor_geom, Polygon):
        ground_floor_multipolygon = MultiPolygon([ground_floor_geom])
    else:
        ground_floor_polygons = [geom for geom in ground_floor_geom if geom.is_valid]
        ground_floor_multipolygon = MultiPolygon(ground_floor_polygons)
    translations = np.arange(-max_translation, max_translation + translation_step,
                             translation_step)

    # Generate all transformation combinations
    transform_params = list(itertools.product(translations, translations))
    def apply_transformations(dx, dy):

        transformed_geometries = (floor_multipolygon)
        # Apply transformations
        transformed_geometries = translate(transformed_geometries, xoff=dx, yoff=dy)

        # Compute scores in parallel
        score_sim = shape_similarity_score(ground_floor_multipolygon, transformed_geometries)
        score = score_sim
        return transformed_geometries, score, (score_sim, (dx, dy))


    # Run grid search in parallel
    total_combinations = len(transform_params)
    results = Parallel(n_jobs=-1)(
        delayed(apply_transformations)(dx, dy)
        for dx, dy in tqdm(transform_params, total=total_combinations, desc="Grid Search Progress")
    )
    # Find the best transformation
    best_score = -np.inf
    best_geometries = None
    best_params = None
    for transformed_geometries, score, params in results:
        if transformed_geometries is not None and score > best_score:
            best_score = score
            best_geometries = transformed_geometries
            best_params = params
    print("best score", best_score)
    best_geometries = snap_floors_to_reference(best_geometries, ground_floor_geom)
    return best_geometries



def grid_search(pand, aligned_column, bgt_outline, pand_data, room_geom, good_fit,
                alpha,
                buffer=0.5,
                angle_step=1,
                scale_step=0.05, scale_range=(0.8, 1.2),
                translation_step=0.2, max_translation=2.0):
    """
    Performs a grid search over rotation angles, scales, and translations to optimize Goodness of Fit (GoF) and hausdorff distance.
    """

    if isinstance(pand[aligned_column].iloc[0], Polygon) and isinstance(pand[bgt_outline].iloc[0], MultiPolygon):
        alpha = 0.8

    print(alpha)

    # only apply translation when its multipolygons
    if isinstance(pand[aligned_column].iloc[0], MultiPolygon) or isinstance(pand[bgt_outline].iloc[0],
                                                                            MultiPolygon) or good_fit == False:
        apply_translation = True
    else:
        apply_translation = False

    # Compute max_translation dynamically based on the bgt_outline geometry
    bgt_geom = pand[bgt_outline].iloc[0]
    boundary_points = extract_boundary_points(bgt_geom)
    max_translation = np.max(pdist(boundary_points))/4

    angles = np.arange(-180, 180, angle_step)
    scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
    translations = np.arange(-max_translation, max_translation + translation_step,
                             translation_step) if apply_translation else np.array([0])

    # Generate all transformation combinations
    transform_params = list(itertools.product(scales, angles, translations, translations))

    def apply_transformations(scale_factor, angle, dx, dy):
        """ Apply scale, rotation, and translation if applicable. """
        transformed_geometries = np.array(pand[aligned_column])
        # Apply transformations
        transformed_geometries = np.array([scale(g, xfact=scale_factor, yfact=scale_factor) for g in transformed_geometries])
        transformed_geometries = np.array([rotate(g, angle, origin='centroid') for g in transformed_geometries])
        if apply_translation:
            transformed_geometries = np.array([translate(g, xoff=dx, yoff=dy) for g in transformed_geometries])


        if good_fit:
            bgt_geometry = pand[bgt_outline].iloc[0].buffer(buffer)
        else:
            bgt_geometry = pand[bgt_outline].iloc[0].buffer(buffer + 5)

        valid_mask = np.array([g.within(bgt_geometry) for g in transformed_geometries])
        if not valid_mask.any():
            return None, None, None
        transformed_geometries = transformed_geometries[valid_mask]

        # Compute scores in parallel
        hausdorff_values = np.array(Parallel(n_jobs=-1)(delayed(averaged_hausdorff_distance)(g, pand[bgt_outline].iloc[0]) for g in transformed_geometries))
        mean_hausdorff = np.mean(hausdorff_values)
        gof_scores = np.array(Parallel(n_jobs=-1)(delayed(goodness_of_fit)(g, pand[bgt_outline].iloc[0]) for g in transformed_geometries))
        mean_gof = np.mean(gof_scores)

        score, score_gof, score_haus = combined_score(mean_gof, mean_hausdorff, alpha)

        return transformed_geometries, score, (score_gof, score_haus, scale_factor, angle, (dx, dy))

    # Run grid search in parallel
    results = Parallel(n_jobs=-1)(delayed(apply_transformations)(s, a, dx, dy) for s, a, dx, dy in transform_params)

    # Find the best transformation
    best_score = -np.inf
    best_geometries = None
    best_params = None
    pand_data['transformed_akte_bg'] = None
    for transformed_geometries, score, params in results:
        if transformed_geometries is not None and score > best_score:
            for value in transformed_geometries:
                pand_data['transformed_akte_bg'] = value
            best_score = score
            best_geometries = transformed_geometries
            best_params = params

    # pand_data.set_geometry(pand_data['transformed_akte_bg'])
    # pand_data.plot()
    # plt.show()

    common_centroid = pand.aligned_geometry.centroid.iloc[0]  # Get the first centroid from the Series
    # Transform geometries
    transformed_rooms = list(pand_data[room_geom])

    # Apply scale, rotation, and translation
    if best_params is not None:
        transformed_rooms = np.array(
            [scale(g, xfact=best_params[2], yfact=best_params[2], origin=common_centroid) for g in transformed_rooms])
        transformed_rooms = np.array([rotate(g, best_params[3], origin=common_centroid) for g in transformed_rooms])

    if apply_translation and best_params is not None:
        transformed_rooms = np.array(
            [translate(g, xoff=best_params[4][0], yoff=best_params[4][1]) for g in transformed_rooms])

    # Store results in DataFrame
    pand_data['optimized_rooms'] = transformed_rooms
    pand["optimized_geometry"] = best_geometries

    pand.reset_index(drop=True, inplace=True)
    pand_data.reset_index(drop=True, inplace=True)

    pand_data = pand_data.merge(pand[['bag_pnd', 'geom_akte_bg', 'bgt_outline']], on='bag_pnd', how='left')

    pand_data["optimized_rooms"] = transformed_rooms

    if transformed_rooms is not None and pand_data['transformed_akte_bg'].iloc[0] is not None:
        transformed_rooms = [refine_alignment(pand_data["bgt_outline"].iloc[i], pand_data['transformed_akte_bg'].iloc[i], g)
                           for i, g in enumerate(transformed_rooms)]

    pand_data["optimized_rooms"] = transformed_rooms
    return pand



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


def is_almost_collinear(points, tolerance=5):
    """Check if three points are almost collinear using the area of the triangle they form."""
    if len(points) < 3:
        return True  # Not enough points

    p1, p2, p3 = points[:3]
    area = abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)

    return area < tolerance



def rigid_transform_polygon(aligned_geom, matched_aligned, matched_ref):
    """Compute a rigid transformation and return the transform object instead of applying it to the polygon or MultiPolygon."""

    if matched_aligned is None or matched_ref is None:
        print("Skipping transformation: Not enough matching points.")
        return None

    # Estimate transformation: rotation, scale, translation
    transform = estimate_transform('similarity', np.array(matched_aligned), np.array(matched_ref))

    return transform

    # Apply transformation
def transform_polygon(polygon, transform):
    transformed_coords = transform(np.array(polygon.exterior.coords))
    transformed_polygon = Polygon(transformed_coords[:, :2])
    return transformed_polygon if transformed_polygon.is_valid else polygon


def remove_collinear_vertices(polygon, tolerance=0.01):
    """Remove nearly collinear points from a polygon using Shapely's simplify()."""
    simplified_polygon = polygon.simplify(tolerance, preserve_topology=True)
    return simplified_polygon if simplified_polygon.is_valid else polygon

def compute_edges_with_angles(geom):
    """Compute midpoints, angles, and lengths of edges for a Polygon or MultiPolygon."""
    edges = []

    if isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        polygons = [geom]

    for polygon in polygons:
        coords = list(polygon.exterior.coords)

        # get the midpoint, angle and length of each e
        for i in range(len(coords) - 1):
            p1, p2 = np.array(coords[i]), np.array(coords[i + 1])
            midpoint = (p1 + p2) / 2
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            length = np.linalg.norm(p2 - p1)
            edges.append((midpoint, angle, length))

    return edges


def find_best_edge_anchors(ref_edges, aligned_edges, distance_threshold, angle_threshold, length_threshold=0.2):
    """Find edge anchor points"""
    ref_tree = KDTree([e[0] for e in ref_edges])
    matched_aligned = []
    matched_ref = []

    for midpoint, angle, length in aligned_edges:
        dist, idx = ref_tree.query(midpoint, k=1)
        ref_midpoint, ref_angle, ref_length = ref_edges[idx]
        if (dist < distance_threshold and
            abs(ref_angle - angle) < angle_threshold and
            abs(ref_length - length) / ref_length < length_threshold):
            matched_aligned.append(midpoint)
            matched_ref.append(ref_midpoint)

    if len(matched_aligned) < 3:
        return None

    # make sure the matched edges are not collinear
    for i in range(len(matched_aligned) - 2):
        if not is_almost_collinear([matched_aligned[i], matched_aligned[i + 1], matched_aligned[i + 2]]):
            return np.array(matched_aligned[i:i + 3]), np.array(matched_ref[i:i + 3])

    return None


def refine_alignment(reference_geom, aligned_geom, geom_rooms, distance_threshold=5, angle_threshold=3):
    """Refine alignment by finding best anchor edges and applying a rigid transformation."""
    reference_geom = remove_collinear_vertices(reference_geom)
    aligned_geom = remove_collinear_vertices(aligned_geom)

    ref_edges = compute_edges_with_angles(reference_geom)
    aligned_edges = compute_edges_with_angles(aligned_geom)

    anchors = find_best_edge_anchors(ref_edges, aligned_edges, distance_threshold, angle_threshold)

    if anchors is None:
        return aligned_geom

    matched_aligned, matched_ref = anchors

    transform =  rigid_transform_polygon(aligned_geom, matched_aligned, matched_ref)
    if aligned_geom.geom_type == "Polygon":
        return transform_polygon(geom_rooms, transform)

    elif aligned_geom.geom_type == "MultiPolygon":
        transformed_polygons = [transform_polygon(poly, transform) for poly in geom_rooms.geoms]
        return MultiPolygon([poly for poly in transformed_polygons if poly.is_valid])


def calculate_polygon_score(polygon1, polygon2, simplify_tolerance=0.01, distance_weight=0.5):
    """
    Calculate the similarity score between two polygons based on the number of edges and
    the maximum distance between any two edges.

    Args:
    - polygon1 (Polygon or GeoSeries): The first polygon (or GeoSeries of polygons) to compare.
    - polygon2 (Polygon or GeoSeries): The second polygon (or GeoSeries of polygons) to compare.
    - simplify_tolerance (float): The tolerance for simplifying the geometry (default is 0.01).

    Returns:
    - similarity_score (float): A score between 0 (different) and 1 (similar) indicating how similar the polygons are.
    """
    # Check if polygon1 and polygon2 are GeoSeries or individual Polygons
    if isinstance(polygon1, (GeoSeries, pd.Series)):
        polygon1 = polygon1.iloc[0]

    if isinstance(polygon2, (GeoSeries, pd.Series)):
        polygon2 = polygon2.iloc[0]
    simplified_polygon1 = polygon1.simplify(tolerance=simplify_tolerance)
    simplified_polygon2 = polygon2.simplify(tolerance=simplify_tolerance)

    # Function to extract edges from a polygon
    def extract_edges(polygon):
        # Check if the input is a Polygon or MultiPolygon
        if isinstance(polygon, Polygon):
            coords = list(polygon.exterior.coords)
            edges = [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
        elif isinstance(polygon, MultiPolygon):
            edges = []
            for poly in polygon.geoms:  # Use .geoms to access individual polygons
                coords = list(poly.exterior.coords)
                edges.extend([(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
        else:
            raise ValueError("Unsupported geometry type")
        return edges


    # Extract edges from both simplified polygons
    edges_polygon1 = extract_edges(simplified_polygon1)
    edges_polygon2 = extract_edges(simplified_polygon2)

    # Count the number of edges
    num_edges_polygon1 = len(edges_polygon1)
    num_edges_polygon2 = len(edges_polygon2)

    # Normalize the number of edges difference (closer to 0 means more similar)
    edge_count_diff = abs(num_edges_polygon1 - num_edges_polygon2)
    max_edge_count = max(num_edges_polygon1, num_edges_polygon2)
    edge_similarity = 1 - (edge_count_diff / max_edge_count if max_edge_count > 0 else 0)

    # Calculate the maximum distance between any two edges
    max_distance = 0
    for edge1 in edges_polygon1:
        for edge2 in edges_polygon2:
            dist = simplified_polygon1.distance(simplified_polygon2)
            max_distance = max(max_distance, dist)

    # Normalize the distance difference (smaller distance is more similar)
    max_possible_distance = simplified_polygon1.bounds[2] - simplified_polygon1.bounds[0]  # max x-axis distance
    distance_similarity = 1 - (max_distance / max_possible_distance if max_possible_distance > 0 else 0)

    # Calculate final similarity score (average of edge similarity and distance similarity)
    similarity_score = (edge_similarity + distance_weight * distance_similarity) / (1 + distance_weight)
    return similarity_score


def fetch_3dbag_data(bag_id):
    """Fetch data from 3DBAG API using the BAG identificatie."""
    formatted_id = f"NL.IMBAG.Pand.{bag_id}"
    url = f"https://api.3dbag.nl/collections/pand/items/{formatted_id}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {bag_id}. Status Code: {response.status_code}")
        return None


#========================================== START CODE ================================================================
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

#========================================== PREPROCESSING =========================================================

perceel_list = []
# get all the 'Kadastrale aanduidingen' (Gemeente, Sectie, Perceelnr) in the deed files
for f in files:
    if f.endswith('.json'):
        parts = f.split('.')
        perceel_list.append(parts[0])


all_panden = []
all_panden_rooms = []
for perceel in perceel_list:
    print(perceel)
    parts = perceel.split('_')
    pand_floors = []

    # If there are multiple parcels (parts[4] and beyond)
    if len(parts) > 4:
        perceel_id_full = str(parts[1]) + str(parts[2]) + ''.join(parts[3:])
        perceel_ids = []
        all_bgt_lokaal_ids = []
        all_bag_pnds = []

        # Store percelen geometries to compute the total bbox
        perceel_geometries = []

        for i in range(3, len(parts), 1):
            perceel_id = str(parts[1]) + str(parts[2]) + str(parts[i])
            perceel_ids.append(perceel_id)
            print(perceel_id)

            selection_perceel = kadpercelen[
                (kadpercelen.KAD_GEM == parts[1]) &
                (kadpercelen.SECTIE == parts[2]) &
                (kadpercelen.PERCEELNUM == int(parts[i]))
                ]
            print("selection_perceel", selection_perceel)

            perceel_geometries.append(selection_perceel.geometry.iloc[0])  # Store for bbox calculation

        # Compute total bounding box from all selected percelen
        total_bounds = gp.GeoSeries(perceel_geometries).total_bounds
        bbox = f'{int(total_bounds[0])},{int(total_bounds[1])},{int(total_bounds[2])},{int(total_bounds[3])}'


        # Make a single API request for the entire bbox
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

        # Combine all perceel geometries into a single MultiPolygon for filtering
        perceel_union = gp.GeoSeries(perceel_geometries).unary_union
        selection_perceel = gp.GeoDataFrame({'geometry': [perceel_union]}, crs=28992)

        # Intersect before storing the geometry
        join_bgt = gp.sjoin(bgt_geom_all_temp, selection_perceel, how='inner', predicate='intersects')
        join_bgt["overlap_area"] = join_bgt.geometry.intersection(perceel_union).area

        # Apply filtering before storing geometries
        join_bgt = join_bgt[join_bgt["overlap_area"] >= 10]

        # Union only the filtered geometries
        if not join_bgt.empty:
            union_geom = join_bgt.unary_union

            # Create final GeoDataFrame
            bgt_geom_all = gp.GeoDataFrame(
                [{'perceel_id': perceel_id_full,
                  'bgt_lokaal_id': ', '.join(set(all_bgt_lokaal_ids)),
                  'bag_pnd': ', '.join(set(all_bag_pnds)),
                  'geometry': union_geom}],
                geometry='geometry', crs=28992)

        perceel_bgt = bgt_geom_all.set_crs('epsg:28992', allow_override=True)
        perceel_bgt = perceel_bgt[['bgt_lokaal_id', 'bag_pnd', 'geometry', 'perceel_id']]



    else:
        selection_perceel = kadpercelen[
            kadpercelen.KAD_GEM.eq(parts[1]) & kadpercelen.SECTIE.eq(parts[2]) & kadpercelen.PERCEELNUM.eq(
                int(parts[3]))]
        bbx = selection_perceel.geometry.bounds
        bbox = f'{int(bbx.minx.iloc[0])},{int(bbx.miny.iloc[0])},{int(bbx.maxx.iloc[0])},{int(bbx.maxy.iloc[0])}'
        selection_perceel['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
        selection_perceel['perceel_id'] = selection_perceel['perceel_id'].astype(str)

        # connect to the BGT
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

    if len(parts) > 4:
        aktes_rooms['perceel_id'] = str(parts[1]) + str(parts[2]) + ''.join(parts[3:])
    else:
        aktes_rooms['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
    aktes_rooms['perceel_id'] = aktes_rooms['perceel_id'].astype(str)

    # join rooms with perceel through perceel ID
    pand_data = aktes_rooms.merge(perceel_bgt, on='perceel_id', how='left')
    # geometry_x = the akte geometry
    pand_data.rename(columns={'geometry_x': 'geom_akte_all', "geometry_y": "geom_bgt"}, inplace=True)
    pand_data = pand_data.set_geometry('geom_akte_all')


    # create a dataframe for the panden with the polygon outline of the BG
    rooms_bg = pand_data[pand_data['verdieping'].fillna('').str.lower().str.contains("begane grond")]
    rooms_bg['geom_akte_all_copy'] = rooms_bg["geom_akte_all"]
    pand_outline = rooms_bg.groupby('bgt_lokaal_id').agg({
        'geom_akte_all_copy': lambda g: g.unary_union, 'perceel_id': 'first', 'bag_pnd': 'first', 'geom_bgt': 'first', 'geom_akte_all': 'first' })
    pand = gp.GeoDataFrame(pand_outline,geometry='geom_akte_all_copy',crs=28992)
    pand.rename(columns={'geom_akte_all_copy': 'geom_akte_bg'}, inplace=True)

    # join the dataframes
    combined_df = pand.merge(pand_data, left_on='bgt_lokaal_id', right_on='bgt_lokaal_id', how='inner')
    combined_df.drop(columns=['geom_bgt_y'], inplace=True)

    combined_df.to_csv(os.path.join("werkmap", f'combined_df.csv'), index=False)

    outline = pand.groupby('perceel_id').agg({'geom_bgt': lambda g: g.unary_union})
    bgt_outline = gp.GeoDataFrame(outline, geometry='geom_bgt', crs=28992)
    bgt_outline = bgt_outline.rename_geometry('bgt_outline')
    pand = pand.merge(bgt_outline, on='perceel_id', how='left')

    pand = gp.GeoDataFrame(pand, geometry='geom_akte_bg')

    # ========================================== INITIALISING =========================================================


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
    pand.set_geometry('geom_akte_bg_scaled', inplace=True)

    centroid_bg = pand['geom_akte_bg'].centroid.iloc[0]
    # scale the rooms
    pand_data['geom_akte_all_scaled'] = pand_data['geom_akte_all'].apply(
        lambda g: scale(g, xfact=scale_factor_value, yfact=scale_factor_value, origin=centroid_bg))

    pand_data.set_geometry('geom_akte_all_scaled', inplace=True)



    # allign centroids -> translation
    pand.set_geometry('geom_akte_bg_scaled', inplace=True)
    bgt_centroid = unary_union(pand.bgt_outline).centroid
    building_centroid = pand.geometry.unary_union.centroid

    translation_vector = (bgt_centroid.x - building_centroid.x,
                          bgt_centroid.y - building_centroid.y)

    pand.set_geometry('geom_akte_bg_scaled')
    pand['aligned_geometry'] = translate_polygon(pand.geometry, translation_vector)
    pand.set_geometry('aligned_geometry')


    # translate the rooms on top of each other
    pand_data = gp.GeoDataFrame(pand_data, geometry='geom_akte_all', crs='EPSG:28992')
    pand_data.set_geometry('geom_akte_all_scaled', inplace=True)

    floors = pand_data.groupby('verdieping')

    pand_data['aligned_rooms'] = None

    for floor, floor_data in floors:
        floor_centroid = floor_data.geometry.unary_union.centroid
        translation_vector = (bgt_centroid.x - floor_centroid.x,
                              bgt_centroid.y - floor_centroid.y)

        pand_data.loc[floor_data.index, 'aligned_rooms'] = floor_data.geometry.apply(
            lambda geom: translate(geom, xoff=translation_vector[0], yoff=translation_vector[1])
        )

    pand_data = gp.GeoDataFrame(pand_data, geometry='aligned_rooms', crs='EPSG:28992')

    score = calculate_polygon_score(pand['aligned_geometry'], pand['bgt_outline'])
    good_fit = True if score > 0.76 else False
    good_fit = True


    # ========================================== OPTIMISATION =========================================================

    # grid search optimization for rotation and scale

    pand.drop_duplicates()
    pand_data.drop(columns=["bag_pnd", "bgt_lokaal_id"])
    pand_data.drop_duplicates()

    pand = grid_search(pand, "aligned_geometry", "bgt_outline", pand_data, "aligned_rooms", good_fit,
                       alpha=0.2, buffer=1,
                       angle_step=1, scale_step=0.05, scale_range=(0.8, 1.2),
                       translation_step=1)



    # pand_data2 = pand_data.drop(columns=["bag_pnd", "bgt_lokaal_id", "geom_bgt"])
    # pand_data2 = pand_data.drop_duplicates()

    pand_data = gp.GeoDataFrame(pand_data, geometry='optimized_rooms', crs='EPSG:28992')

    pand_data['floor_index'] = pand_data['verdieping'].apply(map_floor)
    pand_data = pand_data[pand_data['floor_index'] != -999]
    pand_data.set_geometry('optimized_rooms')

    floors = pand_data.groupby('floor_index')
    optimized_geometries = {}

    for floor_index, floor_data in floors:
        print("floor_index:", floor_index)
        if floor_index == -1:
            below_floor_data = floors.get_group(floor_index + 1)
        elif floor_index == 0:
            below_floor_data = floors.get_group(floor_index)
        else:
            below_floor_data = floors.get_group(floor_index - 1)

        if floor_index - 1 in optimized_geometries and floor_index != 0:
            print("below outline is", floor_index -1)
            below_outline = optimized_geometries[floor_index - 1]
            if below_outline.geom_type == "MultiPolygon":
                below_outline = gp.GeoSeries([poly for poly in below_outline.geoms])
        else:
            below_outline = below_floor_data.geometry
        print("below_floor", below_floor_data)

        floor_outline = floor_data.geometry


        similarity_score = shape_similarity_score(below_outline, floor_outline)
        best_geom = grid_search_room(floor_outline, below_outline, translation_step=0.5)
        plot_geometries(below_outline, best_geom)
        optimized_geometries[floor_index] = best_geom
        print("Similarity score:", similarity_score)

    pand_data['optimized_rooms'] = pand_data['floor_index'].map(optimized_geometries)

    #============================== Height Estimate ==========================================

    null_verdieping_rooms = pand_data[pand_data['verdieping'].isna()]
    if null_verdieping_rooms.empty:
        # get the 3dbag data
        bag_height_mapping = {}

        # Extract unique BAG IDs from pand_data_full
        unique_bag_ids = set()
        for index, row in pand_data.iterrows():
            bag_ids = str(row["bag_pnd"]).split(",")  # Handle multiple IDs
            for bag_id in bag_ids:
                bag_id = bag_id.strip()
                unique_bag_ids.add(bag_id)

        # Process each BAG ID
        for bag_id in unique_bag_ids:
            data = fetch_3dbag_data(bag_id)

            # Default height if data is missing
            if data is None:
                print(f"No data found for bag id: {bag_id}, using default height.")
                bag_height_mapping[bag_id] = 2.8
                continue

            # Extract attributes from 3DBAG data
            else:
                bag_info = data["feature"]["CityObjects"][f"NL.IMBAG.Pand.{bag_id}"]["attributes"]
                b3_h_dak_max = bag_info.get("b3_h_dak_max")
                b3_h_maaiveld = bag_info.get("b3_h_maaiveld")
                total_pand_height = b3_h_dak_max - b3_h_maaiveld

                # Calculate total floors count (assuming floors is defined somewhere)
                total_floors = sum(1 for floor_index, _ in floors if floor_index >= 0)


                extrusion_height = total_pand_height / total_floors if total_floors > 0 else 2.8
                bag_height_mapping[bag_id] = extrusion_height


        # Assign computed heights to pand_data_full
        def get_extrusion_height(bag_pnd):
            bag_ids = str(bag_pnd).split(",")
            heights = [bag_height_mapping.get(bag_id.strip(), 2.8) for bag_id in bag_ids]
            return max(heights)  # Take the maximum height if multiple BAG IDs


        pand_data["extrusion_height"] = pand_data["bag_pnd"].apply(get_extrusion_height)

        # Print to verify
        print(pand_data[["bag_pnd", "extrusion_height"]])




    else:
        # Assign a new group identifier (e.g., 'section')
        null_verdieping_rooms['floor_index'] = -100  # Use a unique index

        # Process these rooms separately
        section_geometry = null_verdieping_rooms.geometry

        y_values = []

        for geom in section_geometry:
            if geom is not None:
                # Convert MultiPolygon to individual Polygons
                if geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        y_values.extend([point[1] for point in poly.exterior.coords])
                elif geom.geom_type == "Polygon":
                    y_values.extend([point[1] for point in geom.exterior.coords])

        # Convert to NumPy array for plotting
        y_values = np.array(y_values)


        from sklearn.cluster import KMeans, DBSCAN

        y_values_reshaped = y_values.reshape(-1, 1)

        # DBSCAN - detect clusters without knowing the amount of clusters beforehand
        # epsilon = max distance between floors to be considered part of the same cluster
        dbscan = DBSCAN(eps=0.5, min_samples=1)
        floor_indices = dbscan.fit_predict(y_values_reshaped)
        num_floors= max(floor_indices) + 1

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(y_values, bins=num_floors, edgecolor="black", alpha=0.7)
        plt.xlabel("Y-Coordinate")
        plt.ylabel("Frequency")
        plt.title("Histogram of Y-Values in Section")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Show plot
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=np.zeros_like(y_values), y=y_values, hue=floor_indices, palette="viridis", s=20)
        plt.xlabel("Dummy X-axis")
        plt.ylabel("Y-Coordinate")
        plt.title("DBSCAN Floor Clustering")
        plt.show()
        print("DBSCAN clustered floors:", floor_indices)

        kmeans = KMeans(n_clusters=num_floors, random_state=42)
        kmeans.fit(y_values_reshaped)

        # Assign each y-value to a cluster (floor)
        floor_labels = kmeans.labels_

        # Print cluster centers (approximate floor heights)
        print("Estimated Floor Heights:", kmeans.cluster_centers_)


        # get the extrusion heights
        sorted_centers = np.sort(kmeans.cluster_centers_, axis=0)
        print("Sorted Centers:", sorted_centers)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=np.zeros_like(y_values), y=y_values, hue=floor_labels, palette="tab10", s=20)
        plt.scatter(np.zeros_like(sorted_centers), sorted_centers, c="red", marker="x", label="Estimated Floor Heights")
        plt.xlabel("Dummy X-axis")
        plt.ylabel("Y-Coordinate")
        plt.title("KMeans Clustered Floor Heights")
        plt.legend()
        plt.show()
        # Calculate the extrusion height (difference in height between each floor)
        extrusion_heights = np.diff(sorted_centers, axis=0).flatten()  # Flatten to get 1D array of differences
        print("Extrusion Heights:", extrusion_heights)

        # Create a mapping of floor indices to the extrusion height values
        floor_to_extrusion = {index: extrusion_heights[i]
                              for i, index in enumerate(sorted(pand_data['floor_index'].unique()))}

        # Assign the extrusion height to each row based on the 'floor_index'
        pand_data['extrusion_height'] = pand_data['floor_index'].map(floor_to_extrusion)

    # Print the updated dataframe with extrusion heights
    print(pand_data[['verdieping', 'extrusion_height']])




    all_panden_rooms.append(pand_data)

panden_rooms = pd.concat(all_panden_rooms, ignore_index=True)

end_time = time.time()
print("Executed time: ", end_time - start_time)


# Apply floor mapping

panden_rooms['optimized_rooms_3d'] = panden_rooms.apply(
    lambda row: extrude_to_3d(row['optimized_rooms'], floor_height=3, floor_index=row['floor_index']),
    axis=1)

panden_rooms = gp.GeoDataFrame(panden_rooms, geometry='optimized_rooms', crs='EPSG:28992')
panden_rooms.set_geometry('optimized_rooms_3d')
panden_rooms.plot()
plt.show()
print(panden_rooms.info())
# panden_rooms2 = panden_rooms.drop(columns=['geom_bgt',  'aligned_rooms', 'geom_akte_all', 'geom_akte_all_scaled'])
# panden_rooms2.to_file(os.path.join("werkmap", f"optimized_rooms_edgematch.shp"), driver="ESRI Shapefile")

panden_rooms = gp.GeoDataFrame(panden_rooms, geometry='optimized_rooms_3d', crs='EPSG:28992')
panden_rooms2 = panden_rooms[['optimized_rooms_3d', 'extrusion_height']]
panden_rooms.to_csv(os.path.join("werkmap", 'separate_pand_rooms.csv'), index=True)

panden_rooms2.to_file(os.path.join("werkmap", "3211test.geojson"), driver="GeoJSON")
