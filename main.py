import math
import os
import statistics
import uuid
import json
from pathlib import Path
from typing import Dict, Any
import geopandas as gpd
import numpy as np
import requests
import geopandas as gp
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from shapely.affinity import translate, scale, rotate
from shapely.geometry import shape
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
from sklearn.cluster import KMeans, DBSCAN

api_bgt = 'https://api.pdok.nl/lv/bgt/ogc/v1/collections/pand/items'



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
    return Polygon(gl)

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
}


def map_floor(floor):
    """map the floor text to an index number, if it does not find a mapping, it returns -999"""
    if pd.isna(floor) or not isinstance(floor, str) or floor.strip() == "":
        return -100

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

def extrude_to_3d(geometry, maaiveld, floor_height=3, floor_index=0):
    """function to extrude geometry to 3D, with a given floor height"""
    if geometry.is_empty:
        return None

    def to_3d(x, y, z=(floor_index * floor_height)+maaiveld):
        return (x, y, z)

    if isinstance(geometry, Polygon):
        return transform(lambda x, y: to_3d(x, y), geometry)

    elif isinstance(geometry, MultiPolygon):
        transformed_polygons = [transform(lambda x, y: to_3d(x, y), poly) for poly in geometry.geoms]
        return MultiPolygon(transformed_polygons)

    return None

def set_wall_thickness(pand_data):
    """assign a wall thickness, based on the building year and amount of storeys"""
    def get_thickness(row):
        year = row['year']
        storeys = row['storeys']

        if year < 1970:
            if storeys <= 5:
                return 29, 12
            elif storeys <= 10:
                return 38, 11
            else:
                return 25, 9
        elif 1970 <= year <= 1985:
            if storeys <= 5:
                return 28, 11
            elif storeys <= 10:
                return 26, 11
            else:
                return 29, 12
        else:  # year > 1985
            if storeys <= 5:
                return 30, 12
            elif storeys <= 10:
                return 38, 13
            else:
                return 35, 15

    pand_data[['thickness_ext', 'thickness_shared']] = pand_data.apply(get_thickness, axis=1, result_type='expand')

def extract_multilines(geom):
    """ Extract exterior and interior geometries """
    lines = []

    if isinstance(geom, Polygon):
        lines.append(LineString(geom.exterior.coords))
        lines.extend(LineString(ring.coords) for ring in geom.interiors)

    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            lines.append(LineString(poly.exterior.coords))
            lines.extend(LineString(ring.coords) for ring in poly.interiors)

    return MultiLineString(lines)

def plot_geometries(ref_floor_geom, transformed_geometries, buffer_distance=0.1):
    """Plots the reference floor geometry, its buffered version, and the best transformed geometries."""
    fig, ax = plt.subplots(figsize=(8, 8))

    if isinstance(ref_floor_geom, gp.GeoSeries):
        geometries = ref_floor_geom.geometry
    else:
        geometries = [ref_floor_geom]

    all_lines = []
    for geom in geometries:
        if isinstance(geom, Polygon):
            all_lines.append(extract_multilines(geom))
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                all_lines.append(extract_multilines(poly))

    buffered_lines = [line.buffer(buffer_distance) for multi_line in all_lines for line in multi_line.geoms]

    # Track which legend items have already been added
    added_labels = set()

    for geom in geometries:
        label = "Reference Floor" if "Reference Floor" not in added_labels else None
        if isinstance(geom, Polygon):
            plot_polygon(geom, ax=ax, facecolor='none', add_points=False, edgecolor='blue', linewidth=1, label=label)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                plot_polygon(poly, ax=ax, facecolor='none', add_points=False, edgecolor='blue', linewidth=1, label=label)
        added_labels.add("Reference Floor")

    for buffered_line in buffered_lines:
        label = "Buffered Lines" if "Buffered Lines" not in added_labels else None
        plot_polygon(buffered_line, ax=ax, facecolor='none', add_points=False, edgecolor='blue',
                     linestyle='dashed', linewidth=1.5, label=label)
        added_labels.add("Buffered Lines")

    for transformed_geom in transformed_geometries:
        label = "Current Floor" if "Current Floor" not in added_labels else None
        plot_polygon(transformed_geom, ax=ax, facecolor='none', edgecolor='red', add_points=False, linewidth=1, label=label)
        added_labels.add("Current Floor")
    ax.set_axis_off()
    ax.set_title("Shape Similarity")
    ax.legend()
    plt.show()


def containment_percentage(geom, ref_geom):
    """ calculate the percentage (0,1) of the geometry that is contained within the reference """
    if geom.is_empty or geom.area == 0:
        return 0.0

    intersection = geom.intersection(ref_geom)
    contained_area = intersection.area
    total_area = geom.area

    return (contained_area / total_area)

def shape_similarity_score(ref_geom, geom, buffer_distance=0.2):
    """ calculate the shape similarity score based on the percentage (0, 1) of the floor boundary
    that falls within a buffered version of the reference floor boundary. """

    if isinstance(ref_geom, gp.GeoSeries):
        ref_geometries = ref_geom.geometry
    else:
        ref_geometries = [ref_geom]

    # extract lines instead of polygons
    all_ref_lines = []
    for rgeom in ref_geometries:
        if isinstance(rgeom, Polygon):
            all_ref_lines.append(extract_multilines(rgeom))
        elif isinstance(rgeom, MultiPolygon):
            for poly in rgeom.geoms:
                all_ref_lines.append(extract_multilines(poly))

    buffered_ref_union = unary_union(
        [line.buffer(buffer_distance) for multi_line in all_ref_lines for line in multi_line.geoms])

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

    # intersection length with the merged buffered reference
    intersected_length = sum(line.intersection(buffered_ref_union).length
                             for multi_line in all_geom_lines
                             for line in multi_line.geoms)

    if total_length == 0:
        return 0.0

    return min(1.0, max(0.0, intersected_length / total_length))

def translate_polygon(geometries, translation_vector):
    if isinstance(translation_vector[0], pd.Series):
        dx, dy = translation_vector[0].iloc[0], translation_vector[1].iloc[0]
    else:
        dx, dy = translation_vector
    return geometries.apply(lambda geom: translate(geom, xoff=dx, yoff=dy))

def goodness_of_fit(polygon, reference):
    """calculate how well two shapes overlap as the proportion of the predicted shape that overlaps with the reference """
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
    """ratio of the overlapping area to the total area covered by both shapes"""
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
    """"greatest distance between any point in the polygon and the closest point in the reference"""
    if isinstance(polygon, (Polygon, MultiPolygon)) and isinstance(reference, (Polygon, MultiPolygon)):
        return hausdorff_distance(polygon, reference)
    return float('inf')

def snap_floors_to_reference(best_geometries, below_floor_geom, threshold=0.2, simplification_tolerance=0.1):
    """snaps the floor geometry to the reference floor (either the ground floor, or the floor below the current floor) using Shapely's snap function, after simplifying."""

    if isinstance(below_floor_geom, Polygon):
        below_floor_geom = MultiPolygon([below_floor_geom])

    # the polygons are simplfied to remove any unnecessary vertices which could impact the snapping
    simplified_below = below_floor_geom.simplify(simplification_tolerance, preserve_topology=True)
    simplified_best = best_geometries.simplify(simplification_tolerance, preserve_topology=True)

    # the union of the floor below, to ensure we only snap the outside boundary, not the interior
    below_union = unary_union(simplified_below)

    # snap the current floor to the reference floor geometry
    snapped_floors = snap(simplified_best, below_union, threshold)

    return snapped_floors

def extract_boundary_points(geometry):
    """extracts the boundary for polygons and multipolygons"""
    if isinstance(geometry, Polygon):
        return np.array(geometry.exterior.coords)
    elif isinstance(geometry, MultiPolygon):
        return np.vstack([np.array(poly.exterior.coords) for poly in geometry.geoms])
    return np.array([])

def averaged_hausdorff_distance(polygon, reference):
    """computes the averaged Hausdorff distance between the polygon and its reference."""
    if not isinstance(polygon, (Polygon, MultiPolygon)) or not isinstance(reference, (Polygon, MultiPolygon)):
        return float('inf')

    # extract boundary points
    polygon_coords = extract_boundary_points(polygon)
    reference_coords = extract_boundary_points(reference)

    if len(polygon_coords) == 0 or len(reference_coords) == 0:
        return float('inf')

    # forward distances
    forward_distances = [reference.distance(Point(p)) for p in polygon_coords]

    # backward distances
    backward_distances = [polygon.distance(Point(r)) for r in reference_coords]

    # compute the averaged Hausdorff distance
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
                translation_step=0.2, snap=False):
    """
    Performs a grid search over translations to optimize shape similarity,
    returning the individual room geometries with the applied transformations.
    """
    ground_floor_geom = gp.GeoSeries(ground_floor_geom)
    boundary_points = extract_boundary_points(ground_floor_geom.unary_union)
    # the max_translation is for the x direction
    max_translation = np.max(pdist(boundary_points)) / 4
    # restricted in y
    max_translation_y = 0.5

    if isinstance(floor_geom, Polygon):
        floor_multipolygon = MultiPolygon([floor_geom])
    else:
        floor_polygons = [geom for geom in floor_geom if geom.is_valid]
        floor_multipolygon = MultiPolygon(floor_polygons)

    if isinstance(ground_floor_geom, Polygon):
        ground_floor_multipolygon = MultiPolygon([ground_floor_geom])
    else:
        ground_floor_polygons = [geom for geom in ground_floor_geom if geom.is_valid]
        ground_floor_multipolygon = MultiPolygon(ground_floor_polygons)

    translations_x = np.arange(-max_translation, max_translation + translation_step,
                             translation_step)
    translations_y = np.arange(-max_translation_y, max_translation_y + translation_step,
                             translation_step)

    transform_params = list(itertools.product(translations_x, translations_y))

    def apply_transformations(dx, dy):
        # perform the translation on the multipolygon
        transformed_geometries = translate(floor_multipolygon, xoff=dx, yoff=dy)

        # the shape similarity score for the transformation
        score_sim = shape_similarity_score(ground_floor_multipolygon, transformed_geometries)
        score = score_sim
        return transformed_geometries, score, (score_sim, (dx, dy))

    # grid search in parallel
    total_combinations = len(transform_params)
    results = Parallel(n_jobs=-1)(
        delayed(apply_transformations)(dx, dy)
        for dx, dy in tqdm(transform_params, total=total_combinations, desc="Room Grid Search Progress")
    )

    best_score = -np.inf
    best_geometries = None
    best_params = None
    for transformed_geometries, score, params in results:
        if transformed_geometries is not None and score > best_score:
            best_score = score
            best_geometries = transformed_geometries
            best_params = params

    transformed_rooms = []
    for room_geom in floor_geom:
        transformed_room = translate(room_geom, xoff=best_params[1][0], yoff=best_params[1][1])
        transformed_rooms.append(transformed_room)

    # optionally snap to the floor below
    if snap:
        transformed_rooms = snap_floors_to_reference(
            gp.GeoSeries(transformed_rooms),
            ground_floor_geom
        )
        # convert back to list if needed elsewhere
        transformed_rooms = list(transformed_rooms)

    return gp.GeoSeries(transformed_rooms)

def grid_search(pand, aligned_column, bgt_outline, pand_data, room_geom,
                alpha,
                buffer=0.5,
                angle_step=1,
                scale_step=0.05, scale_range=(0.8, 1.2),
                translation_step=0.2):
    """
    Performs a grid search over rotation angles, scales, and translations to optimize Goodness of Fit (GoF) and hausdorff distance.
    """

    if isinstance(pand[aligned_column].iloc[0], Polygon) and isinstance(pand[bgt_outline].iloc[0], MultiPolygon):
        alpha = 0.75


    # only apply translation when its multipolygons, this makes the result more precise but it also takes longer
    if isinstance(pand[aligned_column].iloc[0], MultiPolygon) or isinstance(pand[bgt_outline].iloc[0],
                                                                            MultiPolygon):
        apply_translation = True
    else:
        apply_translation = False


    # compute max_translation dynamically based on the bgt_outline geometry
    bgt_geom = pand[bgt_outline].iloc[0]
    boundary_points = extract_boundary_points(bgt_geom)
    # by putting a larger number than 4, the search range becomes smaller, making it faster but with less coverage
    max_translation = np.max(pdist(boundary_points))/4

    angles = np.arange(-180, 180, angle_step)
    scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
    translations = np.arange(-max_translation, max_translation + translation_step,
                             translation_step) if apply_translation else np.array([0])

    transform_params = list(itertools.product(scales, angles, translations, translations))

    def apply_transformations(scale_factor, angle, dx, dy):
        transformed_geometries = np.array(pand[aligned_column])
        transformed_geometries = np.array([scale(g, xfact=scale_factor, yfact=scale_factor) for g in transformed_geometries])
        transformed_geometries = np.array([rotate(g, angle, origin='centroid') for g in transformed_geometries])
        if apply_translation:
            transformed_geometries = np.array([translate(g, xoff=dx, yoff=dy) for g in transformed_geometries])

        # compute scores in parallel
        hausdorff_values = np.array(Parallel(n_jobs=-1)(delayed(averaged_hausdorff_distance)(g, pand[bgt_outline].iloc[0]) for g in transformed_geometries))
        mean_hausdorff = np.mean(hausdorff_values)
        gof_scores = np.array(Parallel(n_jobs=-1)(delayed(goodness_of_fit)(g, pand[bgt_outline].iloc[0]) for g in transformed_geometries))
        mean_gof = np.mean(gof_scores)
        combscore, score_gof, score_haus = combined_score(mean_gof, mean_hausdorff, alpha)

        return transformed_geometries, combscore, (score_gof, score_haus, scale_factor, angle, (dx, dy))

    # run grid search in parallel
    total_combinations = len(transform_params)
    results = Parallel(n_jobs=-1)(
        delayed(apply_transformations)(s, a, dx, dy) for s, a, dx, dy in tqdm(transform_params, total=total_combinations, desc="Grid Search Progress")
    )

    # find the best transformation
    best_score = -np.inf
    best_geometries = None
    best_params = None
    pand_data['transformed_akte_bg'] = None
    for transformed_geometries, score, params in results:
        if  score > best_score:
            for value in transformed_geometries:
                containment = containment_percentage(pand[bgt_outline].iloc[0], value)
                shapesim = shape_similarity_score(pand[bgt_outline].iloc[0], value)
                if containment > 0.85 and shapesim > 0.85:
                    pand_data['transformed_akte_bg'] = value
                    best_score = score
                    best_geometries = transformed_geometries
                    best_params = params
                    no_geom = False
                    # pand_data.set_geometry(pand_data['transformed_akte_bg'])
                    # pand_data.plot()
                    # plt.show()
                else:

                    best_score = score
                    best_geometries = transformed_geometries
                    # pand_data.set_geometry(pand_data['transformed_akte_bg'])
                    # pand_data.plot()
                    # plt.show()
                    best_params = params
                    no_geom = True



    # allow user to retry with custom parameters if no good match found
    if no_geom and manual:
        print("No transformation met the containment and shape similarity thresholds.")
        user_choice = input("Do you want to enter custom parameters? (y/n): ").strip().lower()

        if user_choice == 'y':
            custom_scale = float(input(f"Enter custom scale factor (now {round(best_params[2], 1)}): "))
            custom_angle = float(input(f"Enter custom rotation angle degrees (now {round(best_params[3], 1)}): "))
            custom_dx = float(input(f"Enter custom translation in x meters (now {round(best_params[4][0], 1)}): "))
            custom_dy = float(input(f"Enter custom translation in y meters (now {round(best_params[4][1], 1)}): "))
            newtransformed_geometries, newscore, newparams = apply_transformations(custom_scale, custom_angle,
                                                                          custom_dx,
                                                                          custom_dy)
            # could use better interaction and visualization of changes
            best_geometries = newtransformed_geometries
            best_params = newparams


    common_centroid = pand.aligned_geometry.centroid.iloc[0]
    transformed_rooms = list(pand_data[room_geom])

    # apply scale, rotation, and translation
    if best_params is not None:
        transformed_rooms = np.array(
            [scale(g, xfact=best_params[2], yfact=best_params[2], origin=common_centroid)
             for g in transformed_rooms]
        )

        transformed_rooms = np.array([
            rotate(g, best_params[3], origin=common_centroid) if idx != -100 else g
            for g, idx in zip(transformed_rooms, pand_data['floor_index'])
        ])

    if apply_translation and best_params is not None:
        transformed_rooms = np.array(
            [translate(g, xoff=best_params[4][0], yoff=best_params[4][1]) for g in transformed_rooms])

    # store results in DataFrame
    pand_data['optimized_rooms'] = transformed_rooms
    pand["optimized_geometry"] = best_geometries

    pand.reset_index(drop=True, inplace=True)
    pand_data.reset_index(drop=True, inplace=True)

    pand_data = pand_data.merge(pand[['bag_pnd', 'geom_akte_bg', 'bgt_outline']], on='bag_pnd', how='left')

    pand_data["optimized_rooms"] = transformed_rooms

    # edge matching could be used here
    # if transformed_rooms is not None and pand_data['transformed_akte_bg'].iloc[0] is not None and pand_data['floor_index'].all() != -100:
    #     transformed_rooms = [refine_alignment(pand_data["bgt_outline"].iloc[i], pand_data['transformed_akte_bg'].iloc[i], g)
    #                        for i, g in enumerate(transformed_rooms)]
    pand_data["georef_accuracy"] = (containment + shapesim)/2 * 100
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
    if gof is None or hausdorff is None or np.isnan(gof) or np.isnan(hausdorff):
        return 0.0, 0.0, 0.0
    hausdorff_normalized = 1 / (1 + hausdorff)
    combined_score = alpha * gof + (1 - alpha) * hausdorff_normalized
    return combined_score, gof, hausdorff_normalized

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
    """find edge anchor points"""
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
    """refine alignment by finding best anchor edges and transforming to those (edge matching)"""
    reference_geom = remove_collinear_vertices(reference_geom)
    aligned_geom = remove_collinear_vertices(aligned_geom)

    ref_edges = compute_edges_with_angles(reference_geom)
    aligned_edges = compute_edges_with_angles(aligned_geom)

    anchors = find_best_edge_anchors(ref_edges, aligned_edges, distance_threshold, angle_threshold)

    if anchors is None:
        return aligned_geom

    matched_aligned, matched_ref = anchors

    transform =  rigid_transform_polygon(aligned_geom, matched_aligned, matched_ref)
    if geom_rooms.geom_type == "Polygon":
        return transform_polygon(geom_rooms, transform)

    elif geom_rooms.geom_type == "MultiPolygon":
        transformed_polygons = [transform_polygon(poly, transform) for poly in geom_rooms.geoms]
        return MultiPolygon([poly for poly in transformed_polygons if poly.is_valid])


def fetch_3dbag_data(bag_id):
    """fetch data from 3DBAG API using the BAG identificatie."""
    formatted_id = f"NL.IMBAG.Pand.{bag_id}"
    url = f"https://api.3dbag.nl/collections/pand/items/{formatted_id}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {bag_id}. Status Code: {response.status_code}")
        return None

def extract_all_scale_texts(data):
    scale_factors = []

    def find_scale_in_text(value):
        """Recursively search for scale text in strings within nested dicts/lists."""
        if isinstance(value, str):  # Process only string values
            match = re.search(r'1\s*:\s*(\d+)', value)
            if match:
                print("Match found:", match.group(0))
                scale_factor = match.group(1)  # Assuming get_scale_text() exists
                scale_factors.append(scale_factor)
            else:
                scale_factors.append(999)
        elif isinstance(value, dict):  # If dict, process values only (ignore keys)
            for v in value.values():  # Process values, avoiding dicts as keys
                find_scale_in_text(v)
        elif isinstance(value, list):  # If list, process each item
            for item in value:
                find_scale_in_text(item)

    find_scale_in_text(data)  # Start processing
    return scale_factors


def get_scale_text(text):
    """
    Get the scale from the text in the deed
    :param text:
    :return: scale
    """
    match = re.search(r'1\s*:\s*(\d+)', text)
    if match:
        print("verdieping match", int(match.group(1)))
        return int(match.group(1))
    else:
        return 999

def rotate_geom_arrow(row):
    perceel_id = row.get('perceel_id', None)
    angle = rotation_angles.get(perceel_id, 0)
    geometry_columns = [col for col in row.index if 'geom_akte_bg' in col]

    for col in geometry_columns:
        geometry = row[col]
        row[col] = rotate(geometry, angle, origin='centroid')
    return row

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

            for col in geometry_columns:
                geometry_pol = row[col]
                group.at[index, col] = rotate(geometry_pol, angle, origin='centroid')

    return group

def has_3d_coords(geom):
    """Returns True if geometry is a Polygon or MultiPolygon with all 3D coords."""
    if geom is None or geom.is_empty:
        return False
    if isinstance(geom, (Polygon, MultiPolygon)):
        # Check if exterior coordinates are 3D
        coords = list(geom.exterior.coords)
        return all(len(coord) == 3 for coord in coords)
    if isinstance(geom, list):  # if already a list of faces with (x, y, z)
        return all(len(pt) == 3 for face in geom for pt in face)
    return False

def new_uuid() -> str:
    """Return a compact UUID string (hex, no dashes)."""
    return uuid.uuid4().hex

def convert_to_json_serializable(obj: Any):
    """Recursively cast numpy dtypes → native Python so json.dump() works."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def remove_consecutive_duplicates(coords):
    return [pt for i, pt in enumerate(coords) if i == 0 or pt != coords[i - 1]]

def build_cityjson_model(rooms_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Return a CityJSON dict:

        BIMLegalApartmentComplex   → CityObjectGroup
        ApartmentComplex           → Building
        ApartmentUnit              → BuildingPart
        SharedPart                 → Whether the room is a shared part or private
        BIMLegalSpaceUnit          → Room (leaf) with geometry
    """


    cj: Dict[str, Any] = {
        "type": "CityJSON",
        "version": "2.0",
        "CityObjects": {},
        "vertices": [],
        "metadata": {
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/28992"
        },
    }


    vertices: list[list[float]] = []
    vertex_index: Dict[tuple, int] = {}

    def add_vertex(coord):
        # scale vertices to be integers
        scaled = tuple(int(round(c * 10000)) for c in coord)
        if scaled not in vertex_index:
            idx = len(vertices)
            vertex_index[scaled] = idx
            vertices.append(list(scaled))
        return vertex_index[scaled]


    group_uuid = new_uuid()
    cj["CityObjects"][group_uuid] = {
        "type": "CityObjectGroup",
        "attributes": {
            "name": "BIMLegalApartmentComplex"
        },
        "children": []
    }

    building_uuid = new_uuid()
    cj["CityObjects"][building_uuid] = {
        "type": "Building",
        "attributes": {
            # only one apartment per bim legal model
            "apartmentComplexIndex": "AC-01"
        },
        "children": [],
        "parents": [group_uuid]
    }
    cj["CityObjects"][group_uuid]["children"].append(building_uuid)


    rooms_gdf = rooms_gdf.copy()
    rooms_gdf["apartment"] = pd.to_numeric(rooms_gdf["appartement"], errors="coerce")


    apartment_units: Dict[int, Dict[str, Any]] = {}
    for idx in rooms_gdf["apartment"].dropna().astype(int).unique():
        unit_uuid = new_uuid()
        apartment_units[idx] = {
            "uuid": unit_uuid,
            "children": []
        }
        cj["CityObjects"][unit_uuid] = {
            "type": "BuildingPart",
            "attributes": {
                "apartmentIndex": str(idx)
            },
            "children": apartment_units[idx]["children"],
            "parents": [building_uuid]
        }
        cj["CityObjects"][building_uuid]["children"].append(unit_uuid)

    shared_room_container_uuid = new_uuid()
    cj["CityObjects"][shared_room_container_uuid] = {
        "type": "BuildingPart",
        "attributes": {
            "name": "SharedPart",
            "isSharedPart": True
        },
        "children": [],
        "parents": [building_uuid]

    }

    cj["CityObjects"][building_uuid]["children"].append(shared_room_container_uuid)

    for idx, row in rooms_gdf.iterrows():
        geom = row["optimized_rooms_3d"]
        if geom is None or geom.is_empty:
            continue

        room_uuid = new_uuid()
        solid_faces: list[list[int]] = []

        # create room geometry
        if hasattr(geom, "exterior"):
            coords_base = remove_consecutive_duplicates(list(geom.exterior.coords))

            # unclosed
            if coords_base[0] == coords_base[-1]:
                coords_base = coords_base[:-1]

            coords_top = [(x, y, z + row["extrusion_height"]) for x, y, z in coords_base]

            # reverse so its counterclockwise
            bottom_face = [add_vertex(c) for c in reversed(coords_base)]
            top_face = [add_vertex(c) for c in coords_top]

            # create wall by iterating over the base and top
            wall_faces = []
            for i in range(len(coords_base)):
                p1_base = coords_base[i]
                p2_base = coords_base[(i + 1) % len(coords_base)]
                p1_top = coords_top[i]
                p2_top = coords_top[(i + 1) % len(coords_top)]
                wall_faces.append([add_vertex(p1_base), add_vertex(p2_base),
                                   add_vertex(p2_top), add_vertex(p1_top)])

            solid_faces = [bottom_face, top_face] + wall_faces

        elif isinstance(geom, list):
            for face in geom:
                solid_faces.append([add_vertex(tuple(face)) for face in face])
        else:
            raise TypeError(f"Unsupported geometry type for row {idx}: {type(geom)}")

        # add rooms
        cj["CityObjects"][room_uuid] = {
            "type": "BuildingRoom",
            "attributes": {
                "name": row["ruimte"],
                "extrusion_height": row.get("extrusion_height"),
                "isSharedPart": pd.isna(row["apartment"]),
                "apartmentindex": str(row["apartment"]),
                "bimLegalSpaceUnitType": "m",
                "level" : row.get("floor_index"),
                "georef_accuracy": row.get("georef_accuracy"),
            },
            "geometry": [{
                "type": "MultiSurface",
                "lod": "1",
                "boundaries": [ [face] for face in solid_faces ]
            }]
        }

        # attach to parents
        if pd.notna(row["apartment"]):
            parent_dict = apartment_units[int(row["apartment"])]
        else:
            parent_dict = cj["CityObjects"][shared_room_container_uuid]

        parent_dict["children"].append(room_uuid)

        if pd.notna(row["apartment"]):
            parent_uuid = apartment_units[int(row["apartment"])]["uuid"]
        else:
            parent_uuid = shared_room_container_uuid
        cj["CityObjects"][room_uuid]["parents"] = [parent_uuid]

    if not cj["CityObjects"][shared_room_container_uuid]["children"]:
        cj["CityObjects"].pop(shared_room_container_uuid)
        cj["CityObjects"][building_uuid]["children"].remove(shared_room_container_uuid)


    cj["vertices"] = vertices
    cj["transform"] = {
        "scale": [0.0001, 0.0001, 0.0001],
        "translate": [0.0, 0.0, 0.0]
    }

    return cj


def export_to_cityjson(rooms_gdf: gpd.GeoDataFrame, out_path: Path | str) -> None:
    cj = build_cityjson_model(rooms_gdf)
    cj = convert_to_json_serializable(cj)
    out_path = Path(out_path)
    for i, v in enumerate(cj["vertices"]):
        if len(v) != 3:
            print(f"Invalid vertex at index {i}: {v}")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cj, f, indent=2)
    print(f"CityJSON written ➜ {out_path.resolve()}")


#========================================== START CODE ================================================================
# start code
BASE_DIR = Path(__file__).resolve().parent
# provide the paths to the files
# percelen (instead of a shapefile, the BRK API can also be used)
kad_path = BASE_DIR / "data" / "raw" / "Percelen_aktes_Hilversum.shp"
# the json input folder
json_path = BASE_DIR / "data" / "json_input"
# the output folder
out_path = BASE_DIR / "output"

files = os.listdir(json_path)
if not os.path.exists(out_path):
    os.mkdir(out_path)

start_time = time.time()
# options: text, area (best option)
scale_version = 'area'
# options: azimuth, arrow or none (best option if no arrows)
rotate_version = 'none'
# options: bbox, centroid (best option)
translation_version = 'centroid'
rotation_angles2 = [171.3,  180,  -43.5,  78.9,  6.8,  0.0,  121.6, 120,  22.2,7]
rotation_angles = {'HVS00N1878': 171.3, "HVS00N1882": 180, "HVS00N2359": -43.5, "HVS00N2643": 78.9, "HVS00N2848": 6.8, "HVS00N3211": 0.0, "HVS00N3723": 121.6, "HVS00N4216": 120, "HVS00N555": 22.2, "HVS00N9252":7}
epsilon_value = 1.4
angles_list = []
# cap height values automatically to realistic values, or do this manually if they're off standard values
automatic_capping = True
# keep all prompts to manually adjust the georeferencing parameters if they're off
manual = False
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

    # if there are multiple parcels (parts[4] and beyond)
    if len(parts) > 4:
        perceel_id_full = str(parts[1]) + str(parts[2]) + ''.join(parts[3:])
        perceel_ids = []
        all_bgt_lokaal_ids = []
        all_bag_pnds = []

        # store percelen geometries to compute the total bbox
        perceel_geometries = []

        for i in range(3, len(parts), 1):
            perceel_id = str(parts[1]) + str(parts[2]) + str(parts[i])
            perceel_ids.append(perceel_id)

            selection_perceel = kadpercelen[
                (kadpercelen.KAD_GEM == parts[1]) &
                (kadpercelen.SECTIE == parts[2]) &
                (kadpercelen.PERCEELNUM == int(parts[i]))
                ]

            perceel_geometries.append(selection_perceel.geometry.iloc[0])

        # compute total bounding box from all selected percelen
        total_bounds = gp.GeoSeries(perceel_geometries).total_bounds
        bbox = f'{int(total_bounds[0])},{int(total_bounds[1])},{int(total_bounds[2])},{int(total_bounds[3])}'


        # make a single API request for the entire bbox
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

        perceel_union = gp.GeoSeries(perceel_geometries).unary_union
        selection_perceel = gp.GeoDataFrame({'geometry': [perceel_union]}, crs=28992)

        # intersect with the actual geometry (instead of bounding box)
        join_bgt = gp.sjoin(bgt_geom_all_temp, selection_perceel, how='inner', predicate='intersects')
        join_bgt["overlap_area"] = join_bgt.geometry.intersection(perceel_union).area

        # filter those with not a lot of overlap
        join_bgt = join_bgt[join_bgt["overlap_area"] >= 10]

        # union only the filtered geometries
        if not join_bgt.empty:
            union_geom = join_bgt.unary_union

            bgt_geom_all = gp.GeoDataFrame(
                [{'perceel_id': perceel_id_full,
                  'bgt_lokaal_id': ', '.join(set(all_bgt_lokaal_ids)),
                  'bag_pnd': ', '.join(set(all_bag_pnds)),
                  'geometry': union_geom}],
                geometry='geometry', crs=28992)

        perceel_bgt = bgt_geom_all.set_crs('epsg:28992', allow_override=True)
        perceel_bgt = perceel_bgt[['bgt_lokaal_id', 'bag_pnd', 'geometry', 'perceel_id']]


    # if theres only one perceel
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

        bgt_geom_all_temp = gp.GeoDataFrame(bgt_data, geometry='geometry', crs=28992)

        # now intersect with the actual geometry bc otherwise its too many polygons
        join_bgt = gp.sjoin(bgt_geom_all_temp, selection_perceel, how='inner', predicate='intersects')
        join_bgt["overlap_area"] = join_bgt.geometry.intersection(selection_perceel.geometry.iloc[0]).area
        join_bgt = join_bgt[join_bgt["overlap_area"] >= 2]

        perceel_bgt = join_bgt.set_crs('epsg:28992', allow_override=True)
        if 'perceel_id_left' in join_bgt.columns and 'perceel_id_right' in join_bgt.columns:
            if (join_bgt['perceel_id_left'] == join_bgt['perceel_id_right']).all():
                perceel_bgt = join_bgt.drop(columns=['perceel_id_right']).rename(columns={'perceel_id_left': 'perceel_id'})

        perceel_bgt = perceel_bgt[['bgt_lokaal_id', 'bag_pnd', 'geometry', 'perceel_id']]


    # now get the rooms data from the json
    with open(os.path.join(json_path, f'{perceel}.latest.json')) as f:
        data = json.load(f)

    if scale_version == "text":
        text = "\n".join(f"{key}: {value}" for key, value in data.items())
        scale_factors = extract_all_scale_texts(text)
        if scale_factors:
            scale_factor_whole_file = statistics.mode(scale_factors)
        else:
            scale_factor_whole_file = 999


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

        # initialize scale
        if scale_version == 'text':
            scale_text = verdiepingsaanduiding[-1]
            scale_factor = get_scale_text(scale_text)
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

    pand_data_copy = pand_data.copy()
    # create a dataframe for the panden with the polygon outline of the BG
    rooms_bg = pand_data[pand_data['verdieping'].fillna('').str.lower().str.contains("begane grond")]


    rooms_bg['geom_akte_all_copy'] = rooms_bg["geom_akte_all"]
    pand_outline = rooms_bg.groupby('bgt_lokaal_id').agg({
        'geom_akte_all_copy': lambda g: g.unary_union, 'perceel_id': 'first', 'bag_pnd': 'first', 'geom_bgt': 'first', 'geom_akte_all': 'first' })
    pand = gp.GeoDataFrame(pand_outline,geometry='geom_akte_all_copy',crs=28992)
    pand.rename(columns={'geom_akte_all_copy': 'geom_akte_bg'}, inplace=True)

    # join the dataframes
    combined_df = pand.merge(pand_data, left_on='bgt_lokaal_id', right_on='bgt_lokaal_id', how='inner')
    combined_df.to_csv(os.path.join("werkmap", f'combined_df.csv'), index=False)

    pand_data.drop(columns=['geom_bgt'], inplace=True)

    outline = pand.groupby('perceel_id').agg({'geom_bgt': lambda g: g.unary_union})
    bgt_outline = gp.GeoDataFrame(outline, geometry='geom_bgt', crs=28992)
    bgt_outline = bgt_outline.rename_geometry('bgt_outline')
    pand = pand.merge(bgt_outline, on='perceel_id', how='left')

    pand = gp.GeoDataFrame(pand, geometry='geom_akte_bg')

    # ========================================== INITIALISING =========================================================
    if rotate_version == 'arrow':
        pand = pand.apply(rotate_geom_arrow, axis=1)
    if rotate_version == 'azimuth':
        pand_grouped = pand.groupby('perceel_id').apply(rotate_geom_azimuth)
        pand = pand.merge(pand_grouped, on='bag_pnd', how='left')
        pand = pand.loc[:, ~pand.columns.str.endswith('_x')]
        pand.columns = pand.columns.str.replace('_y', '', regex=False)

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


    if isinstance(scale_factor, pd.Series):
        scale_factor_value = scale_factor.iloc[0]
    else:
        scale_factor_value = scale_factor

    # scale the ground floor
    pand['geom_akte_bg_scaled'] = pand['geom_akte_bg'].apply(
        lambda g: scale(g, xfact=scale_factor_value, yfact=scale_factor_value, origin='centroid'))
    pand.set_geometry('geom_akte_bg_scaled', inplace=True)

    centroid_bg = pand['geom_akte_bg'].centroid.iloc[0]
    # scale all the rooms
    pand_data['geom_akte_all_scaled'] = pand_data['geom_akte_all'].apply(
        lambda g: scale(g, xfact=scale_factor_value, yfact=scale_factor_value, origin=centroid_bg))

    pand_data.set_geometry('geom_akte_all_scaled', inplace=True)




    # align ground floor -> translation
    if translation_version == "centroid":
        pand.set_geometry('geom_akte_bg_scaled', inplace=True)
        bgt_centroid = unary_union(pand.bgt_outline).centroid
        building_centroid = pand.geometry.unary_union.centroid

        translation_vector = (bgt_centroid.x - building_centroid.x,
                              bgt_centroid.y - building_centroid.y)



    if translation_version == 'bbox':
        if len(pand.bgt_outline.bounds) > 1:
            bgt_bbox = pand.bgt_outline.bounds.iloc[0]
        else:
            bgt_bbox = pand.bgt_outline.bounds
        if len(pand.geometry.bounds)> 1:
            building_bbox = pand.geometry.bounds.iloc[0]
        else:
            building_bbox = pand.geometry.bounds

        translation_vector = (bgt_bbox.minx - building_bbox.minx,
                              bgt_bbox.miny - building_bbox.miny)


    pand.set_geometry('geom_akte_bg_scaled')
    pand['aligned_geometry'] = translate_polygon(pand.geometry, translation_vector)
    pand.set_geometry('aligned_geometry')

    # translate the rooms relative to the ground floor, so to the world coordinates
    pand_data = gp.GeoDataFrame(pand_data, geometry='geom_akte_all', crs='EPSG:28992')
    pand_data.set_geometry('geom_akte_all_scaled', inplace=True)
    floors = pand_data.groupby('verdieping')

    pand_data['aligned_rooms'] = None

    for floor, floor_data in floors:
        floor_centroid = floor_data.geometry.unary_union.centroid
        relative_x_offset = floor_centroid.x - building_centroid.x
        relative_y_offset = floor_centroid.y - building_centroid.y
        translation_vector = (bgt_centroid.x - floor_centroid.x + relative_x_offset,
                              bgt_centroid.y - floor_centroid.y + relative_y_offset)

        pand_data.loc[floor_data.index, 'aligned_rooms'] = floor_data.geometry.apply(
            lambda geom: translate(geom, xoff=translation_vector[0], yoff=translation_vector[1])
        )

    pand_data = gp.GeoDataFrame(pand_data, geometry='aligned_rooms', crs='EPSG:28992')

# ======================================== FLOOR ALIGNMENT =========================================================
    # then the separate floors are shifted on top of each other, in the x direction
    pand_data['floor_index'] = pand_data['verdieping'].apply(map_floor)
    pand_data = pand_data[pand_data['floor_index'] != -999]

    pand_data.set_geometry('aligned_rooms')

    # the reference X-centroid from the ground floor (index 0)
    ground_floor = pand_data[pand_data['floor_index'] == 0]
    reference_x = ground_floor.geometry.unary_union.centroid.x


    # shift each floor so that its centroid.x matches the ground floor centroid.x
    for floor_index, floor_data in pand_data.groupby('floor_index'):
        floor_x_centroid = floor_data.geometry.unary_union.centroid.x
        x_shift = reference_x - floor_x_centroid
        pand_data.loc[floor_data.index, 'aligned_rooms'] = floor_data.geometry.apply(
            lambda geom: translate(geom, xoff=x_shift, yoff=0)
        )


    # align each floor to the floor below
    floors = pand_data.groupby('floor_index')

    optimized_geometries = {}
    # exlude the ones that dont have a recognisable number
    floors = floors.filter(lambda x: x.name != -100).groupby('floor_index')
    for floor_index, floor_data in floors:
        # for basements, align above
        if floor_index <= -1:
            below_floor_data = floors.get_group(floor_index + 1)
            # ground floor aligns to itself
        elif floor_index == 0:
            below_floor_data = floors.get_group(floor_index)
        # else its a normal floor which aligns to a floor below
        else:
            below_floor_data = floors.get_group(floor_index - 1)

        if floor_index - 1 in optimized_geometries and floor_index != 0:
            below_outline = optimized_geometries[floor_index - 1]
            if isinstance(below_outline, MultiPolygon):
                below_outline = gp.GeoSeries([poly for poly in below_outline.geoms])
            elif isinstance(below_outline, list):
                below_outline = gp.GeoSeries(below_outline)
        else:
            below_outline = below_floor_data.geometry

        floor_outline = floor_data.geometry

        # calculate the similarity score and best geometry
        similarity_score = shape_similarity_score(below_outline, floor_outline)
        best_geom = grid_search_room(floor_outline, below_outline, translation_step=0.3, snap=False)
        # plot_geometries(below_outline, best_geom)
        optimized_geometries[floor_index] = best_geom

    # assign the optimized geometries back to the original DataFrame
    for floor_index, best_geom in optimized_geometries.items():
        floor_mask = pand_data['floor_index'] == floor_index
        original_index = pand_data[floor_mask].index

        if isinstance(best_geom, gp.GeoSeries):
            geom_list = best_geom.tolist()
        else:
            raise ValueError(f"Expected GeoSeries, got {type(best_geom)} for floor {floor_index}")

        if len(original_index) != len(geom_list):
            raise ValueError(f"Mismatch in geometry count for floor {floor_index}: "
                             f"{len(original_index)} rows, {len(geom_list)} geometries")

        # assign geometries by index
        pand_data.loc[original_index, 'aligned_rooms'] = geom_list



    # ==========================================GEOREF OPTIMISATION ====================================================

    # grid search optimization
    pand_data = pand_data.dropna(subset=['aligned_rooms'])
    pand.drop_duplicates()
    pand_data.drop(columns=["bag_pnd", "bgt_lokaal_id"])
    pand_data.drop_duplicates()


    pand.set_geometry('aligned_geometry', inplace=True)
    pand = grid_search(pand, "aligned_geometry", "bgt_outline", pand_data, "aligned_rooms",
                       alpha=0.25, buffer=1,
                       angle_step=1, scale_step=0.05, scale_range=(0.8, 1.2),
                       translation_step=0.8)

    pand_data = gp.GeoDataFrame(pand_data, geometry='optimized_rooms', crs='EPSG:28992')
    # make a copy before removing the section geometry
    pand_data_section = pand_data.copy()
    pand_data = pand_data[pand_data['floor_index'] != -100]
    pand_data.set_geometry('optimized_rooms')

    # asign bag index to each room
    centroid_rooms = gp.GeoDataFrame(
        pand_data.copy(),
        geometry=pand_data['optimized_rooms'].centroid,
        crs=28992
    )

    # apatial join to find which centroid falls within which BAG geometry
    joined = gp.sjoin(
        centroid_rooms,
        bgt_geom_all_temp[['geometry', 'bag_pnd']],
        how='left',
        predicate='within'
    )

    joined = joined.drop(columns=['index_right'])
    joined = joined.drop_duplicates()
    pand_data = pand_data.drop_duplicates()
    pand_data['bag_pnd']= joined['bag_pnd_right'].values


    # plot
    # fig, ax = plt.subplots(figsize=(10, 10))
    #
    # # Plot original rooms
    # pand_data['optimized_rooms'].plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.5, label='Rooms')
    #
    # # Plot centroids
    # pand_data['optimized_rooms'].centroid.plot(ax=ax, color='red', markersize=10, label='Centroids')
    #
    # # Plot BGT polygons
    # bgt_geom_all_temp.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=0.5, label='BGT Pand')
    #
    # # Add legend
    # plt.legend()
    # plt.title("Rooms, Centroids, and BGT Pand Geometries")
    # plt.axis('equal')
    # plt.show()
    #


    #============================== Height Estimate ==========================================

    # the sections are identified as the geometry without a floor mapping text, which were mapped to index -100 previously
    null_verdieping_rooms = pand_data_section[pand_data_section['floor_index'] == -100]

    # get the 3dbag data
    bag_height_mapping = {}
    unique_bag_ids = set()
    for index, row in pand_data.iterrows():
        bag_ids = str(row["bag_pnd"]).split(",")
        for bag_id in bag_ids:
            bag_id = bag_id.strip()
            unique_bag_ids.add(bag_id)
    for bag_id in unique_bag_ids:
        data = fetch_3dbag_data(bag_id)

        if data is None:
            print(f"No data found for bag id: {bag_id}, using default height.")
            pand_data = pand_data[pand_data['bag_pnd'] != bag_id]
            # pand_data.loc[pand_data["bag_pnd"] == bag_id, "extrusion_height"] = 2.8
            # pand_data.loc[pand_data["bag_pnd"] == bag_id, "maaiveld"] = 0


        else:
            bag_info = data["feature"]["CityObjects"][f"NL.IMBAG.Pand.{bag_id}"]["attributes"]
            b3_h_dak_max = bag_info.get("b3_h_dak_70p")
            b3_h_maaiveld = bag_info.get("b3_h_maaiveld")
            total_pand_height = b3_h_dak_max - b3_h_maaiveld


            # calculate extrusion height
            total_floors = sum(1 for floor_index, _ in floors if floor_index >= 0)
            extrusion_height = total_pand_height / total_floors if total_floors > 0 else 2.8
            pand_data.loc[pand_data["bag_pnd"].str.contains(bag_id, na=False), "extrusion_height"] = extrusion_height
            pand_data.loc[pand_data["bag_pnd"].str.contains(bag_id, na=False), "maaiveld"] = b3_h_maaiveld

    if null_verdieping_rooms.empty:
        for floor_index, floor_data in floors:
            if floor_index < 0:
                pand_data.loc[pand_data["floor_index"] == floor_index, "extrusion_height"] = 2.3

    else:
        # if theres a section
        null_verdieping_rooms['floor_index'] = -100
        section_geometry = null_verdieping_rooms.geometry
        y_values = []

        for geom in section_geometry:
            if geom is not None:
                if geom.geom_type == "MultiPolygon":
                    for poly in geom.geoms:
                        y_values.extend([point[1] for point in poly.exterior.coords])
                elif geom.geom_type == "Polygon":
                    y_values.extend([point[1] for point in geom.exterior.coords])


        y_values = np.array(y_values)

        # plotting
        # ax = null_verdieping_rooms.plot(figsize=(10, 8), color='lightblue', edgecolor='black')
        #
        # # Adding title and labels for clarity
        # plt.title("Geometries of Null Verdieping Rooms")
        # plt.xlabel("X-Coordinate")
        # plt.ylabel("Y-Coordinate")
        #
        # # Show the plot
        # plt.show()


        y_values_reshaped = y_values.reshape(-1, 1)

        # DBSCAN - detect clusters without knowing the amount of clusters beforehand
        # epsilon = max distance between floors to be considered part of the same cluster
        dbscan = DBSCAN(eps=epsilon_value, min_samples=1)
        floor_indices = dbscan.fit_predict(y_values_reshaped)
        num_floors = max(floor_indices) + 1

        # Plot histogram
        # plt.figure(figsize=(8, 5))
        # plt.hist(y_values, bins=num_floors, edgecolor="black", alpha=0.7)
        # plt.xlabel("Y-Coordinate")
        # plt.ylabel("Frequency")
        # plt.title("Histogram of Y-Values in Section")
        # plt.grid(axis="y", linestyle="--", alpha=0.5)
        #
        # # Show plot
        # plt.show()
        #
        # plt.figure(figsize=(8, 5))
        # sns.scatterplot(x=np.zeros_like(y_values), y=y_values, hue=floor_indices, palette="viridis", s=20)
        # plt.xlabel("Dummy X-axis")
        # plt.ylabel("Y-Coordinate")
        # plt.title("DBSCAN Floor Clustering")
        # plt.show()

        kmeans = KMeans(n_clusters=num_floors, random_state=42)
        kmeans.fit(y_values_reshaped)

        # assign each y-value to a cluster (floor)
        floor_labels = kmeans.labels_

        # get the extrusion heights
        sorted_centers = np.sort(kmeans.cluster_centers_, axis=0)
        # Plot
        # plt.figure(figsize=(8, 5))
        # sns.scatterplot(x=np.zeros_like(y_values), y=y_values, hue=floor_labels, palette="tab10", s=20)
        # plt.scatter(np.zeros_like(sorted_centers), sorted_centers, c="red", marker="x", label="Estimated Floor Heights")
        # plt.xlabel("Dummy X-axis")
        # plt.ylabel("Y-Coordinate")
        # plt.title("KMeans Clustered Floor Heights")
        # plt.legend()
        # plt.show()
        # Calculate the extrusion height (difference in height between each floor)

        extrusion_heights = np.diff(sorted_centers, axis=0).flatten()
        unique_floors = sorted(pand_data['floor_index'].unique())
        num_floors_drawing = len(unique_floors)
        # handle mismatch between the amount of drawn storeys and amount calculated
        if len(extrusion_heights) != num_floors_drawing:
            avg_height = extrusion_heights.sum() / num_floors_drawing if num_floors_drawing > 0 else 0
            floor_to_extrusion = {index: avg_height for index in unique_floors}
            print("Calculated amount of floors differs from amount in the division drawing, setting average:", avg_height)
        else:
            floor_to_extrusion = {index: extrusion_heights[i]
                                  for i, index in enumerate(unique_floors)}

        pand_data['extrusion_height'] = pand_data['floor_index'].map(floor_to_extrusion)

    # Handling extreme values
    height_value = 0
    for floor_index, floor_group in pand_data.groupby('floor_index'):
        floor_height = floor_group["extrusion_height"].iloc[0]

        if floor_height < 2.1:
            print(f"Storey height for floor {floor_index} is smaller than expected: {floor_height}")
            if automatic_capping:
                pand_data["extrusion_height"] = 2.1
            else:
                heightcap = input(
                    f"Type 'y' to set the height to 2.1, 'n' to continue with the current height, or enter the height in meters for floor {floor_index}: ")
                if heightcap.lower() == "y":
                    pand_data["extrusion_height"] = 2.1
                elif heightcap.lower() == "n":
                    pass
                else:
                    try:
                        height_value = float(heightcap)
                        pand_data[ "extrusion_height"] = height_value
                    except ValueError:
                        print("Invalid input: please enter 'y', 'n', or a numeric value.")

        elif floor_height > 3.3:
            print(f"Storey height for floor {floor_index} is taller than expected: {floor_height}")
            if automatic_capping:
                pand_data["extrusion_height"] = 3.3
            else:
                heightcap_max = input(
                    f"Type 'y' to set the height to 3.3, 'n' to continue with the current height, or enter the height in meters for floor {floor_index}: ")
                if heightcap_max.lower() == "y":
                    pand_data["extrusion_height"] = 3.3
                elif heightcap_max.lower() == "n":
                    pass
                else:
                    try:
                        height_value = float(heightcap_max)
                        pand_data["extrusion_height"] = height_value
                    except ValueError:
                        print("Invalid input: please enter 'y', 'n', or a numeric value.")

    # pand_data.set_geometry('optimized_rooms')
    # pand_data.plot()
    # plt.show()

    # create 3D BIM Legal CityJSON files
    pand_data["maaiveld"] = 0.0
    pand_data['optimized_rooms_3d'] = pand_data.apply(
        lambda row: extrude_to_3d(row['optimized_rooms'], row['maaiveld'], floor_height=row['extrusion_height'],
                                  floor_index=row['floor_index']),
        axis=1)
    pand_data['appartement'] = pd.to_numeric(pand_data['appartement'], errors='coerce')

    # Filter out non-3D geometries
    pand_data = pand_data[pand_data['optimized_rooms_3d'].apply(has_3d_coords)].copy()
    pand_data = pand_data.drop(columns=['geom_akte_all', 'geom_akte_all_scaled', 'aligned_rooms', 'optimized_rooms'])
    # pand_data.to_file((out_path / f"pand_data_3d_{perceel}.gpkg"), driver="GPKG")
    output_file = out_path / f"{perceel}.city.json"
    export_to_cityjson(pand_data, output_file)


    all_panden_rooms.append(pand_data)

panden_rooms = pd.concat(all_panden_rooms, ignore_index=True)

end_time = time.time()
print("Executed time: ", end_time - start_time)

# optionally create geojson output file with all buildings
panden_rooms = gp.GeoDataFrame(panden_rooms, geometry='optimized_rooms_3d', crs='EPSG:28992')
panden_rooms.set_geometry('optimized_rooms_3d')
panden_rooms = gp.GeoDataFrame(panden_rooms, geometry='optimized_rooms_3d', crs='EPSG:28992')
panden_rooms2 = panden_rooms[["optimized_rooms_3d", "room", "verdieping", "appartement", "ruimte", "perceel_id", "bag_pnd", "extrusion_height"]]
# panden_rooms2.to_file(os.path.join(out_path, f"alternative_result.geojson"), driver="GeoJSON")

