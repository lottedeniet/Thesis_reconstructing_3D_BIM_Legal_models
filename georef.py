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
import re
from shapely.geometry import Polygon, MultiPolygon


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
def get_scale_text(text):
    """
    Get the scale from the text in the deed
    :param text:
    :return:
    """
    match = re.search(r'1:(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        # put something else here
        return 100

def calculate_normal_vector(geometry):
    """
    Calculate a normal vector for a polygon or multipolygon geometry.
    """
    if geometry.is_empty:
        return (0.0, 0.0)  # Return a default vector for empty geometries

    # Handle MultiPolygon or Polygon
    if geometry.geom_type == 'MultiPolygon':
        geometry = max(geometry.geoms, key=lambda g: g.area)  # Largest polygon

    exterior_coords = list(geometry.exterior.coords)
    if len(exterior_coords) > 1:
        # Compute normal vector from the first edge
        x1, y1 = exterior_coords[0]
        x2, y2 = exterior_coords[1]
        dx, dy = x2 - x1, y2 - y1
        normal_vector = (-dy, dx)
        length = np.hypot(normal_vector[0], normal_vector[1])
        return (normal_vector[0] / length, normal_vector[1] / length) if length != 0 else (0.0, 0.0)
    return (0.0, 0.0)  # Default vector for degenerate geometries


def calculate_rotation_angle(normal1, normal2):
    """
    Calculate the rotation angle between two normal vectors.
    """
    # Ensure inputs are numpy arrays
    normal1 = np.array(normal1)
    normal2 = np.array(normal2)

    # Calculate the dot product and magnitude of the vectors
    dot_product = np.dot(normal1, normal2)
    magnitude1 = np.linalg.norm(normal1)
    magnitude2 = np.linalg.norm(normal2)

    # Compute the cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip to handle numerical issues

    # Return the rotation angle in radians
    return np.degrees(np.arccos(cos_angle))


def rotate_geom_normal(row):
    pand_outline = row.get('geom_akte_bg')
    geometry_bgt = row.get('geom_bgt')
    geometry_columns = ['geom_akte_bg', 'geom_akte_all']
    normal_pand_outline = calculate_normal_vector(pand_outline)
    normal_geometry_bgt = calculate_normal_vector(geometry_bgt)

    angle = calculate_rotation_angle(normal_pand_outline, normal_geometry_bgt)
    for col in geometry_columns:
        geometry = row.get(col)
        row[col] = rotate(geometry, angle, origin='centroid')
    return row


def rotate_geom_arrow(row):
    perceel_id = row.get('perceel_id', None)
    angle = rotation_angles.get(perceel_id, 0)
    geometry_columns = [col for col in row.index if 'geom_akte_bg' in col]

    for col in geometry_columns:
        geometry = row[col]
        row[col] = rotate(geometry, angle, origin='centroid')
    return row


def translate_polygon(geometries, translation_vector):
    dx, dy = translation_vector
    return geometries.apply(lambda geom: translate(geom, xoff=dx, yoff=dy))




# start code

# options: text, area
scale_version = 'text'
# options: normal, arrow
rotate_version = 'arrow'
# options: bbox, centroid
translation_version = 'centroid'
rotation_angles = {'HVS00N1878': 171.3, "HVS00N1882": 180, "HVS00N2359": -43.5, "HVS00N2643": 78.9, "HVS00N2848": 6.8, "HVS00N3211": 0.0, "HVS00N3723": 121.6, "HVS00N4216": 120, "HVS00N555": 22.2, "HVS00N9252":7}


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

    # get the geometry and bag id
    features = response_json.get('features', [])
    bgt_data = []
    for feature in features:
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        lokaal_id = properties.get('lokaal_id')
        bag_pnd = properties.get('bag_pnd', None)
        coordinates = geometry.get('coordinates', None)
        bgt_data.append({'bgt_lokaal_id': lokaal_id, 'bag_pnd': bag_pnd, 'geometry': coordinates})

    # convert to polygon geometries
    for item in bgt_data:
        item['geometry'] = shape({'type': 'MultiPolygon', 'coordinates': item['geometry']})

    bgt_geom_all = gp.GeoDataFrame(bgt_data, geometry='geometry', crs=28992)

    # now intersect with the actual geometry bc otherwise its too many polygons
    join_bgt = gp.sjoin(bgt_geom_all, selection_perceel, how='inner', predicate='intersects')

    perceel_bgt = join_bgt.set_crs('epsg:28992', allow_override=True)
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

    if rotate_version == "normal":

        if not pand.empty:
            pand = pand[pand['geom_akte_bg'].notnull() & pand['geom_bgt'].notnull()]

            pand['normal_akte_bg'] = pand['geom_akte_bg'].apply(calculate_normal_vector)
            pand['normal_bgt'] = pand['geom_bgt'].apply(calculate_normal_vector)

            # Calculate rotation angles and filter negligible rotations
            pand['rotation_angle'] = pand.apply(
                lambda row: calculate_rotation_angle(row['normal_akte_bg'], row['normal_bgt']),
                axis=1
            )


            # Apply rotation
            pand['geom_akte_bg_rotated'] = pand.apply(
                lambda row: rotate(row['geom_akte_bg'], row['rotation_angle']),
                axis=1
            )
            # needs fixing
            pand_data['geom_akte_all_rotated'] = pand_data.apply(
                lambda row: rotate(row['geom_akte_all'], row['rotation_angle']),
                axis=1
            )

            # Update geometries
            pand.set_geometry('geom_akte_bg_rotated', inplace=True)
            pand['geom_akte_bg'] = pand['geom_akte_bg_rotated']
            pand_data['geom_akte_all'] = pand_data['geom_akte_all_rotated']
            pand.drop(columns=['geom_akte_bg_rotated'], axis=1, inplace=True)
            pand_data.drop(columns=['geom_akte_all_rotated'], axis=1, inplace=True)


    if rotate_version == 'arrow':
        pand = pand.apply(rotate_geom_arrow, axis=1)

    if scale_version == 'area':
        pand.set_geometry('geom_bgt', inplace=True)
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
        pand['geom_akte_bg_scaled'] = pand['geom_akte_bg'].apply(
            lambda g: scale(g, xfact=scale_factor, yfact=scale_factor, origin='centroid'))
        pand_data['geom_akte_all_scaled'] = pand_data['geom_akte_all'].apply(
            lambda g: scale(g, xfact=scale_factor, yfact=scale_factor, origin='centroid'))

        pand.set_geometry('geom_akte_bg_scaled', inplace=True)
        pand['geom_akte_bg'] = pand['geom_akte_bg_scaled']
        pand_data['geom_akte_all'] = pand_data['geom_akte_all_scaled']
        pand.drop(columns=['geom_akte_bg_scaled'], axis=1, inplace=True)
        pand_data.drop(columns=['geom_akte_all_scaled'], axis=1, inplace=True)

    pand = gp.GeoDataFrame(pand, geometry='geom_akte_bg')
    # allign centroids -> translation
    if translation_version == 'centroid':
        bgt_centroid = perceel_bgt.geometry.centroid.iloc[0]
        if len(pand.geometry.centroid) > 1:
            building_centroid = pand.geometry.centroid.iloc[0]
        else:
            building_centroid = pand.geometry.centroid

        translation_vector = (  bgt_centroid.x - building_centroid.x,
                                bgt_centroid.y - building_centroid.y)

    if translation_version == 'bbox':
        if len(perceel_bgt.geometry.bounds) > 1:
            bgt_bbox = perceel_bgt.geometry.bounds.iloc[0]
        else:
            bgt_bbox = perceel_bgt.geometry.bounds
        if len(pand.geometry.bounds)> 1:
            building_bbox = pand.geometry.bounds.iloc[0]
        else:
            building_bbox = pand.geometry.bounds

        translation_vector = (bgt_bbox.minx - building_bbox.minx,
                              bgt_bbox.miny - building_bbox.miny)

    pand.set_geometry('geom_akte_bg')
    pand['aligned_geometry'] = translate_polygon(pand.geometry, translation_vector)
    # still need to translate the rooms too
    # pand_data['geom_akte_all'] = translate_polygon(pand_data['geom_akte_all'], translation_vector)
    pand['geom_bgt'] = translate_polygon(pand['geom_bgt'], translation_vector)

    pand = gp.GeoDataFrame(pand, geometry='aligned_geometry', crs='EPSG:28992')
    all_panden.append(pand)

panden = pd.concat(all_panden)


panden.to_csv('separate_pand.csv', index=True)


panden.set_geometry('aligned_geometry', crs='EPSG:28992')
panden2 = panden.drop(columns=['geom_akte_bg', 'geom_bgt'])
panden2.to_file(rotate_version + "_" + translation_version + "_" + scale_version + "2.shp", driver="ESRI Shapefile")
