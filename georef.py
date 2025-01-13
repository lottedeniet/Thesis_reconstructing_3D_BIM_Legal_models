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
import math

from shapely.geometry.linestring import LineString

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
            print("angle", angle)

            for col in geometry_columns:
                geometry_pol = row[col]
                group.at[index, col] = rotate(geometry_pol, angle, origin='centroid')

    return group


def translate_polygon(geometries, translation_vector):
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

# start code

# options: text, area
scale_version = 'area'
# options: azimuth, arrow
rotate_version = 'azimuth'
# options: bbox, centroid
translation_version = 'bbox'
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

    outline = pand.groupby('perceel_id').agg({'geom_bgt': lambda g: g.unary_union})
    bgt_outline = gp.GeoDataFrame(outline, geometry='geom_bgt', crs=28992)
    bgt_outline = bgt_outline.rename_geometry('bgt_outline')
    pand = pand.merge(bgt_outline, on='perceel_id', how='left')

    pand = gp.GeoDataFrame(pand, geometry='geom_akte_bg')

    if rotate_version == 'arrow':
        pand = pand.apply(rotate_geom_arrow, axis=1)
    if rotate_version == 'azimuth':
        pand_grouped = pand.groupby('perceel_id').apply(rotate_geom_azimuth)
        pand = pand.merge(pand_grouped, on='bag_pnd', how='left')
        pand = pand.loc[:, ~pand.columns.str.endswith('_x')]
        pand.columns = pand.columns.str.replace('_y', '', regex=False)

    print(pand.info())
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
        # pand_data['geom_akte_all_scaled'] = pand_data['geom_akte_all'].apply(
        #     lambda g: scale(g, xfact=scale_factor, yfact=scale_factor, origin='centroid'))

        pand.set_geometry('geom_akte_bg_scaled', inplace=True)
        pand['geom_akte_bg'] = pand['geom_akte_bg_scaled']
        # pand_data['geom_akte_all'] = pand_data['geom_akte_all_scaled']
        pand.drop(columns=['geom_akte_bg_scaled'], axis=1, inplace=True)
        # pand_data.drop(columns=['geom_akte_all_scaled'], axis=1, inplace=True)
        pand.set_geometry('geom_akte_bg', inplace=True)



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
    pand = gp.GeoDataFrame(pand, geometry='geom_bgt', crs='EPSG:28992')

    pand = gp.GeoDataFrame(pand, geometry='aligned_geometry', crs='EPSG:28992')



    pand['gof'] = pand.apply(lambda row: goodness_of_fit(row['aligned_geometry'], row['bgt_outline']), axis=1)


    all_panden.append(pand)

panden = pd.concat(all_panden)


panden.to_csv('separate_pand.csv', index=True)

import matplotlib.pyplot as plt

# Methods and their average performance
methods = ['Azimuth', 'Arrow', 'Bbox', 'Centroid', 'Text','Area']
average_performance = [0.278,  0.288, 0.279, 0.287, 0.252, 0.314]
colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']

# Plotting the results
plt.barh(methods, average_performance, color=colors)
plt.xlabel('Average Performance')
plt.title('Average Performance of Methods')
plt.show()

panden.set_geometry('aligned_geometry', crs='EPSG:28992')
panden2 = panden.drop(columns=['geom_akte_bg', 'geom_bgt', 'bgt_outline'])
print(panden2['gof'].sum()/len(panden2))
panden2.to_file("gof_"+ rotate_version + "_" + translation_version + "_" + scale_version + ".shp", driver="ESRI Shapefile")
# panden2.to_file( "gof.shp", driver="ESRI Shapefile")


