import os
import json
from operator import contains
from urllib import response

import requests
import geopandas as gp
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import shape
from shapely import geometry as geom
import re
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.lib import unary_union
from shapely.ops import unary_union

pd.set_option('display.max_rows', None)

kad_path = r'C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\original_data\aanvullende_data\Hilversum\Percelen_aktes_Hilversum.shp'
json_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\original_data\gevectoriseerde_set\hilversum_set\observations\snapshots\latest"
out_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\werkmap\output"
pand_geom_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\werkmap\panden.gpkg"
api_bgt = 'https://api.pdok.nl/lv/bgt/ogc/v1/collections/pand/items'
files = os.listdir(json_path)
kadpercelen = gp.read_file(kad_path)
pand_geom = gp.read_file(pand_geom_path)

if not os.path.exists(out_path):
    os.mkdir(out_path)
    print('output directory created')

scale_version = 'text'
rotation_version = 'none'
translation_version = 'centroid'


def calcScale(bbxobj, pix):
    # get the length of the xbbox and ybbox of the kad perceel
    xlen = float((bbxobj.maxx - bbxobj.minx).iloc[0])
    ylen = float((bbxobj.maxy - bbxobj.miny).iloc[0])
    # if the building is drawn horizontally
    if xlen < ylen:
        # why not the other way around? if the x is smaller, the page is horizontal so the pixel approx is for the x right?
        return ylen / float(pix)
    else:
        return xlen / float(pix)


def get_scale_text(text):
    match = re.search(r'1:(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        # put something else here
        return 100


# make polygon geometries based on points
def egetGeometry(plist, pdict):
    gl = []
    for p in plist:
        gl.append(pdict[p])
    return geom.Polygon(gl)


# add attribute info to rooms
def addValue(cat, clist, room):
    try:
        clist.append(data['text'][data['rooms'][room][cat]]['value'])
    except:
        clist.append('')


def translate_polygon(geometries, translation_vector):
    dx, dy = translation_vector
    return geometries.apply(lambda geom: translate(geom, xoff=dx, yoff=dy))


def compute_translation_vector(polygon_centroid, bp):
    Tx = bp.x.iloc[0] - polygon_centroid.x
    Ty = bp.y.iloc[0] - polygon_centroid.y
    return Tx, Ty


def calculate_centroid(pointDict):
    # pointDict is a dictionary with point IDs as keys and (x, y) as values
    n = len(pointDict)
    centroid_x = sum(coord[0] for coord in pointDict.values()) / n
    centroid_y = sum(coord[1] for coord in pointDict.values()) / n
    print('centroid_x vect', centroid_x)
    return centroid_x, centroid_y


parcels_df = kadpercelen[['geometry', 'KAD_GEM', 'SECTIE', 'PERCEELNUM']]

perceel_list = []
# get all the 'Kadastrale aanduidingen' (Gemeente, Sectie, Perceelnr)
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
    bp = selection_perceel.centroid
    print('centroid_x perceel', bp.x.iloc[0])

    # bounds => envelope per perceel
    bbx = selection_perceel.geometry.bounds
    bbox = f'{int(bbx.minx)},{int(bbx.miny)},{int(bbx.maxx)},{int(bbx.maxy)}'

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
    selection_perceel['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
    selection_perceel['perceel_id'] = selection_perceel['perceel_id'].astype(str)
    # join if the bgt intersects with the perceel
    join_bgt = gp.sjoin(bgt_geom_all, selection_perceel, how='inner', predicate='intersects')

    perceel_bgt = join_bgt.set_crs('epsg:28992', allow_override=True)
    perceel_bgt = perceel_bgt[['bgt_lokaal_id', 'bag_pnd', 'geometry', 'perceel_id']]

    # now get the rooms data
    with open(os.path.join(json_path, f'{perceel}.latest.json')) as f:
        data = json.load(f)

    # empty list to create the attributes properly
    roomIDs = []
    appartementsnummer = []
    ruimteomschrijving = []
    verdiepingsaanduiding = []
    geometry = []
    attachment = []
    room_polygons = []
    # filter per verdieping
    for r in data['rooms'].keys():
        roomIDs.append(r)
        addValue('appartementsnummer', appartementsnummer, r)
        addValue('ruimteomschrijving', ruimteomschrijving, r)
        addValue('verdiepingaanduiding', verdiepingsaanduiding, r)
        attachment.append(data['rooms'][r]['attachment'])

        if scale_version == 'text':
            scale_text = verdiepingsaanduiding[-1]
            scale_factor = get_scale_text(scale_text)
        else:
            None

        pointDict = {}
        for pt in data['points'].keys():
            x, y = data['points'][pt]['position']
            pointDict[pt] = [
                x / scale_factor,
                # y is top down so first make bottom up
                (float(data['meta']['frontDimensions'][1]) - y) / scale_factor]

        geometry.append(egetGeometry(data['rooms'][r]['points'], pointDict))

    vec_rooms = gp.GeoDataFrame(
        data=zip(roomIDs, verdiepingsaanduiding, appartementsnummer, ruimteomschrijving, attachment),
        geometry=geometry, crs="EPSG:28992",
        columns=['room', 'verdieping', 'appartement', 'ruimte', 'attachment'])

    vec_rooms['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
    vec_rooms['perceel_id'] = vec_rooms['perceel_id'].astype(str)

    # join rooms with perceel through perceel ID
    rooms_perceel = vec_rooms.merge(perceel_bgt, on='perceel_id', how='left')

    pand_data = rooms_perceel.set_geometry('geometry_x')

    # #create a dataframe for the panden with the polygon outline of the BG
    rooms_bg = pand_data[pand_data['verdieping'].fillna('').str.lower().str.contains("begane grond")]
    pand_outline = rooms_bg.groupby('bgt_lokaal_id')['geometry_x'].apply(lambda g: g.unary_union)
    pand = gp.GeoDataFrame(
        pand_outline,
        geometry=pand_outline,
        crs=rooms_bg.crs)

    pand = pand.merge(pand_data, left_on='bgt_lokaal_id', right_on='bgt_lokaal_id', how='left')

    # allign centroids -> translation
    if translation_version == 'centroid':
        bgt_centroid = perceel_bgt.geometry.centroid.iloc[0]
        print('bgt_centroid', bgt_centroid)
        if len(pand.geometry.centroid) > 1:
            building_centroid = pand.geometry.centroid.iloc[0]
        else:
            building_centroid = pand.geometry.centroid


        translation_vector = (
            bgt_centroid.x - building_centroid.x,
            bgt_centroid.y - building_centroid.y
        )
        print(pand.info())
        pand['aligned_geometry'] = translate_polygon(pand.geometry, translation_vector)
        pand['vec_pand_rooms'] = translate_polygon(pand['geometry_x_y'], translation_vector)
        pand['bgt_geometry'] = translate_polygon(pand['geometry_y'], translation_vector)
    else:
        None

    pand = gp.GeoDataFrame(pand, geometry='geometry_x_x', crs='EPSG:28992')
    all_panden.append(pand)

panden = pd.concat(all_panden)
panden = panden.set_crs('epsg:28992', allow_override=True)

# outline BG pand
panden = panden.set_geometry('aligned_geometry')
panden.plot()
plt.show()
panden1 = panden['aligned_geometry']
panden1.to_file("test_scale_centroid.shp", driver="ESRI Shapefile")
# georeferenced pand
panden = panden.set_geometry('bgt_geometry')
panden.to_csv('panden.csv', index=False)
# panden2 = panden['bgt_geometry']
# panden2.to_file("bgt_geometry2", driver="ESRI Shapefile")
panden.plot()
plt.show()

# separate_pand = panden.groupby(['bgt_lokaal_id']).first()
panden.to_csv('separate_pand.csv', index=True)

