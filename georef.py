import os
import json
from operator import contains
from urllib import response

import requests
import geopandas as gp
import pandas as pd
from shapely.geometry import shape
from shapely import geometry as geom
import re
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)

kad_path = r'C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\original_data\aanvullende_data\Hilversum\Percelen_aktes_Hilversum.shp'
json_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\original_data\gevectoriseerde_set\hilversum_set\observations\snapshots\latest"
out_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\werkmap\output"
pand_geom_path = r"C:\Users\NietLottede\OneDrive - Kadaster\Documenten\Lotte\github_code\thesis\werkmap\panden.gpkg"
api_bgt = 'https://api.pdok.nl/lv/bgt/ogc/v1/collections/pand/items'


if not os.path.exists(out_path):
    os.mkdir(out_path)
    print('output directory created')

files = os.listdir(json_path)
kadpercelen = gp.read_file(kad_path)
pand_geom = gp.read_file(pand_geom_path)

def calcScale(bbxobj,pix):
    # get the length of the xbbox and ybbox of the kad perceel
    xlen = float((bbxobj.maxx-bbxobj.minx).iloc[0])
    ylen = float((bbxobj.maxy-bbxobj.miny).iloc[0])
    # if the building is drawn horizontally
    if xlen < ylen:
        # why not the other way around? if the x is smaller, the page is horizontal so the pixel approx is for the x right?
        return ylen/float(pix)
    else:
        return xlen/float(pix)

def get_scale_text(text):
    print(text)
    match = re.search(r'1:(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        return 100

# make polygon geometries based on points
def egetGeometry(plist,pdict):
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

parcels_df = kadpercelen[['geometry', 'KAD_GEM', 'SECTIE', 'PERCEELNUM']]

perceel_list = []
# get all the 'Kadastrale aanduidingen' (Gemeente, Sectie, Perceelnr)
for f in files:
    if f.endswith('.json'):
        parts = f.split('.')
        perceel_list.append(parts[0])




# kadperceel_pand = gp.sjoin(kadpercelen, kadpandpoint, how='left', predicate='contains')
all_panden = []

for perceel in perceel_list:
    print(perceel)
    parts = perceel.split('_')
    selection_perceel = kadpercelen[
        kadpercelen.KAD_GEM.eq(parts[1]) & kadpercelen.SECTIE.eq(parts[2]) & kadpercelen.PERCEELNUM.eq(
            int(parts[3]))]
    bp = selection_perceel.centroid

    # bounds => envelope per perceel
    bbx = selection_perceel.geometry.bounds
    bbox = f'{int(bbx.minx)},{int(bbx.miny)},{int(bbx.maxx)},{int(bbx.maxy)}'

    # connect to the BGT
    params = {'bbox': bbox, 'bbox-crs': 'http://www.opengis.net/def/crs/EPSG/0/28992', 'crs':'http://www.opengis.net/def/crs/EPSG/0/28992'}
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

    #now intersect with the actual geometry bc otherwise its too many polygons
    selection_perceel['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
    selection_perceel['perceel_id'] = selection_perceel['perceel_id'].astype(str)
    # join if the bgt intersects with the perceel
    join_bgt = gp.sjoin(bgt_geom_all, selection_perceel, how='inner', predicate='intersects')

    perceel_bgt = join_bgt.set_crs('epsg:28992', allow_override=True)
    perceel_bgt = perceel_bgt[['bgt_lokaal_id', 'bag_pnd', 'geometry', 'perceel_id']]

    # # join with bag
    # bag_geom = pand_geom.set_geometry('geometry')
    # perceel_bgt_bag = perceel_bgt.merge(bag_geom, left_on='bag_pnd', right_on='identificatie', how='left')
    # print('perceel_bgt_bag')
    # print(perceel_bgt_bag.info())

    # now get the rooms data
    with open(os.path.join(json_path, f'{perceel}.latest.json')) as f:
        data = json.load(f)

    #  get the approximate size in pixels of the building
    pixapprox = max(data['meta']['frontDimensions']) / 3
    scale = calcScale(bbx, pixapprox)
    # create pointdict

    # translate to the centroid kinda? and scaled according to scale factor
    # float(data['meta']['frontDimensions'][1]) - y => just go from top left to bottom left y coord
    # pointDict = {}
    # for pt in data['points'].keys():
    #     x, y = data['points'][pt]['position']
    #     pointDict[pt] = [x * scale + float(bp.x.iloc[0]),
    #                      (float(data['meta']['frontDimensions'][1]) - y) * scale + float(bp.y.iloc[0])]


    # empty list to create the attributes properly
    roomIDs = []
    appartementsnummer = []
    ruimteomschrijving = []
    verdiepingsaanduiding = []
    geometry = []
    attachment = []
    # filter per verdieping
    for r in data['rooms'].keys():
        roomIDs.append(r)
        addValue('appartementsnummer', appartementsnummer, r)
        addValue('ruimteomschrijving', ruimteomschrijving, r)
        addValue('verdiepingaanduiding', verdiepingsaanduiding, r)
        attachment.append(data['rooms'][r]['attachment'])
        scale_text = verdiepingsaanduiding[-1]
        scale_factor = get_scale_text(scale_text)

        pointDict = {}
        for pt in data['points'].keys():
            x, y = data['points'][pt]['position']
            pointDict[pt] = [
                x/ scale_factor + float(bp.x.iloc[0]),
                (float(data['meta']['frontDimensions'][1])- y) / scale_factor + float(bp.y.iloc[0]),
            ]

        geometry.append(egetGeometry(data['rooms'][r]['points'], pointDict))

    vec_rooms = gp.GeoDataFrame(data=zip(roomIDs, verdiepingsaanduiding, appartementsnummer, ruimteomschrijving, attachment),
                          geometry=geometry, crs="EPSG:28992",
                          columns=['room', 'verdieping', 'appartement', 'ruimte', 'attachment'])

    vec_rooms['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
    vec_rooms['perceel_id'] = vec_rooms['perceel_id'].astype(str)

    # join rooms with perceel through perceel ID
    rooms_perceel = vec_rooms.merge(perceel_bgt, on='perceel_id', how='left')
    pand_data = rooms_perceel.set_geometry('geometry_y')
    print('rooms_perceel')
    print(rooms_perceel.info())
    # #create a dataframe for the panden with the polygon outline of the BG
    rooms_bg = pand_data[pand_data['verdieping'].fillna('').str.lower().str.contains("begane grond")]
    pand_outline = rooms_bg.groupby('bgt_lokaal_id')['geometry_x'].apply(lambda g: g.unary_union)
    pand = pd.DataFrame(pand_outline).reset_index()
    pand = pand.merge(pand_data, left_on='bgt_lokaal_id', right_on='bgt_lokaal_id', how='left')
    print('pand')
    print(pand.info())
    pand = gp.GeoDataFrame(pand, geometry='geometry_x_x', crs='EPSG:28992')
    all_panden.append(pand)

panden = pd.concat(all_panden)
panden.rename(columns={'geometry_x_x':'vec_pand_outline', 'geometry_x_y': 'vec_pand_rooms', 'geometry_y' : 'bgt_geometry'}, inplace=True)


# # only get the id numbers
# panden['num_id'] = panden['pand_id'].str.extract(r'(\d+)').astype(int)




panden = panden.set_geometry('vec_pand_rooms')
panden = panden.set_crs('epsg:28992', allow_override=True)
panden.plot()
plt.show()
# outline BG pand
panden = panden.set_geometry('vec_pand_outline')
panden.plot()
plt.show()
# georeferenced pand
panden = panden.set_geometry('bgt_geometry')
panden.to_csv('panden.csv', index=False)
panden.plot()
plt.show()


separate_pand = panden.groupby(['bgt_lokaal_id']).first()
separate_pand.to_csv('separate_pand.csv', index=True)

