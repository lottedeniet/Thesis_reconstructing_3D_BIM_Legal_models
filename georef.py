import os
import json
from operator import contains

import geopandas as gp
import pandas as pd
from shapely import geometry as geom
from shapely.affinity import translate
import matplotlib.pyplot as plt
from win32gui import Polygon

kad_path = r'C:\Users\NietLottede\Documents\Lotte\original_data\aanvullende_data\Hilversum\Percelen_aktes_Hilversum.shp'
json_path = r"C:\Users\NietLottede\Documents\Lotte\original_data\gevectoriseerde_set\hilversum_set\observations\snapshots\latest"
out_path = r"C:\Users\NietLottede\Documents\Lotte\github_code\thesis\werkmap\output"
pand_geom_path = r"C:\Users\NietLottede\Documents\Lotte\github_code\thesis\werkmap\panden.gpkg"

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
    parts = perceel.split('_')
    selection = kadpercelen[
        kadpercelen.KAD_GEM.eq(parts[1]) & kadpercelen.SECTIE.eq(parts[2]) & kadpercelen.PERCEELNUM.eq(
            int(parts[3]))]
    bp = selection.centroid
    # bounds => envelope per perceel
    bbx = selection.bounds
    print(bbx)

    with open(os.path.join(json_path, f'{perceel}.latest.json')) as f:
        data = json.load(f)

    #  get the approximate size in pixels of the building
    pixapprox = max(data['meta']['frontDimensions']) / 3
    scale = calcScale(bbx, pixapprox)
    # create pointdict

    # translate to the centroid kinda? and scaled according to scale factor
    # float(data['meta']['frontDimensions'][1]) - y => just go from top left to bottom left y coord
    pointDict = {}
    for pt in data['points'].keys():
        x, y = data['points'][pt]['position']
        pointDict[pt] = [x * scale + float(bp.x.iloc[0]),
                         (float(data['meta']['frontDimensions'][1]) - y) * scale + float(bp.y.iloc[0])]


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
        geometry.append(egetGeometry(data['rooms'][r]['points'], pointDict))

    vec_rooms = gp.GeoDataFrame(data=zip(roomIDs, verdiepingsaanduiding, appartementsnummer, ruimteomschrijving, attachment),
                          geometry=geometry, crs="EPSG:28992",
                          columns=['room', 'verdieping', 'appartement', 'ruimte', 'attachment'])

    vec_rooms['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
    vec_rooms['perceel_id'] = vec_rooms['perceel_id'].astype(str)
    selection['perceel_id'] = str(parts[1]) + str(parts[2]) + str(parts[3])
    selection['perceel_id'] = selection['perceel_id'].astype(str)
    # join rooms with perceel through perceel ID
    rooms_perceel = vec_rooms.merge(selection[['perceel_id', 'geometry']], on='perceel_id', how='left')

    # use the centroid of the bag panden to check if the pand is on the perceel
    # not optimal, remakes all points for every perceel
    centroid_pand = pand_geom.centroid
    pand_geom['centroid'] = centroid_pand
    rooms_perceel = rooms_perceel.set_geometry('geometry_y')
    pand_geom = pand_geom.set_geometry('centroid')

    # pand data still has all the buildings
    pand_data = rooms_perceel.sjoin(pand_geom, how='left', predicate='contains')
    pand_data.plot()
    plt.show()
    print(pand_data.info())
    # rooms = pand_data[['room', 'verdieping', 'appartement', 'ruimte', 'attachment', 'geometry_x', 'pand_id']]
    # # rooms.to_file(os.path.join(out_path, f'{perceel}.rooms.gpkg'), driver='GPKG')

    # #create a dataframe for the panden with the polygon outline of the BG
    rooms_bg = pand_data[pand_data['verdieping'].str.contains("BEGANE GROND", na=False)]
    # look into multiple panden per perceel
    pand_outline = rooms_bg.groupby('identificatie')['geometry_x'].apply(lambda g: g.unary_union)
    pand = pd.DataFrame(pand_outline).reset_index()
    pand = pand.merge(pand_data, left_on='identificatie', right_on='identificatie', how='left')
    print(pand.info())
    pand = gp.GeoDataFrame(pand, geometry='geometry_x_x', crs='EPSG:28992')


    all_panden.append(pand)

panden = pd.concat(all_panden)
print(panden.info())
panden.rename(columns={'geometry_x_x':'vec_pand_outline', 'geometry_x_y': 'vec_pand_rooms', 'pand_id_x': 'pand_id', 'pand_id_y':'room_id'}, inplace=True)

# need a better way to structure, now just dropped the rooms
if 'pand_id' in panden.columns:
    panden = panden.drop_duplicates(subset=['pand_id'])

# print("panden")
# panden.to_csv('panden.csv', index=False)

# scale tests
# align centroid

# #not working yet, there are not enough geometries and the ones i have are invalid
# def translate(polygon, dx, dy):
#     new_coords = [(x + dx, y + dy) for x, y in polygon.exterior.coords]
#     return polygon(new_coords)
#
# panden = panden[panden['vec_pand_outline'].notna() & panden['geometry'].notna() & panden['vec_pand_rooms'].notna()]
#
# print(panden['vec_pand_outline'].head())
#
# panden['vec_pand_outline'] = panden.apply(lambda row:translate(row['vec_pand_outline'],
#                                                                row['geometry'].centroid.x - row['vec_pand_outline'].centroid.x,
#                                                                row['geometry'].centroid.y - row['vec_pand_outline'].centroid.y),axis=1)
# print(panden['vec_pand_outline'].head())

    # panden = panden.drop('geometry_x_y', axis=1)
    # panden = panden.drop('geometry', axis=1)
    # panden.to_file(os.path.join(out_path, f'{perceel}.pand.gpkg'), driver='GPKG')

    # plotting

panden = panden.set_geometry('vec_pand_rooms')
panden = panden.set_crs('epsg:28992', allow_override=True)
# vectorised pand
panden.plot()
plt.show()
# outline BG pand
panden = panden.set_geometry('vec_pand_outline')
panden.plot()
plt.show()
# georeferenced pand
panden = panden.set_geometry('geometry')
panden.plot()
plt.show()




