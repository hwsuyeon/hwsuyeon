import os
import pandas as pd
import geopandas as gpd
import networkx as nx
import json
import time
import numpy as np
from shapely.geometry import Point, mapping, Polygon, LineString

import warnings
warnings.filterwarnings("ignore")

starttime = time.time()

os.chdir('../')
filedir = './inputData/' # input file path
finpath = './outputData/evacuation/evacuationDist/' # output file path

import sys
damagefile = sys.argv[1]
sidoCode = sys.argv[2]
encod  = 'UTF8'
sidoCode = str(sidoCode)

date, dtype = damagefile.split("_")[1], damagefile.split("_")[2]
mtype = "EVADS"
sep = '_'
baseJSON = mtype+sep+date+sep+dtype+sep

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


one = gpd.read_file(filedir + 'Common/damagePolygon/processed/' + sidoCode + '/' + damagefile + '.shp')
Pone = one.to_crs({'init' :'epsg:5179'})
OVER = gpd.GeoDataFrame(index = [0], crs = {'init' :'epsg:5179'}, geometry = [Pone.unary_union])

dangerLayer = OVER

buildLayer = gpd.read_file(filedir + "Common/buildPOP/BuildingPOP_" + sidoCode + ".shp", encoding='euckr')
shelLayer = gpd.read_file(filedir + "Common/shelter/shelter_" + sidoCode + ".shp", encoding='utf-8')


dangerLayer_buf = dangerLayer.buffer(2000)
building_intersect = buildLayer['geometry'].intersects(dangerLayer.unary_union)
shelter_intersect = shelLayer['geometry'].intersects(dangerLayer_buf.unary_union)

shelter_diff = shelLayer[shelter_intersect]


building_intersect = buildLayer[building_intersect]
shelter_diff = shelter_diff[~shelter_diff['geometry'].intersects(dangerLayer.unary_union)]


evaPop = list(building_intersect['POP'])
capPop = list(shelter_diff['capacity'])

BuildingPoints = []
build_x = list(building_intersect.centroid.x)
build_y = list(building_intersect.centroid.y)
for i in range(len(building_intersect)):
    BuildingPoints.append((build_x[i], build_y[i]))


ShelterPoints = []
shel_x = list(shelter_diff.centroid.x)
shel_y = list(shelter_diff.centroid.y)
for i in range(len(shelter_diff)):
    ShelterPoints.append((shel_x[i], shel_y[i]))


roadnetwork = nx.read_shp(filedir + "Common/road/1NGII_" + sidoCode + ".shp", simplify=False)
roadnetwork2 = roadnetwork.to_undirected()


rawPoint = []
gp_edgelist = list(roadnetwork2.edges())
for i in range(len(gp_edgelist)):
    for j in range(2):
        rawPoint.append(gp_edgelist[i][j])
uniq_rawP = list(set(rawPoint))


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


need2link_build = []
BuildingRoad = []
for i, bP in enumerate(BuildingPoints):
    min_dist = 1e3
    for rawP in rawPoint:
        # distance b/w a and b
        d = dist(bP, rawP)
        if d < min_dist:
            min_dist = d
            rP = rawP
    need2link_build.append((bP, rP))
    BuildingRoad.append({'id': i,
                         'RoadPoint': bP,
                         'residents' : evaPop[i]})


need2link_shelt = []
ShelterRoad = []
for i, sP in enumerate(ShelterPoints):
    min_dist = 1e3
    for rawP in rawPoint:
        # distance b/w a and b
        d = dist(sP, rawP)
        if d < min_dist:
            min_dist = d
            rP = rawP
    need2link_shelt.append((sP, rP))
    ShelterRoad.append({'id': i,
                     'RoadPoint': sP,
                     'capacity' : capPop[i]})


roadnetwork2.add_edges_from(need2link_build)


roadnetwork2.add_edges_from(need2link_shelt)

#### A*

nx.set_edge_attributes(roadnetwork2,
                       {e: dist(e[0], e[1]) for e in roadnetwork2.edges()}, 'cost')


def takeSecond(elem):
    return elem[1]

routeTable = pd.DataFrame(columns = ['BuildId', 'ShelId', 'Length', 'Route'])


for i, source in enumerate(BuildingRoad):
    for j, target in enumerate(ShelterRoad):
        try:
            shortPath = nx.astar_path(roadnetwork2,
                                         source['RoadPoint'],
                                         target['RoadPoint'],
                                         heuristic = dist,
                                         weight='cost')
        except:
            try:
               
                d = []
                for k, rawP in enumerate(uniq_rawP):
                    d.append((k, dist(source['RoadPoint'], rawP)))
                d.sort(key=takeSecond)
                
                
                for dth in range(1, len(d)):
                    roadnetwork2.add_edge(source['RoadPoint'], uniq_rawP[d[dth][0]])
                    if nx.has_path(roadnetwork2, source['RoadPoint'], target['RoadPoint']):
                        break                   
                
                shortPath = nx.astar_path(roadnetwork2,
                                         source['RoadPoint'],
                                         target['RoadPoint'],
                                         heuristic = dist,
                                         weight='cost')
            except:
                shortPath = 'NA'
            
        if shortPath != 'NA':
            length = sum(roadnetwork2[u][v].get('cost', 1) for u, v in zip(shortPath[:-1], shortPath[1:]))
            
            row = pd.DataFrame(data = [[i, j, length, shortPath]],
                               columns = ['BuildId', 'ShelId', 'Length', 'Route'])
            routeTable = routeTable.append(row)

routeTable.loc[:, 'RouteId'] = range(len(routeTable))


routeList = []
binBuildList = []
binShelList = []

while sum(evaPop) > 0 and sum(capPop) > 0:

    conRouteTable = routeTable[routeTable['BuildId'].isin(binBuildList) == False]

    conRouteTable = conRouteTable[conRouteTable['ShelId'].isin(binShelList) == False]

    buildId = conRouteTable.loc[conRouteTable['Length'] == min(conRouteTable['Length']),
                                 'BuildId'].iloc[0]

    subTable = conRouteTable.loc[conRouteTable['BuildId'] == buildId, :]

    subTableSort = subTable.sort_values(by=['Length'])
    
    for i in range(len(subTableSort)):

        eva = evaPop[buildId]

        shelId = subTableSort.iloc[i, 1]

        cap = capPop[shelId]


        if cap >= eva:
  
            value = eva

            routeList.append(subTableSort.iloc[i, 4])

            evaPop[buildId] = 0

            capPop[shelId] = cap - eva

            binBuildList.append(buildId)

            routeTable.loc[routeTable['RouteId'] == subTableSort.iloc[i, 4], 'Pop'] = value

            break


        else:

            value = cap

            routeList.append(subTableSort.iloc[i, 4])

            capPop[shelId] = 0

            evaPop[buildId] = eva - cap

            binShelList.append(shelId)

            routeTable.loc[routeTable['RouteId'] == subTableSort.iloc[i, 4], 'Pop'] = value       


routeTable = routeTable[routeTable['Pop'].isna() == False]


routelist = []
for i in range(len(routeTable)):
    routelist.append(LineString(routeTable['Route'].iloc[i]))

routegeo = pd.DataFrame(routelist, columns = ['geometry'])

routeTable = routeTable.reset_index()
routeTable = pd.concat([routeTable, routegeo], axis = 1)


routeTable = routeTable.drop(columns=['Route', 'index'])
routeTable = gpd.GeoDataFrame(routeTable)


routeTable.crs = {'init' : 'epsg:5179'}


import shapefile
from osgeo import gdal
from json import dumps

route_wgs = routeTable.to_crs(epsg = 4326)

dsv = route_wgs.dissolve('ShelId', aggfunc = 'sum')
dsv['style'] = [{"stroke": {'color': list(np.random.choice(range(255),size=3))+[1], "width":3, "linecap":"round"}} for i in dsv.index]
dsv = dsv.loc[:, ['Pop', 'style', 'geometry']].reset_index()

buffer = []
for i in range(len(dsv)):
    atr = dict(dsv.iloc[i, :-1])
    geom = dsv.geometry[i].__geo_interface__
    buffer.append(dict(type="Feature", geometry=geom, properties=atr))
    

geojson = open(finpath + '4_' + baseJSON + 'ROUTE.json', "w")
geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2, cls=NpEncoder) + "\n")
geojson.close()

import numpy as np
import folium
from folium import plugins

shelter_wgs = shelter_diff.to_crs(epsg = 4326)
shelter_wgs['allo'] = list(dsv.Pop) + list(np.repeat(0, len(shelter_wgs)-len(dsv)))
shelter_wgs['style'] = [{'color': [255, 0, 0, 1]} for i in range(len(shelter_wgs))]
shelter_wgs = shelter_wgs.loc[:, ['민방위', 'capacity', 'allo', 'style', 'geometry']]
shelter_wgs.to_file(finpath + '3_' + baseJSON + 'SHELTER.json', driver = 'GeoJSON', encoding=encod)



building_wgs = building_intersect.to_crs(epsg = 4326)
building_wgs['style'] = [{'stroke': {'color': [100, 100, 100, 0.5], 'width':1}, "fill": {'color': [100, 100, 100, 0.3]}} for i in range(len(building_wgs))]
building_wgs = building_wgs.loc[:, ['POP', 'style', 'geometry']]
building_wgs.to_file(finpath + '2_' + baseJSON + 'BUILDING.json', driver = 'GeoJSON', encoding=encod)


danger_wgs = dangerLayer.to_crs(epsg = 4326)
danger_wgs['style'] = [{'stroke': {'color': [0, 0, 255, 0.5], 'width':1}, "fill": {'color': [0, 0, 255, 0.3]}}]

danger_wgs.to_file(finpath + '1_'+ baseJSON + 'DAMAGE.json', driver = 'GeoJSON', encoding=encod)



import pyproj
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium import plugins

shelter = json.load(open(finpath + '3_' + baseJSON + 'SHELTER.json', encoding=encod))
BUILDING_J = json.load(open(finpath + '2_' + baseJSON + 'BUILDING.json', encoding=encod))
OVER1_J = json.load(open(finpath + '1_' + baseJSON + 'DAMAGE.json', encoding=encod))
Route_J = json.load(open(finpath + '4_' + baseJSON + 'ROUTE.json', encoding=encod))

center = list(reversed(list(danger_wgs['geometry'].centroid[0].coords[0])))
base = folium.Map(location = center, zoom_start=16)

def rgb_to_hex(lst):
    r, g, b = int(lst[0]), int(lst[1]), int(lst[2])
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

shellist = {}
for i, js in enumerate(shelter['features']):
    shellist[i] = js['properties']['민방위']

routegroup = []
lgd_txt = '<span style="color: {col};">{txt}</span>'

for feat in Route_J['features']:
    lycol = rgb_to_hex(feat['properties']['style']['stroke']['color'][0:3])
    style_function = lambda x: {'color': rgb_to_hex(x['properties']['style']['stroke']['color'][0:3]), 'weight':3}
    fg = folium.FeatureGroup(name = lgd_txt.format(txt = shellist[int(feat['properties']['ShelId'])], col = lycol), show=False)
    route = folium.GeoJson(feat, style_function=style_function).add_to(fg)
    routegroup.append(fg)
    
    
buildingtomap = folium.GeoJson(BUILDING_J, show=False).add_to(base)
buildingtomap.add_children(folium.features.GeoJsonTooltip(fields=['POP'],
                                                          aliases=['피해인구(명):']))
buildingtomap.layer_name = '건물'


sheltertomap = folium.GeoJson(shelter, show=False).add_to(base)
sheltertomap.add_children(folium.features.GeoJsonTooltip(fields=['민방위', 'capacity', 'allo'],
                                                         aliases=['기관명:', '수용인원:', '할당인원:'],
                                                         style=('font-size: 24px')))
sheltertomap.layer_name = '대피소'
style_function = lambda x: {
                            'color':'skyblue',
                            'fillColor':"cyan",
                            'opacity': 0.1,
                            'fillOpacity':0.4}


overflow1 = folium.GeoJson(OVER1_J,
    style_function = style_function,
    show=True).add_to(base)
overflow1.layer_name = '피해지역'
for i in routegroup:
    base.add_child(i)
base.add_child(folium.map.LayerControl())
base.save(finpath + '0_' + baseJSON + 'MAP.html')


print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')