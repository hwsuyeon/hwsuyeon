import os
import rasterio
import geopandas as gpd
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import rasterstats as rs
import numpy as np
from osgeo import gdal
from rasterio.features import shapes
from shapely.geometry import Point
from pyproj import Proj, transform
import shapefile
import json
from json import dumps
import folium
from folium import plugins

import warnings
warnings.filterwarnings("ignore")

starttime = time.time()

import sys
sidoCode = sys.argv[1]
xcoor = sys.argv[2]
ycoor = sys.argv[3]

from datetime import datetime
current_time = datetime.now()
dateandtime = current_time.strftime("%Y%m%dT%H%M")


os.chdir('../')
filedir = './inputData/' # input file directory
finpath = './outputData/sinkArea/'

sidoCode = str(sidoCode)

dtype = "TYPHN"
mtype = "ESINK"
sep = '_'
baseJSON = mtype+sep+dateandtime+sep+dtype+sep

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
createFolder(finpath + sidoCode)

createFolder(finpath + sidoCode + '/jsonResult')

dem = filedir + 'Common/dem/dem_' + sidoCode + '.tif'

shp_fn = finpath + sidoCode + '/' + sidoCode + '_clip_frame.shp'

dem_out = filedir + 'sinkArea/elevation/clipElevation/elevation_'+sidoCode+'_clip.tif'

proj_WGS84 = Proj(init='epsg:4326')
proj_UTMK = Proj(init='epsg:5179')
poiPoint = transform(proj_WGS84, proj_UTMK, float(xcoor), float(ycoor))
poiPoint = Point(poiPoint)
crs = {'init': 'epsg:5179'}
polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[poiPoint.buffer(500)])

polygon.geometry[0] = polygon.geometry.envelope[0]
polygon.to_file(shp_fn)


options = gdal.WarpOptions(cutlineDSName=shp_fn,
                           cropToCutline=True)
outBand = gdal.Warp(srcDSOrSrcDSTab=dem,
                    destNameOrDestDS=dem_out,
                    options=options)
outBand= None


data = rasterio.open(dem_out)


with rasterio.open(dem_out) as dem_out_src:

    dem_data = dem_out_src.read(1, masked=True)
    dem_meta = dem_out_src.profile

pointDF = pd.DataFrame({'a': range(len([poiPoint]))})
pointGDF = gpd.GeoDataFrame(pointDF, geometry = [poiPoint])


dem_stats = rs.zonal_stats(pointGDF,
                           dem_data,
                           nodata=-999,
                           affine=dem_meta['transform'],
                           geojson_out=True,
                           copy_properties=True,
                           stats="mean")


dem_stats_df = gpd.GeoDataFrame.from_features(dem_stats)

poi_dem = dem_stats_df['mean'][0]


gdal_data = gdal.Open(dem_out)
gdal_band = gdal_data.GetRasterBand(1)
nodataval = gdal_band.GetNoDataValue()



data_array = gdal_data.ReadAsArray().astype(np.float)



if np.any(data_array == nodataval):
    data_array[data_array == nodataval] = np.nan


def isBiggerThanMin(x):
    return x > -9999

data_min = 9999
data_max = -9999
for i in data_array:
    data_list = list(filter(isBiggerThanMin, list(i)))
    if min(data_list) < data_min:
        data_min = min(data_list)
    if max(data_list) > data_max:
        data_max = max(data_list)

numlevel = 3
floodlevel = [1, 3, 5]

level_list = []
level_list.append(poi_dem)

for depth in floodlevel:
    level_list.append(poi_dem+depth)


driver = gdal.GetDriverByName('GTiff')
dem_file = gdal.Open(dem_out)
band = dem_file.GetRasterBand(1)
lista = band.ReadAsArray()



for j in range(dem_file.RasterXSize):
    for i in range(dem_file.RasterYSize):
        if lista[i,j] == nodataval:
            lista[i,j] = nodataval
        elif lista[i, j] <= level_list[0]:
            lista[i,j] = 1
        elif level_list[0] < lista[i,j] <= level_list[1]:
            lista[i,j] = 2
        elif level_list[1] < lista[i,j] <= level_list[2]:
            lista[i,j] = 3
        elif level_list[2] < lista[i,j] <= level_list[3]:
            lista[i,j] = 4
        else:
            lista[i,j] = 5



file2 = driver.Create(finpath +sidoCode + '/' + sidoCode + '_elevation_reclass.tif',
                      dem_file.RasterXSize, dem_file.RasterYSize, 1)
file2.GetRasterBand(1).WriteArray(lista)



proj = dem_file.GetProjection()
georef = dem_file.GetGeoTransform()
file2.SetProjection(proj)
file2.SetGeoTransform(georef)
file2.FlushCache()
file2 = None


uljin_dem_reclass = finpath +sidoCode + '/' + sidoCode + '_elevation_reclass.tif'
input_raster = gdal.Open(uljin_dem_reclass)
output_raster = finpath +sidoCode + '/' + sidoCode + '_elevation_reproject.tif'
outBand = gdal.Warp(output_raster, input_raster, dstSRS='EPSG:4326')
outBand = None


mask = None
with rasterio.open(output_raster) as src:
    img = src.read(1)
    results = ({'properties': {'raster_val': v}, 'geometry': s}
               for i, (s, v) 
               in enumerate(
                   shapes(img, mask=mask,
                          transform=src.transform
                         )))


geoms = list(results)
gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)

gpd_polygonized_raster.crs = {'init' : 'epsg:4326'}

floodLayer = gpd_polygonized_raster
floodLayer = floodLayer[floodLayer['raster_val'] <= 4]
floodLayer = floodLayer[floodLayer['raster_val'] != 0.0]

alllist = [number - poi_dem for number in level_list]


colors=['#0000ff', '#26baff', '#2bfcf5', '#00ffcc']

def hextofloats(h):
    return list(int(h[i:i + 2], 16) for i in (1, 3, 5))

floodLayerList = []
for i in range(numlevel+1):
    
    tmpLayer = floodLayer[floodLayer['raster_val'] == float(i+1)]
    tmpLayer['style'] = [{'stroke': {'color': hextofloats(colors[i])+[1], 'width':0.0001}, "fill": {'color': hextofloats(colors[i])+[0.5]}} for x in range(len(tmpLayer))]
    tmpLayer = tmpLayer.loc[:, ['raster_val', 'style', 'geometry']].reset_index()
    
    buffer = []
    for x in range(len(tmpLayer)):
        atr = dict(tmpLayer.iloc[x, :-1])
        geom = tmpLayer.geometry[x].__geo_interface__
        buffer.append(dict(type="Feature", geometry=geom, properties=atr))
    
    geojson = open(finpath + sidoCode + '/jsonResult' +'/' + str(i+1) + '_' + baseJSON + "FLOOD%s.json" % int(alllist[i]), "w")
    geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2, cls=NpEncoder) + "\n")
    geojson.close()
    
    floodLayerList.append(json.load(open(finpath + sidoCode + '/jsonResult' +'/' + str(i+1) + '_' + baseJSON + "FLOOD%s.json" % int(alllist[i]))))


studyAreaLayer = gpd.read_file(shp_fn)
pointGDF = gpd.GeoDataFrame(pointDF, geometry = [poiPoint])
pointGDF['style'] = [{'color':[255, 0, 0, 1]}]
pointGDF.crs = {'init' : 'epsg:5179'}
pointGDF = pointGDF.to_crs(epsg = 4326)
pointGDF = pointGDF.loc[:, ['style', 'geometry']]

pointGDF.to_file(finpath + sidoCode + '/jsonResult' + '/' + '5_' + baseJSON + 'POI.json', driver = 'GeoJSON')


buildingLayer = gpd.read_file(filedir + "Common/buildPOP/BuildingPOP_" + sidoCode + ".shp")

building_intersect = buildingLayer['geometry'].intersects(studyAreaLayer.unary_union)
building_intersect = buildingLayer[building_intersect]

coldic = {'stroke': {'color': [255, 0, 0, 0.5], 'width':1}, "fill": {'color': [255, 0, 0, 0.3]}}
building_intersect['style'] = [coldic for i in range(len(building_intersect))]

buildingLayer = building_intersect.to_crs(epsg = 4326)
buildingLayer = buildingLayer.loc[:, ['POP', 'style', 'geometry']].reset_index(drop = True)

buffer = []
for x in range(len(buildingLayer)):
    atr = dict(buildingLayer.iloc[x, :-1])
    geom = buildingLayer.geometry[x].__geo_interface__
    buffer.append(dict(type="Feature", geometry=geom, properties=atr))

geojson = open(finpath + sidoCode + '/jsonResult' +'/' + '6_' + baseJSON + "BUILDING.json", "w")
geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2, cls=NpEncoder) + "\n")
geojson.close()


import branca.colormap as cm


def rgb_to_hex(lst):
    r, g, b = int(lst[0]), int(lst[1]), int(lst[2])
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

style_function = lambda x: {'fillColor': rgb_to_hex(x['properties']['style']['stroke']['color'][0:3]), 
                            'color': '',
                            'weight': 0.0001,
                            'fillOpacity': 0.5}


center = list(reversed(pointGDF.geometry[0].coords[0]))

base = folium.Map(location = center, zoom_start=15)

for i in range(len(floodLayerList)):
    folium.GeoJson(floodLayerList[i], show=False,
                   style_function=style_function,
                   tooltip=folium.GeoJsonTooltip(
                   fields=['raster_val'],
                   aliases=['수위(m) :'],
                   localize=True,
                   style=('font-size: 24px')),
               name='침수심 (+%sm)' % round(alllist[i], 2)).add_to(base)


BUILDING_J = json.load(open(finpath + sidoCode + '/jsonResult' +'/' + '6_' + baseJSON + "BUILDING.json"))

style_function = lambda x: {'color':'red',
                            'fillColor':"red",
                            'opacity': 1,
                            'fillOpacity':1}

buildingtomap = folium.GeoJson(BUILDING_J, style_function = style_function, show=False).add_to(base)

buildingtomap.add_child(folium.features.GeoJsonTooltip(fields=['POP'],
                                                       aliases=['인구(명):']))
buildingtomap.layer_name = '건물'

tmp_base = folium.FeatureGroup(name='POI').add_to(base)
icon_red = folium.Icon(color='red')
tip = """<b>관심 지점: </b> %sm""" % (round(poi_dem, 3))
ttip = folium.Tooltip(tip, style = (" font-size: 24px;"))
folium.Marker((float(pointGDF['geometry'].y), float(pointGDF['geometry'].x)),
                tooltip=ttip,
                icon=icon_red).add_to(tmp_base)


folium.LayerControl(collapsed=False).add_to(base)

base.save(finpath + sidoCode + '/' + '0_' + baseJSON + 'MAP.html')

print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')
