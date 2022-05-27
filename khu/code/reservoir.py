from email.mime import base
import geopandas as gpd
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import time
import rasterio
import rasterstats
import descartes
import folium
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

starttime = time.time()

import sys
sidoCode = sys.argv[1]
encod  = 'UTF8'

from datetime import datetime
current_time = datetime.now()
dateandtime = current_time.strftime("%Y%m%dT%H%M")

dtype = "TYPHN"
mtype = "RESER"
sep = '_'
baseJSON = mtype+sep+dateandtime+sep+dtype+sep

import os
os.chdir('../')
filedir = './inputData/' # input file directory
finpath = './outputData/reservoir/' # output file directory

 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
createFolder(finpath + sidoCode)


RES = gpd.read_file(filedir + 'reservoir/totalReservoir.shp', encoding = 'cp949')  # EPSG 5179, 저수지 shapefile

BUILD = gpd.read_file(filedir+'Common/build_all/build_all_' + sidoCode + '.shp')

rf = rasterio.open(filedir+ 'Common/dem/dem_' + sidoCode + '.tif', mode = 'r')  # DEM 자료

# assign raster values to a numpy nd array
rf_array = rf.read(1)
affine = rf.transform

RES = RES.to_crs({'init' :'epsg:5179'})

Range = 500
Buffer = RES.buffer(int(Range))
BUFFER = gpd.GeoDataFrame(geometry = Buffer)

BUFFER.crs = {'init': 'epsg:5179'}

RES2 = RES.drop(['geometry'], axis = 'columns')
RANGE = pd.concat([RES2, BUFFER], axis = 1) 
RANGE['NUM'] = RANGE.index
RANGE_e = RANGE.loc[:, ['NUM', 'NAME_1', 'geometry']]

TJBUILD = gpd.sjoin(BUILD, RANGE_e, how = 'left', op = 'intersects').drop(['index_right'], axis = 1)
exBuild = TJBUILD[~TJBUILD.NUM.isnull()]
exbuild = exBuild.reset_index(drop = True)

exRANGE = RANGE_e.loc[RANGE_e['NUM'].isin(list(np.unique(exBuild.NUM))), :]
exRange = exRANGE.reset_index(drop = True)

exRES = RES.iloc[list(np.unique(exBuild.NUM)), :].reset_index(drop = True).loc[:, ['NAME_1', 'geometry']]

average_rf = rasterstats.zonal_stats(exbuild, rf_array, affine = affine, stats = ['mean'], geojson_out = False)
average_RES = rasterstats.zonal_stats(exRES, rf_array, affine = affine, stats = ['max'], geojson_out = False)

exRange['ele'] = pd.DataFrame(average_RES).iloc[:, 0]
exbuild['ele'] = pd.DataFrame(average_rf).fillna(0).iloc[:, 0]

reslist = []
rangelist = []
resultlist = []

import random
def hextofloats(h):
    return list(int(h[i:i + 2], 16) for i in (1, 3, 5))

    
numlist = list(range(100))
    
for i in range(len(exRange)):
    name = exRange.iloc[i, :]['NAME_1']
    overRange = exbuild.loc[exbuild.NAME_1 == name, :]
    overEx = overRange[overRange['ele'] <= exRange.iloc[i, :]['ele']].reset_index(drop = True)
    
    random_number = random.randint(0,16777215)
    hex_number = str(hex(random_number))
    rancol ='#'+ hex_number[2:]
    

    bldcol = {'stroke': {'color': [0, 0, 0, 0], 'width':0}, "fill": {'color': hextofloats(rancol)+[1]}}
    rangecol = {'stroke': {'color': [0, 0, 0, 0], 'width':0}, "fill": {'color': hextofloats(rancol)+[0.2]}}
    rescol = {'stroke': {'color': hextofloats(rancol)+[1], 'width':1}, "fill": {'color': hextofloats(rancol)+[0.7]}}

    overEx['style'] = [bldcol for x in range(len(overEx))]
    
    rangelist.append(rangecol)
    reslist.append(rescol)

    if len(overEx) != 0:
        Pover = overEx.to_crs({'init' :'epsg:4326'})
        resultlist.append(Pover)
 
for i in range(len(resultlist)):
    resultlist[i].to_file(finpath + sidoCode + '/' + str(i+3) + '_' + baseJSON + 'BUILDING'+ str(i) + '.json', driver = 'GeoJSON', encoding = encod)
    
exRange['style'] = rangelist
exRES['style'] = reslist
wgs = {'init' :'epsg:4326'}
P_exRange = exRange.to_crs(wgs)
P_exRES = exRES.to_crs(wgs)


P_exRange.to_file(finpath + sidoCode + '/'+ '1_' + baseJSON + 'RANGE.json', driver = 'GeoJSON', encoding = encod)
P_exRES.to_file(finpath + sidoCode + '/' '2_' + baseJSON + 'RESERVOIR.json', driver = 'GeoJSON', encoding = encod)

print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')