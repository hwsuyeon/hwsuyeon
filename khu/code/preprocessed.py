import geopandas as gpd
import os
import time
import warnings
warnings.filterwarnings("ignore")

starttime = time.time()

import sys
damagefile = sys.argv[1]

os.chdir('../')
filedir = './inputData/Common/damagePolygon' # input file directory

encod = 'cp949'
bound = gpd.read_file(filedir + '/simpleboundary.shp', encoding = encod) 
flood = gpd.read_file(filedir + '/raw/' + damagefile + '.shp', encoding = encod)
crs = {'init' :'epsg:5179'}
flood = flood.to_crs(crs)
floodG = gpd.GeoDataFrame(index = range(len(flood)), geometry = flood['geometry'])
dissolved = gpd.sjoin(floodG, bound).dissolve('시군코드')

dissolved['CODE'] = dissolved.index
dissolved = dissolved.reset_index(drop = True)

def createfile(idx, geom, path):
    gdf = gpd.GeoDataFrame({"CODE": [idx]}, crs = {'init' :'epsg:5179'}, geometry = [geom])
    gdf.to_file(path + '/processed/' + str(idx) + '/' + damagefile +'.shp', encoding = encod)

dissolved.apply(lambda row: createfile(row['CODE'], row['geometry'], filedir),axis=1)


print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')