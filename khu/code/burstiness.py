#-*- coding:utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from dateutil.parser import parse
import branca.colormap as cm
import os
import time
import json
import sys

starttime = time.time()

sidoCode = sys.argv[1]

sidoCode = str(sidoCode)
encod  = 'cp949'

from datetime import datetime
current_time = datetime.now()
dateandtime = current_time.strftime("%Y%m%dT%H%M")

dtype = "FORFI"
mtype = "BURST"
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
# path

os.chdir('../')
inpath = './inputData/' 

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 

finpath = './outputData/burstiness/' # output file directory

createFolder(finpath + sidoCode)

finpath = './outputData/burstiness/' + sidoCode + '/'

combination = gpd.read_file(inpath + 'burstiness/kfs_origin_buffer.shp', encoding = 'cp949') 

combination['startTime'] = [parse(str(combination.loc[i, 'start_y'])
                                  + str(combination.loc[i, 'start_m']).zfill(2)
                                  + str(combination.loc[i, 'start_d']).zfill(2)
                                  + str(combination.loc[i, 'start_t'].zfill(5)[:2])
                                  + str(combination.loc[i, 'start_t'].zfill(5)[3:5])) for i in range(len(combination))]

combination['endTime'] = [parse(str(combination.loc[i, 'end_y'])
                                + str(combination.loc[i, 'end_m']).zfill(2)
                                + str(combination.loc[i, 'end_d']).zfill(2)
                                + str(combination.loc[i, 'end_t'].zfill(5)[:2])
                                + str(combination.loc[i, 'end_t'].zfill(5)[3:5])) for i in range(len(combination))]


combination.sort_values(by = ['startTime'], axis = 0, ascending = True, inplace = True)
combination.reset_index(drop = True, inplace = True) 


def getInterEventTimes(events):
 
    events.sort_values(by = ['startTime'], axis = 0, ascending = True, inplace = True)
    events.reset_index(drop = True, inplace = True) 
    a = []
    for i in range(len(events)-1):
        time2 = events.startTime[i+1]
        time1 = events.startTime[i]

        if i != len(events):
            tau = time2 - time1 
            a.append(tau)
        else:
            pass
        i += 1
    return a


def burstinessMeasure(n, Tau):
    if len(Tau) < 2:
        pass
    else:
        aa = pd.DataFrame(Tau, columns=['value'])
        std = aa.value.std()
        mean = aa.value.mean() 

        if mean == 0 or std == 0:
            pass
        else:
            r = std / mean
            if (np.sqrt(n+1)-2) * r + np.sqrt(n-1) is not 0 and r <= np.sqrt(n-1):
                A = (np.sqrt(n+1) * r - np.sqrt(n-1)) / ((np.sqrt(n+1)-2) * r + np.sqrt(n-1))
                return A
            else:
                pass


def bootstrapping(n, interEventTime, testValue, iterationNumber):
    import random
    arr_As = []
    for i in range(iterationNumber):
        choice_a = [random.choice(interEventTime) for i in range(len(interEventTime))]
        A_prime = burstinessMeasure(n, choice_a)
        arr_As.append(A_prime)
        i = i+1
    
    # confidence level(95%)
    arr_As.sort(reverse = True)
    upperEnvelope = arr_As[int(iterationNumber * 0.025)]
    lowerEnvelope = arr_As[int(np.ceil(iterationNumber * 0.975))]

    if testValue >= lowerEnvelope and testValue <= upperEnvelope:
        return [0.05, testValue, upperEnvelope, lowerEnvelope]
    else:
        pass


def LITB(events, codeColumn, adm, iterationNumber):
    allCode = np.unique(events[str(codeColumn)])
    admCode = [code for code in allCode if str(adm) in code[0:2]]
    arr = []
    for i in admCode:
        each_unit = events.loc[events['adm_dr_cd'] == str(i)]
        each_unit = each_unit.drop_duplicates('startTime')
        if len(each_unit) >= 3:
            interEventTime = getInterEventTimes(each_unit)
            bustyIndex= burstinessMeasure(len(each_unit), interEventTime)
            result = bootstrapping(len(each_unit), interEventTime, bustyIndex, iterationNumber)
            result.append(i)
            arr.append(result)
        else:
            arr.append([0, 0, 0, 0, i])
    
    return arr



d_sgis = [['서울특별시', 11], ['부산광역시', 21], ['대구광역시', 22], ['인천광역시', 23], ['광주광역시', 24], ['대전광역시', 25],
    ['울산광역시', 26], ['세종특별자치시', 29], ['경기도', 31], ['강원도', 32], ['충청북도', 33], ['충청남도', 34], ['전라북도', 35],
    ['전라남도', 36], ['경상북도', 37], ['경상남도', 38], ['제주특별자치도', 39]]

d_bjd = [['서울특별시', 11], ['부산광역시', 26], ['대구광역시', 27], ['인천광역시', 28], ['광주광역시', 29], ['대전광역시', 30],
    ['울산광역시', 31], ['세종특별자치시', 36], ['경기도', 41], ['강원도', 42], ['충청북도', 43], ['충청남도', 44], ['전라북도', 45],
    ['전라남도', 46], ['경상북도', 47], ['경상남도', 48], ['제주특별자치도', 50]]


dct = dict(zip([str(x) for x in dict(d_bjd).values()], [str(x) for x in dict(d_sgis).values()]))
data = LITB(combination, 'adm_dr_cd', dct[sidoCode], 100)

df = pd.DataFrame(columns=['slevel', 'burstiness', 'upper', 'lower', 'adm_dr_cd'])

for i in range(len(data)) :
    df = df.append(pd.Series(data[i], index=df.columns), ignore_index=True)

df.sort_values(by = ['burstiness'], axis = 0, ascending = False, inplace = True)
df.reset_index(drop = True, inplace = True) 

admregions = gpd.read_file(inpath + 'burstiness/New Folder/bnd_dong_00_2020_2020_2Q.shp', encoding = 'cp949')

df_area = pd.merge(admregions, df, left_on="ADM_DR_CD", right_on="adm_dr_cd")
gdf_area = gpd.GeoDataFrame(df_area)
gdf_area.crs = {'init' :'epsg:5179'}
gdf_area = gdf_area.to_crs({'init': 'epsg:4326'})
gdf_area["index"] = gdf_area.index
geoPath = gdf_area.to_json()

from branca.colormap import linear

def hextofloats(h):
    return list(int(h[i:i + 2], 16) for i in (1, 3, 5))

color_scale = cm.linear.RdBu_09.scale(gdf_area.burstiness.min(), gdf_area.burstiness.max()).colors
color_scale.reverse()
custom_scale = cm.LinearColormap(colors=color_scale).scale(gdf_area.burstiness.min(), gdf_area.burstiness.max()).to_step(10) 

collist2 = [{'stroke': {'color': [0, 0, 0, 0], 'width':0.5}, "fill": {'color': hextofloats(custom_scale(bursty))+[0.8]}} for bursty in gdf_area['burstiness']]
gdf_area['style'] = collist2
gdf_area = gdf_area.loc[:, ['ADM_DR_CD', 'burstiness', 'style', 'geometry']]

gdf_area.to_file(finpath + '1_' + baseJSON + 'BURSTY' + sidoCode + '.json', driver = 'GeoJSON', encoding=encod)

combination["index"] = combination.index
comb_select = combination[combination['adm_dr_cd'].str.startswith(dct[sidoCode])].reset_index(drop = True)
freq = pd.DataFrame(comb_select.groupby("adm_dr_cd").count()['index'])
freq['ADM_DR_CD'] = freq.index
freq = freq.reset_index(drop= True)
df_freq = pd.merge(admregions, freq, on="ADM_DR_CD")
gdf_freq = gpd.GeoDataFrame(df_freq)
gdf_freq.crs = {'init' :'epsg:5179'}
gdf_freq = gdf_freq.to_crs({'init': 'epsg:4326'})
gdf_freq.rename(columns = {'index' : 'freq'}, inplace = True)
gdf_freq['index'] = gdf_freq.index
freq_json = gdf_freq.to_json()

color_scale2 = cm.linear.Reds_09.scale(gdf_freq.freq.min(), gdf_freq.freq.max()).colors
custom_scale2 = cm.LinearColormap(colors=color_scale2).scale(gdf_freq.freq.min(), gdf_freq.freq.max()).to_step(10) 

collist2 = [{'stroke': {'color': [0, 0, 0, 0], 'width':0.5}, "fill": {'color': hextofloats(custom_scale2(freq))+[0.8]}} for freq in gdf_freq['freq']]
gdf_freq['style'] = collist2
gdf_freq = gdf_freq.loc[:, ['ADM_DR_CD', 'freq', 'style', 'geometry']]

gdf_area.to_file(finpath + '2_' + baseJSON + 'FREQ' + sidoCode + '.json', driver = 'GeoJSON', encoding=encod)


newList = []
def switchLatLng(geos):
    for geo in geos:
        if isinstance(geo[0], list):
            switchLatLng(geo)
        else:
            newList.append([geo[1], geo[0]])
    return newList

import folium.plugins
import folium

center = list(reversed(list(gdf_area['geometry'].centroid[0].coords[0])))

m = folium.plugins.DualMap(location=center, zoom_start = 8, tiles='stamenterrain')

x1,y1,x2,y2 = gdf_area['geometry'].total_bounds
m.fit_bounds([[y1, x1], [y2, x2]])

color_scale = cm.linear.RdBu_09.scale(gdf_area.burstiness.min(), gdf_area.burstiness.max()).colors
color_scale.reverse()
custom_scale = cm.LinearColormap(colors=color_scale).scale(gdf_area.burstiness.min(), gdf_area.burstiness.max()).to_step(10) 
custom_scale.caption = 'Burstiness'

style_function = lambda x: {'color':'black',
                            'fillColor':custom_scale(x['properties']['burstiness']),
                            'weight':0.5,
                            'fillOpacity':0.8}

layer_1 = folium.features.GeoJson(geoPath, style_function=style_function, control = False)


tooltip_1= folium.features.GeoJsonTooltip(['ADM_DR_NM', 'burstiness'], aliases= ["발생 지역(읍면동): ", "Burstiness: "], 
                                          style = (" font-size: 30px;"), sticky=False)
layer_1.add_child(tooltip_1)


color_scale2 = cm.linear.Reds_09.scale(gdf_freq.freq.min(), gdf_freq.freq.max()).colors
custom_scale2 = cm.LinearColormap(colors=color_scale2).scale(gdf_freq.freq.min(), gdf_freq.freq.max()).to_step(10) 
custom_scale2.caption = 'Frequency'

style_function = lambda x: {'color':'black',
                            'fillColor':custom_scale2(x['properties']['freq']),
                            'weight':0.5,
                            'fillOpacity':0.8}

layer_2 = folium.features.GeoJson(freq_json, style_function=style_function, control = False)


tooltip_2= folium.features.GeoJsonTooltip(['ADM_DR_NM', 'freq'], aliases= ["발생 지역(읍면동): ", "산불 발생 빈도: "], 
                                          style = (" font-size: 15px;"), sticky=False)
layer_2.add_child(tooltip_2)


combination = combination.to_crs({'init': 'epsg:4326'})
combination["index"] = combination.index
comb_select = combination[combination['adm_dr_cd'].str.startswith(dct[sidoCode])].reset_index(drop = True)
comb_select['startTime'] = comb_select['startTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
comb_select['endTime'] = comb_select['endTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

layer_3 = folium.plugins.MarkerCluster()

for i, v in comb_select.iterrows():
    popup = """
    <b>발생지역(읍면동):</b> %s<br>
    <b>피해면적:</b> %s ha<br>
    """ % (v['adm_dr_nm'], v['ha'])
    tool = folium.Tooltip(popup, style = (" font-size: 15px;"), sticky=True)
    lat, lon = list(reversed(list(v['geometry'].centroid.coords[0])))
    folium.CircleMarker(location = [lat, lon], radius = v['radius'], 
    tooltip = tool, color='red', fill_color='red', fill=True).add_to(layer_3)


custom_scale.add_to(m.m1)
custom_scale2.add_to(m.m2)

fg_1 = folium.FeatureGroup(name='Burstiness Index').add_to(m.m1)
fg_2 = folium.FeatureGroup(name='누적산불발생빈도', show = False).add_to(m.m1)
fg_3 = folium.FeatureGroup(name='산불발생 위치').add_to(m.m2)

layer_1.add_to(fg_1)
layer_2.add_to(fg_2)
layer_3.add_to(fg_3)

fg_1.add_to(m.m1)
fg_2.add_to(m.m1)
fg_3.add_to(m.m2)

folium.LayerControl(collapsed=True).add_to(m)

m.save(finpath + '0_' + baseJSON + 'MAP' + sidoCode + '.html')

print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')