import geopandas as gpd
import pandas as pd
import os
import folium
import json
import time

import warnings
warnings.filterwarnings("ignore")

starttime = time.time()

import sys
typcode = sys.argv[1]

from datetime import datetime
current_time = datetime.now()
dateandtime = current_time.strftime("%Y%m%dT%H%M")

dtype = "TYPHN"
mtype = "TYPHN"
sep = '_'
baseJSON = mtype+sep+dateandtime+sep+dtype+sep

os.chdir('../')

filedir = './inputData/' # input file directory

SIGUNGU = gpd.read_file(filedir + 'Common/sigunguBound/sigunguBound_100.shp')

total_line = gpd.read_file(filedir + 'typhoonInfo/route/route.shp')
dbdir = filedir + 'typhoonInfo/damage/'
now_line = total_line[total_line['re_number'] == typcode].reset_index(drop=True)

utmk = {'init' :'epsg:5179'}
now_line.crs = utmk
now_line.geometry[0] = now_line.geometry.buffer(25000)[0]


tmp_list1 = os.listdir(dbdir)
tmp_list1 = [file[0:4] for file in tmp_list1]

total_line = total_line[total_line['re_number'].isin(tmp_list1)]

total_line.crs = utmk

def clip(road, danger):
    poly = danger.geometry.unary_union
    spatial_index = road.sindex

    bbox = poly.bounds
    sidx = list(spatial_index.intersection(bbox))  
    shp_sub = road.iloc[sidx]

    clipped = shp_sub.copy()
    clipped['geometry'] = shp_sub.intersection(poly)
    final_clipped = clipped[~clipped.is_empty]
    return final_clipped

final = clip(total_line, now_line)

final["length"] = final["geometry"].length

final_df = gpd.GeoDataFrame(final)
final_df_sort = final_df.sort_values(by=['length'], ascending=False)

result = final_df_sort.iloc[1:4, ]

result = result.reset_index(drop=True)

df = pd.DataFrame()
for i in range(len(result)):
    Line = total_line[total_line.OBJECTID.isin([result['OBJECTID'][i]])]
    df = df.append(Line)

color = ['red', 'green', 'yellow']
df['color'] = color
df = df.reset_index(drop=True)


num_tp = list(df['re_number'])

tmp_list = os.listdir(dbdir)

shp_list = dict(zip(num_tp, [[],[],[]]))
tb_list = dict(zip(num_tp, [[],[],[]]))

SIGUNGU.crs = {'init' :'epsg:5179'}
SIGUNGU = SIGUNGU.to_crs({'init' :'epsg:4326'})

SIGUNGU = SIGUNGU.to_crs({'init' :'epsg:5179'})

for i in range(len(num_tp)) :
    for j in range(len(tmp_list)) :
        if num_tp[i] in tmp_list[j]:
            db = pd.read_excel(dbdir + "/" + tmp_list[j], sheet_name='피해목록', engine='openpyxl', skiprows=[0])
            db.dropna(inplace = True)
            gdb = db.copy()
            gdb.dropna(inplace = True)
            gdb = gdb.astype({"CODE":'int'})
            gdb = gdb.astype({"CODE":'str'})
            J_gdb = gpd.GeoDataFrame(pd.merge(SIGUNGU, gdb, left_on='SIGUNGU_CD', right_on='CODE'))
            tb_list[num_tp[i]].append(db)
            shp_list[num_tp[i]].append(J_gdb)

col_list = ['총 이재민 세대', '총 이재민 명', '인명피해 계', '건물피해 계', '건물피해액',
            '선박피해 계(척)', '선박피해액', '농경지피해액', '농작물피해 계', '공공시설피해액소계', '사육시설피해액소계', '총 피해액']

tmp = pd.DataFrame(columns = ['re_number'] + col_list)

for i in range(len(num_tp)) :
    val = list(tb_list[num_tp[i]][0][tb_list[num_tp[i]][0]['지역명'].str.contains("총계")][col_list].sum())
    tmp = tmp.append(pd.Series([list(tb_list.keys())[i]] + val, index = tmp.columns), ignore_index=True)

total_d_line = gpd.GeoDataFrame(pd.merge(df, tmp, on = 're_number'))

total_d_line.crs = {'init' :'epsg:5179'}

total_d_line_p = total_d_line.to_crs({'init' :'epsg:4326'})

colormap = [[255, 0, 0, 1], [0, 128, 0, 1], [255, 255, 0, 1]]
total_d_line_p['style'] = [{'stroke': {'color': col, 'width':5}} for col in colormap]

finpath = './outputData/typhoonInfo/'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
createFolder(finpath + typcode)

total_d_line_p.to_file(finpath + typcode + '/'+ '2_' + baseJSON + 'PROUTE.json', driver = 'GeoJSON')


shp_list[num_tp[0]][0].crs = {'init' :'epsg:5179'}
shp_list[num_tp[1]][0].crs = {'init' :'epsg:5179'}
shp_list[num_tp[2]][0].crs = {'init' :'epsg:5179'}

shp_list1 = shp_list[num_tp[0]][0].to_crs({'init' :'epsg:4326'})
shp_list2 = shp_list[num_tp[1]][0].to_crs({'init' :'epsg:4326'})
shp_list3 = shp_list[num_tp[2]][0].to_crs({'init' :'epsg:4326'})

col1 = {'stroke': {'color': [255, 0, 0, 1]}, "fill": {'color': [255, 0, 0, 0.2]}}
col2 = {'stroke': {'color': [0, 128, 0, 1]}, "fill": {'color': [0, 128, 0, 0.2]}}
col3 = {'stroke': {'color': [255, 255, 0, 1]}, "fill": {'color': [255, 255, 0, 0.2]}}

shp_list1['style'] = [col1 for i in range(len(shp_list1))]
shp_list2['style'] = [col2 for i in range(len(shp_list2))]
shp_list3['style'] = [col3 for i in range(len(shp_list3))]

shp_list1.to_file(finpath + typcode + '/'+ '3_' + baseJSON + 'NO1DMG.json', driver = 'GeoJSON')
shp_list2.to_file(finpath +  typcode + '/'+ '4_' + baseJSON + 'NO2DMG.json', driver = 'GeoJSON')
shp_list3.to_file(finpath +  typcode + '/'+ '5_' + baseJSON + 'NO3DMG.json', driver = 'GeoJSON')



now_line_p = now_line.to_crs({'init' :'epsg:4326'})
now_line_p['style'] = [{'stroke': {'color': [0, 0, 0, 1]}, "fill": {'color': [0, 0, 0, 0.2]}}]

now_line_p.to_file(finpath + typcode + '/'+ '1_' + baseJSON + 'CUROUTE.json', driver = 'GeoJSON')

total_j = json.load(open(finpath + typcode + '/'+ '2_' + baseJSON + 'PROUTE.json', 'rt', encoding='cp949'))

shp1_j = json.load(open(finpath + typcode + '/'+ '3_' + baseJSON + 'NO1DMG.json', 'rt', encoding='cp949'))
shp2_j = json.load(open(finpath + typcode + '/'+ '4_' + baseJSON + 'NO2DMG.json', 'rt', encoding='cp949'))
shp3_j = json.load(open(finpath + typcode + '/'+ '5_' + baseJSON + 'NO3DMG.json', 'rt', encoding='cp949'))

now_j = json.load(open(finpath + typcode + '/'+ '1_' + baseJSON + 'CUROUTE.json', 'rt', encoding='cp949'))

center = [37.5, 127.0]
base = folium.Map(location = center, zoom_start=5)

style_function = lambda x: {'color':x['properties']['color'], 'weight':5}
style_function2 = lambda x: {'fillColor':'red', 'color':'red', 'fillOpacity':0.2}
style_function3 = lambda x: {'fillColor':'green', 'color':'green', 'fillOpacity':0.2}
style_function4 = lambda x: {'fillColor':'yellow', 'color':'yellow', 'fillOpacity':0.2}
style_function0 = lambda x: {'fillColor': '#000000', 'color': '#000000'}

route1 = folium.features.GeoJson(total_j['features'][0], style_function=style_function, show=False).add_to(base)
route2 = folium.features.GeoJson(total_j['features'][1], style_function=style_function, show=False).add_to(base)
route3 = folium.features.GeoJson(total_j['features'][2], style_function=style_function, show=False).add_to(base)

nowline = folium.features.GeoJson(now_j, style_function = style_function0).add_to(base)

shp1 = folium.GeoJson(shp1_j, style_function=style_function2, show=False).add_to(base)
shp2 = folium.GeoJson(shp2_j, style_function=style_function3, show=False).add_to(base)
shp3 = folium.GeoJson(shp3_j, style_function=style_function4, show=False).add_to(base)

nowline.layer_name = '현재경로'
route1.layer_name ='1순위'
shp1.layer_name = '1순위 태풍 피해지역'
route2.layer_name ='2순위'
shp2.layer_name = '2순위 태풍 피해지역'
route3.layer_name ='3순위'
shp3.layer_name = '3순위 태풍 피해지역'

field_nm = ['re_number','총 피해액', '총 이재민 명', '인명피해 계', '건물피해 계', '선박피해 계(척)', 
            '농경지피해액', '농작물피해 계', '공공시설피해액소계', '사육시설피해액소계']
al_nm = ['태풍번호:', '총 피해액(원):', '피해 이재민 수(명):', '인명피해(명):', '건물피해(개):', '선박피해(척):', '농경지피해액(원):', 
         '농작물피해(건):', '공공시설피해액(원):', '사육시설피해액(원):']

route1.add_child(folium.features.GeoJsonTooltip(fields= field_nm, aliases=al_nm, sticky=True))

route2.add_child(folium.features.GeoJsonTooltip(fields= field_nm, aliases=al_nm, sticky=True))

route3.add_child(folium.features.GeoJsonTooltip(fields= field_nm, aliases=al_nm, sticky=True))

field_nm_poly = ['SIGUNGU_NM', '총 피해액', '총 이재민 명', '인명피해 계', '건물피해 계', '선박피해 계(척)',
            '농경지피해액', '농작물피해 계', '공공시설피해액소계', '사육시설피해액소계']
al_nm_poly = ['행정구역명: ', '총 피해액(원):', '피해 이재민 수(명):', '인명피해(명):', '건물피해(개):', '선박피해(척):', '농경지피해액(원):',
         '농작물피해(건):', '공공시설피해액(원):', '사육시설피해액(원):']

shp1.add_child(folium.features.GeoJsonTooltip(fields= field_nm_poly, aliases=al_nm_poly, sticky=False))

shp2.add_child(folium.features.GeoJsonTooltip(fields= field_nm_poly, aliases=al_nm_poly, sticky=False))

shp3.add_child(folium.features.GeoJsonTooltip(fields= field_nm_poly, aliases=al_nm_poly, sticky=False))

base.add_child(folium.map.LayerControl())
base.save(finpath + typcode + '/' + '0_' + baseJSON + 'MAP.html')

print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')