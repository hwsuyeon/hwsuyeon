# -*- coding: utf-8 -*-

import geopandas as gpd
import pandas as pd
import pyproj
import numpy as np
import os
import folium
import json
import time
from branca.colormap import linear
from folium import plugins

import warnings
warnings.filterwarnings("ignore")

starttime = time.time()

import sys
damagefile = sys.argv[1]
sidoCode = sys.argv[2]

Range = "500"
encod  = 'UTF8'

date, dtype = damagefile.split("_")[1], damagefile.split("_")[2]
mtype = "DMGDA"
sep = '_'
baseJSON = mtype+sep+date+sep+dtype+sep

os.chdir('../')
filedir = './inputData/' # input file directory
filedirfinal = './outputData/damage/damageDay/' # output file directory

sidoCode = str(sidoCode)

one = gpd.read_file(filedir + 'Common/damagePolygon/processed/' + sidoCode + '/' + damagefile + '.shp') 
Pone = one.to_crs({'init' :'epsg:5179'})
OVER = gpd.GeoDataFrame(index = [0], crs = {'init' :'epsg:5179'}, geometry = [Pone.unary_union])


SHP = gpd.read_file(filedir+'Common/build_all/build_all_' + sidoCode + '.shp')
SHP['OBJECTID'] = range(1, len(SHP)+1)
SHP.rename(columns = {'A14' : 'TOTALAREA', 'A8':'USABILITY', 'A10': 'STRCT_CD'}, inplace = True)

SHP = SHP[SHP.geometry != None]
SHP = SHP[~SHP.USABILITY.isnull()]
SHP = SHP.reset_index(drop = True)

JIPGYE = gpd.read_file(filedir + 'Common/jipgye_bound/jipgye_bound_' + sidoCode + '.shp')
test = pd.read_csv(filedir + 'Common/population/' + sidoCode + '_2020년_성연령별인구.txt', sep='^')
struct = pd.read_csv(filedir+'damage/buildingID.csv')
merge = gpd.read_file(filedir+'damage/land_use/land_use_' + sidoCode + '.shp') 
economic = pd.read_csv(filedir + 'damage/economic_ratio.csv', encoding = 'cp949')
eco = economic.loc[economic['CODE']==int(sidoCode), 'RATIO'].values[0] 
eco = eco * 0.01

OVER.crs = {'init' :'epsg:5179'}
SHP.crs = {'init' :'epsg:5179'}
JIPGYE.crs = {'init' :'epsg:5179'}
merge.crs = {'init' :'epsg:5179'}

Buffer = OVER.buffer(int(Range))
BUFFER = gpd.GeoDataFrame(geometry=Buffer)
BUFFER.crs = {'init': 'epsg:5179'}

crs = {'init' :'epsg:4326'}

P_BUFFER = BUFFER.to_crs(crs)
OVER1 = OVER.to_crs(crs)
OVER1.geometry = OVER1.buffer(0)
SHP = SHP.to_crs(crs)
JIPGYE = JIPGYE.to_crs(crs)
merge = merge.to_crs(crs)

TJSHP = SHP
TJSHP = gpd.sjoin(TJSHP, JIPGYE, how = 'left', op='within').drop(['index_right', 'BASE_DATE', 'ADM_CD'], axis = 1)
TJSHP = TJSHP.rename({'TOT_REG_CD':'code'}, axis='columns')

TJSHP['rescode'] = np.nan
Ls = ['01000', '02000']
for i in range(len(TJSHP)):
    if TJSHP['USABILITY'][i] in Ls:
        TJSHP['rescode'][i] = 'RES'
    else:
        TJSHP['rescode'][i] = "NoRES"

area2 = pd.DataFrame(TJSHP['TOTALAREA'].groupby([TJSHP['code'], TJSHP['rescode']]).sum().reset_index())
area2 = area2.rename({'TOTALAREA':'J_TOTALAREA'}, axis='columns')
area2['code2'] = area2['code'] + area2['rescode']
TJSHP['code2'] = TJSHP['code'] + TJSHP['rescode']

area2 = area2[['J_TOTALAREA', 'code2']]
TTJSHP = pd.merge(TJSHP, area2, on = 'code2')

tots = np.unique(test['tot_oa_cd']).tolist()


a = range(1, 4, 1)
b = range(4, 14, 1)
c = range(14, 22, 1)

num = list(range(1, 10, 1))
df = pd.DataFrame(columns=[str(x) for x in num])
test = test.fillna({"value":4})

for i in range(len(tots)):
    test1 = test.loc[test['tot_oa_cd'] == tots[i], ['item', 'value']] 
    test1['grp'] = test1.item.str.split('_0').str[1]
    test1 = test1.fillna(0)
    test1['grp'] = test1['grp'].astype(int)  
    aa = test1.loc[test1['grp'].isin(a), ['value']].sum().values.tolist()
    bb = test1.loc[test1['grp'].isin(b), ['value']].sum().values.tolist() 
    cc = test1.loc[test1['grp'].isin(c), ['value']].sum().values.tolist()
    dd = aa + bb + cc
    ee = [sum(dd)]
    if ee[0] != 0 :
        ff = aa[0] / ee[0]
        gg = bb[0] / ee[0]
        hh = cc[0] / ee[0]
    else:
        ff = 0
        gg = 0
        hh = 0
    jj = [int(ee[0] * eco)]
    ii = [str(tots[i])] + aa + bb + cc + [ff] + [gg] + [hh] + ee +jj
    numb = list(range(1, 10, 1))
    test2 = pd.DataFrame(ii, index=[str(x) for x in numb])
    test3 = test2.T
    df = df.append(test3)
    i = i+1
    

df.rename(columns = {df.columns[0]:'code'}, inplace = True)
df.rename(columns = {df.columns[1]:'youth'}, inplace = True)
df.rename(columns = {df.columns[2]:'young'}, inplace = True)
df.rename(columns = {df.columns[3]:'old'}, inplace = True)
df.rename(columns = {df.columns[4]:'youth(%)'}, inplace = True)
df.rename(columns = {df.columns[5]:'young(%)'}, inplace = True)
df.rename(columns = {df.columns[6]:'old(%)'}, inplace = True)
df.rename(columns = {df.columns[7]:'allPop'}, inplace = True)
df.rename(columns = {df.columns[8]:'ecoPop'}, inplace = True)


df = df.reset_index(drop=True)
df.to_csv(filedir + 'evacuation/pop_' + sidoCode +'.csv', encoding = encod, index = False)

TJTJSHP = pd.merge(TTJSHP, df, on = 'code')
TJTJSHP['POP'] = np.nan  

Ls = ['01000', '02000']
for i in range(len(TJTJSHP)):
    if TJTJSHP['USABILITY'][i] in Ls:
        TJTJSHP['POP'][i] = TJTJSHP['allPop'].iloc[i] * TJTJSHP['TOTALAREA'].iloc[i]/TJTJSHP['J_TOTALAREA'].iloc[i]
    else:
        TJTJSHP['POP'][i] = TJTJSHP['ecoPop'].iloc[i] * TJTJSHP['TOTALAREA'].iloc[i]/TJTJSHP['J_TOTALAREA'].iloc[i]

AFTJTJSHP = TJTJSHP

AFTJTJSHP['POP'] = AFTJTJSHP['POP'].astype(int)
AFTJTJSHP.loc[AFTJTJSHP["POP"] == 0,"POP"] = 1
SJAFTJTJSHP = AFTJTJSHP


SJAFTJTJSHP['USE']=np.nan
SJAFTJTJSHP['USE']=SJAFTJTJSHP['USABILITY']
SJAFTJTJSHP.loc[SJAFTJTJSHP['USE'] == '01000', 'USE'] = '단독주택'
SJAFTJTJSHP.loc[SJAFTJTJSHP['USE'] == '02000', 'USE'] = '공동주택'

struct.rename(columns = {struct.columns[0]:'STRCT_CD'}, inplace = True)
struct['STRCT_CD'] = struct['STRCT_CD'].astype(str)

SAFSJAFTJTJSHP = pd.merge(SJAFTJTJSHP, struct, on = 'STRCT_CD', how = 'left')

GSAFSJAFTJTJSHP = SAFSJAFTJTJSHP['geometry']
GGOVER1 = OVER1['geometry']
SAFSJAFTJTJSHP['overflow'] = np.nan

for i in range(len(SAFSJAFTJTJSHP)):
    if GGOVER1.iloc[0].intersects(GSAFSJAFTJTJSHP.iloc[i]) == True:
        SAFSJAFTJTJSHP['overflow'].iloc[i] = 1
        i = i+1
    else:
        i = i+1


Agri_I = merge[merge['LV2_CODE'].isin(["210", "220", "230", "240", "250"])]

GAgri_I = Agri_I['geometry']
GOVER = OVER1['geometry']
idx = []
TmpList = []
for i in range(len(GAgri_I)):
    if GOVER.iloc[0].intersects(GAgri_I.iloc[i]) == True:
        idx.append(Agri_I.index[i])

for i in Agri_I.index:
    if i in idx:
        TmpList.append(1)
        i 
    else:
        TmpList.append('NA')
        
SJAgri_I = Agri_I
SJAgri_I['Overflow'] = TmpList


SAFSJAFTJTJSHP2 = SAFSJAFTJTJSHP[SAFSJAFTJTJSHP['overflow'].isin([1])]

buildPOP = SAFSJAFTJTJSHP[SAFSJAFTJTJSHP['USABILITY'].isin(["01000", "02000"])]
buildPOP = gpd.GeoDataFrame(buildPOP.loc[:, ["POP", "geometry"]])
buildPOP.crs = {'init' :'epsg:4326'}
buildPOP = buildPOP.to_crs({'init' :'epsg:5179'})
buildPOP.to_file(filedir + 'Common/buildPOP/BuildingPOP_' + sidoCode + '.shp')

def hextofloats(h):
    return list(int(h[i:i + 2], 16) for i in (1, 3, 5))

colormap = linear.OrRd_09.scale(SAFSJAFTJTJSHP2.POP.min(), SAFSJAFTJTJSHP2.POP.max()).to_step(10)
collist = [{'stroke': {'color': [0, 0, 0, 0], 'width':0.5}, "fill": {'color': hextofloats(colormap(pop))+[1]}} for pop in SAFSJAFTJTJSHP2['POP']]
SAFSJAFTJTJSHP2['style'] = collist

SAFSJAFTJTJSHP2.to_file(filedirfinal + '4_' + baseJSON + 'BUILDING1.json', driver = 'GeoJSON', encoding=encod)
BUILDING_J = json.load(open(filedirfinal + '4_' + baseJSON + 'BUILDING1.json', 'rt', encoding=encod))

OVER_Agri = SJAgri_I.loc[SJAgri_I["Overflow"] == 1]

coldic2 = {'stroke': {'color': [255, 0, 0, 0.5], 'width':1}, "fill": {'color': [255, 0, 0, 0.4]}}
OVER_Agri['style'] = [coldic2 for i in range(len(OVER_Agri))]

OVER_Agri.to_file(filedirfinal + '3_' + baseJSON + 'AGRI1.json', driver = 'GeoJSON', encoding=encod)
AGRI_J = json.load(open(filedirfinal + '3_' + baseJSON + 'AGRI1.json', 'rt', encoding=encod))


OVER1['POP'] = SAFSJAFTJTJSHP2['POP'].sum(axis=0)
OVER1['numofB'] = len(SAFSJAFTJTJSHP2)


OVER1['style'] = [{'stroke': {'color': [0, 0, 255, 0.5], 'width':1}, "fill": {'color': [0, 0, 255, 0.3]}}]
OVER1.to_file(filedirfinal + '1_' + baseJSON + 'DAMAGE1.json', driver = 'GeoJSON', encoding=encod)
OVER1_J = json.load(open(filedirfinal + '1_' + baseJSON + 'DAMAGE1.json', 'rt', encoding=encod)) 

SAFSJAFTJTJSHP_B = pd.merge(SJAFTJTJSHP, struct, on = 'STRCT_CD', how = 'left')

GSAFSJAFTJTJSHP_B = SAFSJAFTJTJSHP_B['geometry']
GGBUFFER = P_BUFFER['geometry']
SAFSJAFTJTJSHP_B['overflow'] = np.nan

for i in range(len(SAFSJAFTJTJSHP_B)):
    if GGBUFFER.iloc[0].intersects(GSAFSJAFTJTJSHP_B.iloc[i]) == True:
        SAFSJAFTJTJSHP_B['overflow'].iloc[i] = 1
        i = i+1
    else:
        i = i+1

Agri_I_B = merge[merge['LV2_CODE'].isin(["210", "220", "230", "240", "250"])]

GAgri_I_B =Agri_I_B['geometry']
GGBUFFER = P_BUFFER['geometry']
idx = []
TmpList = []


for i in range(len(GAgri_I_B)):
    if GGBUFFER.iloc[0].intersects(GAgri_I_B.iloc[i]) == True:
        idx.append(Agri_I_B.index[i])

for i in Agri_I_B.index:
    if i in idx:
        TmpList.append(1)
        i 
    else:
        TmpList.append('NA')
        
SJAgri_I_B = Agri_I_B
SJAgri_I_B['Overflow'] = TmpList

SAFSJAFTJTJSHP_B2 = SAFSJAFTJTJSHP_B[SAFSJAFTJTJSHP_B['overflow'].isin([1])]

colormap2 = linear.YlGn_09.scale(SAFSJAFTJTJSHP_B2.POP.min(), SAFSJAFTJTJSHP_B2.POP.max()).to_step(10)
collist2 = [{'stroke': {'color': [0, 0, 0, 0], 'width':0.5}, "fill": {'color': hextofloats(colormap2(pop))+[1]}} for pop in SAFSJAFTJTJSHP_B2['POP']]
SAFSJAFTJTJSHP_B2['style'] = collist2

SAFSJAFTJTJSHP_B2.to_file(filedirfinal + '4_' + baseJSON + 'BUILDING2.json', driver = 'GeoJSON', encoding=encod)
BUILDING_B_J = json.load(open(filedirfinal + '4_' + baseJSON + 'BUILDING2.json', 'rt', encoding=encod))


BUFFER_Agri = SJAgri_I_B.loc[SJAgri_I_B["Overflow"] == 1] 

coldic3 = {'stroke': {'color': [255, 165, 0, 0.5], 'width':1}, "fill": {'color': [255, 165, 0, 0.4]}}
BUFFER_Agri['style'] = [coldic3 for i in range(len(BUFFER_Agri))]

BUFFER_Agri.to_file(filedirfinal + '3_' + baseJSON + 'AGRI2.json', driver = 'GeoJSON', encoding=encod)
AGRI_B_J = json.load(open(filedirfinal + '3_' + baseJSON + 'AGRI2.json', 'rt', encoding=encod))


P_BUFFER['POP'] = SAFSJAFTJTJSHP_B2['POP'].sum(axis=0)
P_BUFFER['numofB'] = len(SAFSJAFTJTJSHP_B2)


P_BUFFER['style'] = [{'stroke': {'color': [135, 206, 235, 0.5], 'width':1}, "fill": {'color': [135, 206, 235, 0.3]}}]  

P_BUFFER.to_file(filedirfinal + '1_' + baseJSON + 'DAMAGE2.json', driver = 'GeoJSON', encoding=encod)
P_BUFFER_J = json.load(open(filedirfinal + '1_' + baseJSON + 'DAMAGE2.json', 'rt', encoding=encod))


newList = []
def switchLatLng(geos):
    for geo in geos:
        if isinstance(geo[0], list):
            switchLatLng(geo)
        else:
            newList.append([geo[1], geo[0]])
    return newList



center = list(reversed(list(OVER1['geometry'].centroid[0].coords[0])))
testmap6 = folium.Map(location = center, tiles=None, overlay=False, zoom_start=17)


Damage = folium.FeatureGroup(name='침수 피해 지역', overlay = False).add_to(testmap6)

over = plugins.FeatureGroupSubGroup(Damage, name='침수구역', show = False).add_to(testmap6)
style_function = lambda x: {'color':'blue',
                            'fillColor':"blue",
                            'opacity': 0.5,
                            'fillOpacity':0.3}
overRange = folium.features.GeoJson(OVER1_J, style_function = style_function, popup = folium.GeoJsonPopup(fields = ['POP', 'numofB'], 
                                    aliases=['총 인명피해: ', '총 건물피해: '], style="font-size: 12pt;"), control = False).add_to(over)

Bld_C = plugins.FeatureGroupSubGroup(Damage, name='인명 피해 단계구분도', show = False).add_to(testmap6)

colormap = linear.OrRd_09.scale(SAFSJAFTJTJSHP2.POP.min(), SAFSJAFTJTJSHP2.POP.max()).to_step(10, )
colormap.caption = '침수 구역 내 인명 피해'
style_function = lambda x: {'color':'black',
                            'fillColor':colormap(x['properties']['POP']),
                            'weight':0.5,
                            'fillOpacity':1.0}

Building = folium.features.GeoJson(BUILDING_J, style_function=style_function, 
                                    popup = folium.GeoJsonPopup(fields = ['POP', '코드값의미', 'A9'], 
                                    aliases=['인명피해: ', '건물구조: ', '건물용도: '], style="font-size: 12pt;"), control = False).add_to(Bld_C)  

colormap.add_to(testmap6)


agri = plugins.FeatureGroupSubGroup(Damage, name='농작물 피해', show = False).add_to(testmap6)
style_function = lambda x: {'color':'red',
                            'fillColor':"red",
                            'opacity': 0.5,
                            'fillOpacity':0.4}

agriRange = folium.features.GeoJson(AGRI_J, style_function = style_function, 
                                    popup = folium.GeoJsonPopup(fields = ['LV2_NAME'], aliases=['피해 농업지역: '], style = "font-size: 12pt;"), control = False).add_to(agri)



Expect = folium.FeatureGroup(name='예상 피해 지역', overlay = False).add_to(testmap6)

buffer = plugins.FeatureGroupSubGroup(Expect, name='침수예상구역', show = False).add_to(testmap6)
style_function = lambda x: {'color':'skyblue',
                            'fillColor':"skyblue",
                            'opacity': 0.5,
                            'fillOpacity':0.3}
bufferRange = folium.features.GeoJson(P_BUFFER_J, style_function = style_function, popup = folium.GeoJsonPopup(fields = ['POP', 'numofB'], 
                                    aliases=['총 예상 인명피해: ', '총 예상 건물피해: '], style="font-size: 12pt;"), control = False).add_to(buffer)


Bld_B_C = plugins.FeatureGroupSubGroup(Expect, name='예상 인명 피해 단계구분도', show = False).add_to(testmap6)

colormap2 = linear.YlGn_09.scale(SAFSJAFTJTJSHP_B2.POP.min(), SAFSJAFTJTJSHP_B2.POP.max()).to_step(10)
colormap2.caption = '예상 침수 구역 내 인명 피해'
style_function = lambda x: {'color':'black',
                            'fillColor':colormap2(x['properties']['POP']),
                            'weight':0.5,
                            'fillOpacity':1.0}

Building_B = folium.features.GeoJson(BUILDING_B_J, style_function=style_function, 
                                    popup = folium.GeoJsonPopup(fields = ['POP', '코드값의미', 'A9'], 
                                    aliases=['인명피해: ', '건물구조: ', '건물용도: '], style="font-size: 12pt;"), control = False).add_to(Bld_B_C)  

colormap2.add_to(testmap6)

agri_B = plugins.FeatureGroupSubGroup(Expect, name='예상 농작물 피해', show = False).add_to(testmap6)
style_function = lambda x: {'color':'orange',
                            'fillColor':"orange",
                            'opacity': 0.5,
                            'fillOpacity':0.4}

agriRange = folium.features.GeoJson(AGRI_B_J, style_function = style_function, 
                                    popup = folium.GeoJsonPopup(fields = ['LV2_NAME'], aliases=['예상 피해 농업지역: '], style = "font-size: 12pt;"), control = False).add_to(agri_B)



folium.TileLayer('openstreetmap',overlay=True, name="basemap").add_to(testmap6)

folium.LayerControl(collapsed=False).add_to(testmap6)

testmap6.save(filedirfinal + '0_' + baseJSON + 'MAP.html')

print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')