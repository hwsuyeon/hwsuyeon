import shapefile
import pandas as pd
import os
import geopandas as gpd
import numpy as np
import networkx as nx
import json
from shapely.geometry import Point, mapping, Polygon, LineString, MultiLineString, asShape, asMultiPolygon, LinearRing
import folium
from tqdm import tqdm
from pyproj import Proj, transform, Geod
from operator import itemgetter
from itertools import groupby

import warnings
warnings.filterwarnings("ignore")

import time
starttime = time.time()


os.chdir('../')
filedir = './inputData/' # input file path
finpath = './outputData/isolate/liveisolate/' # output file path

import sys

sidoCode = sys.argv[1]
lon = sys.argv[2]
lat = sys.argv[3]

encod  = 'UTF8'
sidoCode = str(sidoCode)

from datetime import datetime
current_time = datetime.now()
dateandtime = current_time.strftime("%Y%m%dT%H%M")

dtype = "TYPHN"
mtype = "ISORE"
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


rL = filedir + 'Common/road/' + 'NGII_CDM_도로중심선_simple_' + sidoCode + '.shp' ## 시군구별 도로 자료(지오앤 요청 자료), EPSG 5179
#rL = filedir + "Common/road/1NGII_" + sidoCode + ".shp"

bL = filedir + 'isolate/bridge/' + 'NGII_CDM_교량_' + sidoCode + '.shp' ## 시군구별 교량 자료(자체제작), EPSG 5179
SI = filedir + 'Common/li_Bound/' + 'TL_SCCO_LI.shp' ## 행정구역 읍면리 데이터

roadLayer = gpd.read_file(rL)
bridge = gpd.read_file(bL)
SIG = gpd.read_file(SI)

## multipart to singlepart
def multi2sing(link_dataframe):
    link_dataframe = link_dataframe.drop(list(link_dataframe.columns)[:-1], axis=1)

    link_geotypeL = link_dataframe.geom_type

    multi_index = []
    for i, link_gtype in enumerate(link_geotypeL):
        if link_gtype != 'LineString':
            multi_index.append(i)
            
    tmp_link_dataframe = pd.DataFrame(columns = ['geometry'])
    for i in multi_index:
        for link_ith_geo in link_dataframe.loc[i].geometry:
            tmp_link_dataframe = tmp_link_dataframe.append({'geometry':link_ith_geo}, ignore_index=True)
            
    link_dataframe = link_dataframe.drop(multi_index)
    link_dataframe = link_dataframe.append(tmp_link_dataframe)
    link_dataframe = link_dataframe.reset_index(drop=True)
    
    return link_dataframe   

link_dataframe = multi2sing(roadLayer)
bridge = multi2sing(bridge)


lon = float(lon)
lat = float(lat)

location2point = Point((lon, lat))
input_point = gpd.GeoDataFrame({'geometry':[location2point]})
input_point.crs = {'init' : 'epsg:4326'}
input_point = input_point.to_crs(epsg=5179)


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

mindist = 100
for i, brgC in enumerate(bridge.centroid):
    # distance b/w a and b
    d = dist((input_point.geometry[0].x, input_point.geometry[0].y),
             (brgC.x, brgC.y))
    if d < mindist :
        mindist = d
        bridge_ith = i


# gdf to nx
def gdf_to_nx(
    gdf_network,
    approach="primal",
    length="mm_len",
    multigraph=False,
    directed=False,
    angles=True,
    angle="angle",
):
    gdf_network = gdf_network.copy()
    if "key" in gdf_network.columns:
        gdf_network.rename(columns={"key": "__key"}, inplace=True)
    if multigraph and directed:
        net = nx.MultiDiGraph()
    elif multigraph and not directed:
        net = nx.MultiGraph()
    elif not multigraph and directed:
        net = nx.DiGraph()
    else:
        net = nx.Graph()

    net.graph["crs"] = gdf_network.crs
    gdf_network[length] = gdf_network.geometry.length
    fields = list(gdf_network.columns)

    if approach == "primal":
        _generate_primal(net, gdf_network, fields, multigraph)

    elif approach == "dual":
        if directed:
            raise ValueError("Directed graphs are not supported in dual approach.")

        _generate_dual(
            net, gdf_network, fields, angles=angles, multigraph=multigraph, angle=angle
        )

    else:
        raise ValueError(
            f"Approach {approach} is not supported. Use 'primal' or 'dual'."
        )

    return net

def _generate_primal(G, gdf_network, fields, multigraph):
    G.graph["approach"] = "primal"
    key = 0
    for row in gdf_network.itertuples():
        first = row.geometry.coords[0]
        last = row.geometry.coords[-1]

        data = [r for r in row][1:]
        attributes = dict(zip(fields, data))
        if multigraph:
            G.add_edge(first, last, key=key, **attributes)
            key += 1
        else:
            G.add_edge(first, last, **attributes)
            
def _generate_dual(G, gdf_network, fields, angles, multigraph, angle):
    """
    Generate dual graph
    Helper for gdf_to_nx.
    """
    G.graph["approach"] = "dual"
    key = 0

    sw = libpysal.weights.Queen.from_dataframe(gdf_network, silence_warnings=True)
    cent = gdf_network.geometry.centroid
    gdf_network["temp_x_coords"] = cent.x
    gdf_network["temp_y_coords"] = cent.y

    for i, row in enumerate(gdf_network.itertuples()):
        centroid = (row.temp_x_coords, row.temp_y_coords)
        data = [f for f in row][1:-2]
        attributes = dict(zip(fields, data))
        G.add_node(centroid, **attributes)

        if sw.cardinalities[i] > 0:
            for n in sw.neighbors[i]:
                start = centroid
                end = (
                    gdf_network["temp_x_coords"].iloc[n],
                    gdf_network["temp_y_coords"].iloc[n],
                )
                p0 = row.geometry.coords[0]
                p1 = row.geometry.coords[-1]
                geom = gdf_network.geometry.iloc[n]
                p2 = geom.coords[0]
                p3 = geom.coords[-1]
                points = [p0, p1, p2, p3]
                shared = [x for x in points if points.count(x) > 1]
                if shared:  # fix for non-planar graph
                    remaining = [e for e in points if e not in [shared[0]]]
                    if len(remaining) == 2:
                        if angles:
                            angle_value = _angle(remaining[0], shared[0], remaining[1])
                            if multigraph:
                                G.add_edge(start, end, key=0, **{angle: angle_value})
                                key += 1
                            else:
                                G.add_edge(start, end, **{angle: angle_value})
                        else:
                            if multigraph:
                                G.add_edge(start, end, key=0)
                                key += 1
                            else:
                                G.add_edge(start, end)

roadnetwork = gdf_to_nx(link_dataframe)
roadnetwork = roadnetwork.to_undirected()

# 전체 네트워크의 모든 점
rawPoint = []
gp_edgelist = list(roadnetwork.edges())
for i in tqdm(range(len(gp_edgelist))):        
    for j in range(2):
        rawPoint.append(gp_edgelist[i][j])

# dangle 찾기
dangle = []
for k in tqdm(range(len(rawPoint))):
    rP_kth = rawPoint[k]
    first_idx = rawPoint.index(rP_kth)
    try:
        rawPoint.index(rP_kth, first_idx+1)
    except:
        dangle.append(rP_kth)

# 0.1m 이내로 끊어진 도로 찾기
need2link = []
for dangdang2 in tqdm(dangle):
    for rawP in rawPoint:
        # distance b/w a and b
        d = dist(dangdang2, rawP)
        if d < 0.1:
            if dangdang2 == rawP:
                continue
            need2link.append((dangdang2, rawP))
            break

# 끊어진 도로 잇기
roadnetwork.add_edges_from(need2link)

# sub graph 제작
sub_graphs = (roadnetwork.subgraph(c).copy() for c in nx.connected_components(roadnetwork))
sub_graphs = list(sub_graphs)


bridge_point = bridge['geometry'][bridge_ith].coords
roadnetwork2 = [[np.nan]]
for sub_G_ith in sub_graphs:
    if bridge_point[0] in sub_G_ith.nodes():
        roadnetwork2 = sub_G_ith.copy()
        break

"""try: # 서브그래프 없을 시 패스
    if np.isnan(roadnetwork2)[0][0]:
        print('there is no subgraph')
except: # 서브네트워크가 라인 한 개의 섬일 때 패스
    if roadnetwork2.number_of_nodes() <= 1:
        print('isolated bridge')"""

# 네트워크에 교량 밖에 없을 시 패스
# raod_bld>새폴더>1NGII 울진군전체 도로 데이터 사용 시 여기서 오류

"""if roadnetwork2.number_of_edges() <= 1:
    print('isolated bridge')"""

# 교량 제거
try:
    roadnetwork2.remove_edge(bridge_point[0],
                             bridge_point[-1])
except:
    # 교량의 mid point 제거
    roadnetwork2.remove_nodes_from(list(bridge_point)[1:-1])

"""# 교량 끝점에서 끝점 갈 수 있으면 pass
if nx.has_path(roadnetwork2, bridge_point[0], bridge_point[-1]):
    print('Not isolated')"""

# 교량 제거되어 분리된 sub graphs
sub_graphs2 = (roadnetwork2.subgraph(c).copy() for c in nx.connected_components(roadnetwork2))
sub_graphs2_L = list(sub_graphs2)
# 서브그래프가 하나 밖에 없으면(=분리 안 됨) 패스
"""if len(sub_graphs2_L) == 1:
    print('Not isolated')"""

# node가 적은 sub graph(고립네트워크)선택 (나중에 아닐 경우도 고민해봐야함)
total_nodes = sum([num_edges.number_of_edges() for num_edges in sub_graphs2_L])
min_num_nodes = 1e100
for i in range(len(sub_graphs2_L)):
    num_nodes = sub_graphs2_L[i].number_of_nodes()
    if num_nodes < min_num_nodes:
        if total_nodes == 1:
            if num_nodes == 0:
                continue
        min_num_nodes = num_nodes
        sub_graphs3 = sub_graphs2_L[i]
"""if sub_graphs3.number_of_edges() == 0:
    print('there is topology error')"""

# 교량 중 서브그래프와 맞닿아 있지 않은 점
for i in range(-1, 1):
    if bridge_point[i] not in sub_graphs3.nodes():
        out_bridge_Point = Point(bridge_point[i])

## shapefile 만들기   
# 고립네트워크 to shp
edgeList = []
for edge_ith in list(sub_graphs3.edges()):
    edgeList.append(LineString([edge_ith[0], edge_ith[1]]))
geo_df = gpd.GeoDataFrame({'geometry':[MultiLineString(edgeList)]}) # geo_df.length[0]

"""if geo_df.convex_hull.area[0] > 48000000:
    print('too large isolated area')"""

total_length = 1
while True:
    touch_link = link_dataframe[link_dataframe.touches(geo_df.geometry.explode().boundary.unary_union)]
    if len(touch_link) > 0:
        geo_df = touch_link
    else:
        geo_df = link_dataframe[link_dataframe.intersects(Point(list(sub_graphs3.nodes())))]
    geo_df = geo_df[~geo_df.intersects(out_bridge_Point)]
    tmp_length = len(geo_df)
    if tmp_length > total_nodes:
        continue
    if tmp_length == total_length:
        if len(geo_df) == 1:
            break
        geo_df = gpd.GeoDataFrame({'geometry':[MultiLineString(geo_df.unary_union)]})
        break
    total_length = len(geo_df)

# 면적 48km2 이상이면 패스
geo_df['area'] = geo_df.convex_hull.area
"""if geo_df['area'].iloc[0] > 48000000:
    print('too large isolated area')"""

# 면적이 0.64km2 미만이면 0, 이상이면 1
if geo_df['area'].iloc[0] < 640000:
    geo_df['label'] = 0
else:
    geo_df['label'] = 1

# 선택된 고립지에 30m buffer 적용
Buffer = geo_df.geometry.buffer(30)
BUFFER = gpd.GeoDataFrame(geo_df.copy(), geometry = Buffer)
BUFFER['code'] = 1
tmp_BUFFER = BUFFER.dissolve(by='code')



SIG_ttmp = SIG[SIG.geometry.intersects(geo_df.unary_union)]
SIG_ttmp = SIG_ttmp.to_crs(epsg=4326)
SIG_ttmp.to_file(finpath + '1_' + baseJSON + 'LIBOUND.json', driver = 'GeoJSON', encoding = 'utf-8')
SIG_select_J = json.load(open(finpath + '1_' + baseJSON + 'LIBOUND.json', encoding = 'utf-8'))

geo_df.crs = {'init' :'epsg:5179'}
geo_df = geo_df.to_crs(epsg = 4326)
geo_df.to_file(finpath + '6_' + baseJSON + 'ROUTE.json', driver='GeoJSON')
route_J = json.load(open(finpath + '6_' + baseJSON + 'ROUTE.json'))

tmp_BUFFER.crs = {'init' :'epsg:5179'}
tmp_BUFFER = tmp_BUFFER.to_crs(epsg = 4326)
tmp_BUFFER.to_file(finpath + '2_' + baseJSON + 'ISOLATE.shp', driver = 'ESRI Shapefile') 
tmp_BUFFER.to_file(finpath + '2_' + baseJSON + 'ISOLATE.json', driver = 'GeoJSON')
P_BUFFER_J = json.load(open(finpath + '2_' + baseJSON + 'ISOLATE.json'))

visual_brid = bridge[bridge.index==bridge_ith]
visual_brid.crs = {'init' : 'epsg:5179'}
visual_brid = visual_brid.to_crs(epsg=4326)
visual_brid.to_file(finpath + '5_' + baseJSON + 'BRIDGE.json', driver = 'GeoJSON', encoding = 'utf-8')
visual_brid_J = json.load(open(finpath + '5_' + baseJSON + 'BRIDGE.json', encoding = 'utf-8'))

"""road = roadLayer.to_crs(epsg=4326)
road['geometry'].to_file(finpath + 'road.json', driver = 'GeoJSON', encoding = 'utf-8')
road_J = json.load(open(finpath + 'road.json'))
"""

crs = {'init' :'epsg:4326'}

isolate = gpd.read_file(finpath +'2_' + baseJSON + 'ISOLATE.shp', encoding = 'cp949') # 고립위험지역

build = gpd.read_file(filedir + "Common/buildPOP/BuildingPOP_" + sidoCode + ".shp", encoding='euckr')
shel = gpd.read_file(filedir + "Common/shelter/shelter_" + sidoCode + ".shp", encoding='utf-8')

buildLayer = build.to_crs(crs)
shelLayer = shel.to_crs(crs)

building_intersect = buildLayer['geometry'].intersects(isolate.unary_union)

dangerLayer_buf = isolate.buffer(0.1)

shelter_intersect = shelLayer['geometry'].intersects(dangerLayer_buf.unary_union)
shelter_diff = shelLayer[shelter_intersect]['geometry'].difference(isolate.unary_union)



evaPop = []
for i in range(len(buildLayer)):
    if building_intersect[i] == True:
        evaPop.append(buildLayer['POP'][i])


capPop = []
for i in range(len(shelLayer)):
    if shelter_intersect[i] == True:
        if shelter_diff.is_empty[i] == False:
            capPop.append(shelLayer['capacity'][i])

road = gpd.read_file(rL)
roadLayer = road.to_crs(4326)

road_intersect = roadLayer['geometry'].intersects(dangerLayer_buf.unary_union)

building_intersect = buildLayer[building_intersect]
road_intersect = roadLayer[road_intersect]
shelter_diff = shelLayer[shelter_intersect * shelter_diff.is_empty == False]

shelter_diff = shelter_diff.reset_index()

# 건물 중심점
BuildingPoints = []
build_x = list(building_intersect.centroid.x)
build_y = list(building_intersect.centroid.y)
for i in range(len(building_intersect)):
    BuildingPoints.append((build_x[i], build_y[i]))

# 대피소 중심점
ShelterPoints = []
shel_x = list(shelter_diff.centroid.x)
shel_y = list(shelter_diff.centroid.y)
for i in range(len(shelter_diff)):
    ShelterPoints.append((shel_x[i], shel_y[i]))

def roadPoint(locationCoor, bufferSize):
    # input : 대피소의 좌표, 대피소의 버퍼 크기. output : 설정된 버퍼 내의 로드 포인트
    pt = Point(locationCoor)
    disasterC = json.dumps(mapping(pt.buffer(bufferSize))) # buffer size 설정
    disasterC = json.loads(disasterC)['coordinates']
    poly = Polygon(disasterC[0]) # 대피소 지점을 중심으로 폴리곤 생성
#     tmpCircle = list(poly.exterior.coords)

    Tmp = [list(coor.coords) for coor in road_intersect['geometry'] if poly.intersects(coor) == True]
    # 중첩된 부분 찾기 - 함수화 하는 경우, 사전에 선언한 내용을 명시하기 (ROADSHP), input으로 넣기
    
    shelterBuffer = []
    for coorL in Tmp: # 중첩된 부분의 좌표 찾기
        for i in coorL:
            shelterBuffer.append(i)
    return shelterBuffer


# 건물 - 도로 포인트
BuildingRoad = []
for i, buildPT in enumerate(BuildingPoints):
    Roadpt = roadPoint(buildPT, 0.00005) ## 약 5m
    if len(Roadpt) != 0:
        BuildingRoad.append({'id': i,
                             'RoadPoint': Roadpt[1], # 1은 중심점, 0은 밖 첫 노드
                             'residents' : list(building_intersect['POP'])[i]})

shel_ID = list(shelter_diff.index)

# 대피소에 인접한 도로 포인트
ShelterRoad = []
for i, buildPT in enumerate(ShelterPoints):
    Roadpt = roadPoint(buildPT, 0.00003) ## 약 3m
    if len(Roadpt) != 0:
        ShelterRoad.append({'id': shel_ID[i],
                            'RoadPoint': Roadpt[1], # 1은 중심점, 0은 밖 첫 노드
                            'capacity' : list(shelter_diff['capacity'])[i]})



road_intersect.to_file(finpath + "road_intersect2.shp")

#### A*

# read road shapefile
roadnetwork = nx.read_shp(finpath + "road_intersect2.shp", simplify=False)
roadnetwork2 = roadnetwork.to_undirected()



def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


nx.set_edge_attributes(roadnetwork2,
                       {e: dist(e[0], e[1]) for e in roadnetwork2.edges()}, 'cost')


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
            shortPath = 'NA'
        if shortPath != 'NA':
            length = sum(roadnetwork2[u][v].get('cost', 1) for u, v in zip(shortPath[:-1], shortPath[1:]))
            
            row = pd.DataFrame(data = [[i, ShelterRoad[j]['id'], length, shortPath]],
                               columns = ['BuildId', 'ShelId', 'Length', 'Route'])
            routeTable = routeTable.append(row)

routeTable.loc[:, 'RouteId'] = range(len(routeTable))

#### 건물순 할당

temp_time = time.time()
routeList = []
binBuildList = []
binShelList = []

# 원래 가지고 있던 road_ngii>NGII_CDM_도로중심선_simple 데이터는 여기서 에러 발생

while sum(evaPop) > 0 and sum(capPop) > 0:
    # 대피인원이 0인 리스트에 포함되지 않는 테이블만 추출
    conRouteTable = routeTable[routeTable['BuildId'].isin(binBuildList) == False]
    # 다시 수용인원이 0인리스트에 포함되지 않는 테이블만 추출
    conRouteTable = conRouteTable[conRouteTable['ShelId'].isin(binShelList) == False]
    # 길이가 최소인 건물, 대피소 쌍의 건물 번호
    buildId = conRouteTable.loc[conRouteTable['Length'] == min(conRouteTable['Length']),
                                 'BuildId'].iloc[0]
    # 최소 거리의 건물만으로 서브 테이블 생성
    subTable = conRouteTable.loc[conRouteTable['BuildId'] == buildId, :]
    # 서브 테이블을 길이 순으로 재정렬
    subTableSort = subTable.sort_values(by=['Length'])
    
    for i in range(len(subTableSort)):
        # 건물의 대피인원 저장
        eva = evaPop[buildId]
        # 대피소 번호 저장
        shelId = subTableSort.iloc[i, 1]
        # 대피소의 수용력 저장
        cap = capPop[shelId]

        # 수용인원이 대피인원 이상일 경우
        if cap >= eva:
            # 대피인원 전부 할당
            value = eva
            # 사용할 경로의 리스트에 추가
            routeList.append(subTableSort.iloc[i, 4])
            # 대피인원을 0
            evaPop[buildId] = 0
            # 수용력은 대피인원만큼 감소
            capPop[shelId] = cap - eva
            # 모두 대피한 건물 리스트에 추가
            binBuildList.append(buildId)
            # routeTable에 대피(이동)인원 저장
            routeTable.loc[routeTable['RouteId'] == subTableSort.iloc[i, 4], 'Pop'] = value
            # subTable 반복 종료
            break

        # 대피인원이 많을 경우
        else:
            # 대피인원은 수용 가능 인원만큼만
            value = cap
            # 사용할 경로의 리스트에 추가
            routeList.append(subTableSort.iloc[i, 4])
            # 수용력을 0
            capPop[shelId] = 0
            # 대피인원 일부 할당
            evaPop[buildId] = eva - cap
            # 모두 수용한 대피소 리스트에 추가
            binShelList.append(shelId)
            # routeTable에 대피(이동)인원 저장
            routeTable.loc[routeTable['RouteId'] == subTableSort.iloc[i, 4], 'Pop'] = value 


#### 지도 시각화
# 대피 인원 없는 경로 제외
routeTable = routeTable[routeTable['Pop'].isna() == False]


# 좌표 리스트를 라인 형태로
routelist = []
for i in range(len(routeTable)):
    routelist.append(LineString(routeTable['Route'].iloc[i]))

routegeo = pd.DataFrame(routelist, columns = ['geometry'])

routeTable = routeTable.reset_index()
routeTable = pd.concat([routeTable, routegeo], axis = 1)


routeTable = routeTable.drop(columns=['Route', 'index'])
routeTable = gpd.GeoDataFrame(routeTable)


routeTable.crs = {'init' : 'epsg:4326'}



# 경고 메시지 발생을 무시하는 코드
import warnings
warnings.filterwarnings("ignore")

import shapefile
from osgeo import gdal
from json import dumps



route_wgs = routeTable
dsv = route_wgs.dissolve('ShelId', aggfunc = 'sum')
dsv['style'] = [{"stroke": {'color': list(np.random.choice(range(255),size=3))+[1], "width":3, "linecap":"round"}} for i in dsv.index]
dsv = dsv.loc[:, ['Pop', 'style', 'geometry']].reset_index()

buffer = []
for i in range(len(dsv)):
    atr = dict(dsv.iloc[i, :-1])
    geom = dsv.geometry[i].__geo_interface__
    buffer.append(dict(type="Feature", geometry=geom, properties=atr))
    

geojson = open(finpath + '7_' + baseJSON + 'ALLROUTE.json', "w")
geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2, cls=NpEncoder) + "\n")
geojson.close()


import pyproj
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium import plugins

# 대피소 좌표계 변환 후 json으로

shelter_wgs = shelter_diff.to_crs(epsg = 4326)
shelter_wgs['allo'] = list(dsv.Pop) + list(np.repeat(0, len(shelter_wgs)-len(dsv)))
shelter_wgs['style'] = [{'color': [255, 0, 0, 1]} for i in range(len(shelter_wgs))]
shelter_wgs = shelter_wgs.loc[:, ['민방위', 'capacity', 'allo', 'style', 'geometry']]
shelter_wgs.to_file(finpath + '4_' + baseJSON + 'SHELTER.json', driver = 'GeoJSON', encoding=encod)



# 건물 좌표계 변환 후 json으로

building_wgs = building_intersect.to_crs(epsg = 4326)
building_wgs['style'] = [{'stroke': {'color': [100, 100, 100, 0.5], 'width':1}, "fill": {'color': [100, 100, 100, 0.3]}} for i in range(len(building_wgs))]
building_wgs = building_wgs.loc[:, ['POP', 'style', 'geometry']]
building_wgs.to_file(finpath + '3_' + baseJSON + 'BUILDING.json', driver = 'GeoJSON', encoding=encod)


dangerLayer = isolate

danger_wgs = dangerLayer.to_crs(epsg = 4326)
danger_wgs['style'] = [{'stroke': {'color': [0, 0, 255, 0.5], 'width':1}, "fill": {'color': [0, 0, 255, 0.3]}}]

danger_wgs.to_file(finpath + '2_' + baseJSON + 'ISOLATE.json', driver = 'GeoJSON', encoding=encod)



shelter = json.load(open(finpath + '4_' + baseJSON + 'SHELTER.json', encoding=encod))
BUILDING_J = json.load(open(finpath + '3_' + baseJSON + 'BUILDING.json', encoding=encod))
OVER1_J = json.load(open(finpath + '2_' + baseJSON + 'ISOLATE.json', encoding=encod))
Route_J = json.load(open(finpath + '7_' + baseJSON + 'ALLROUTE.json', encoding=encod))


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


style0 = {'fillColor': '#FFFFFF', 'color': '#FFFFFF'}
style1 = {'fillColor': '#B22222', 'color': '#B22222'}
style2 = {'fillOpacity': '0.02', 'color': '#3399FF'}
style3 = {'fillOpacity' : '0.02', 'color' : 'red'}

buffer = folium.GeoJson(P_BUFFER_J, style_function=lambda x:style1, show = False).add_to(base)
buffer.layer_name = '고립위험지역'
visual_brid = folium.GeoJson(visual_brid_J, style_function=lambda x:style3, show = False).add_to(base)
visual_brid.layer_name = '침수위험교량'
select = folium.GeoJson(SIG_select_J, style_function=lambda x:style2, show = False).add_to(base)
select.layer_name = '지방자치단체(리)'
route = folium.GeoJson(route_J, style_function=lambda x:style0, show = False).add_to(base)
route.layer_name = '고립위험지역경로'

for i in routegroup:
    base.add_child(i)
    
base.add_child(folium.map.LayerControl())

base.save(finpath + '0_' + baseJSON + 'MAP.html')

print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')

