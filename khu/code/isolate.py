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
from tabulate import tabulate

import warnings
warnings.filterwarnings("ignore")

import time
starttime = time.time()


os.chdir('../')
filedir = './inputData/' # input file path
finpath = './outputData/isolate/isolate/' # output file path

import sys

sidoCode = sys.argv[1]
sidoCode = str(sidoCode)

from datetime import datetime
current_time = datetime.now()
dateandtime = current_time.strftime("%Y%m%dT%H%M")

dtype = "TYPHN"
mtype = "ISOLA"
sep = '_'
baseJSON = mtype+sep+dateandtime+sep+dtype+sep

rL = filedir + 'Common/road/' + 'NGII_CDM_도로중심선_simple_' + sidoCode + '.shp' ## 시군구별 도로 자료(지오앤 요청 자료), EPSG 5179
bL = filedir + 'isolate/bridge/' + 'NGII_CDM_교량_' + sidoCode + '.shp' ## 시군구별 교량 자료(자체제작), EPSG 5179

print("[2] 전국 도로네트워크, 교량 정보 Load")


roadLayer = gpd.read_file(rL)
bridge = gpd.read_file(bL)

## 멀티파트를 싱글파트로
def multi2sing(link_dataframe):
    link_dataframe = link_dataframe.drop(list(link_dataframe.columns)[:-1], axis=1)
    
    # multi 라인 찾기
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


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

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

# 고립지 탐색
for point in tqdm(range(len(bridge))):
    # 교량 포함하는 sub graph 탐색
    bridge_point = bridge['geometry'][point].coords
    roadnetwork2 = [[np.nan]]
    for sub_G_ith in sub_graphs:
        if bridge_point[0] in sub_G_ith.nodes():
            roadnetwork2 = sub_G_ith.copy()
            break

    try: # 서브네트워크 없을 시 패스
        if np.isnan(roadnetwork2)[0][0]:
            continue
    except: # 서브네트워크가 라인 한 개의 섬일 때 패스
        if roadnetwork2.number_of_nodes() <= 1:
            continue
        
    # 네트워크에 교량 밖에 없을 시 패스
    if roadnetwork2.number_of_edges() <= 1:
        continue
        
    # 교량 제거
    try:
        roadnetwork2.remove_edge(bridge_point[0],
                                 bridge_point[-1])
    except:
        # 교량의 mid point 제거
        roadnetwork2.remove_nodes_from(list(bridge_point)[1:-1])
        
    # 교량 끝점에서 끝점 갈 수 있으면 pass
    if nx.has_path(roadnetwork2, bridge_point[0], bridge_point[-1]):
        continue

    # 교량 제거되어 분리된 sub graphs
    sub_graphs2 = (roadnetwork2.subgraph(c).copy() for c in nx.connected_components(roadnetwork2))
    sub_graphs2_L = list(sub_graphs2)
    # 서브그래프가 하나 밖에 없으면(=분리 안 됨) 패스
    if len(sub_graphs2_L) == 1:
        continue
    
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
    if sub_graphs3.number_of_edges() == 0:
        continue
    
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
    
    if geo_df.convex_hull.area[0] > 48000000:
        continue

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
    if geo_df['area'].iloc[0] > 48000000:
        continue
    
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
        
    # 좌표계 지정 후 저장
    geo_df.crs = {'init' :'epsg:5179'}
    geo_df = geo_df.to_crs(epsg = 4326)
    geo_df.to_file(finpath + 'route_isolated_' + str(point) + '.json', driver='GeoJSON', encoding = 'utf-8')
    
    tmp_BUFFER.crs = {'init' :'epsg:5179'}
    tmp_BUFFER = tmp_BUFFER.to_crs(epsg = 4326)
    tmp_BUFFER.to_file(finpath + 'BUFFER_' + str(point) + '.json', driver = 'GeoJSON')


a = []
for i in os.listdir(finpath)[0:int(len(os.listdir(finpath))/2)]:
    a.append(json.load(open(finpath + i)))
    
b = []
for k in os.listdir(finpath)[int(len(os.listdir(finpath))/2+1):int(len(os.listdir(finpath)))]:
    b.append(json.load(open(finpath + k, encoding = 'utf-8')))


print("[4] 시각화")

#### folium module 이용해 시각화

visual_brid = bridge

visual_brid.crs = {'init' : 'epsg:5179'}
visual_brid = visual_brid.to_crs(epsg=4326)
visual_brid['code'] = 1

center = [visual_brid.dissolve(by='code').centroid.y[1],
          visual_brid.dissolve(by='code').centroid.x[1]]
base = folium.Map(location = center, zoom_start=8)

style0 = {'fillColor': '#FFFFFF', 'color': '#FFFFFF'}
style1 = {'fillColor': '#B22222', 'color': '#B22222'}
style2 = {'fillOpacity': '0.02', 'color': '#3399FF'}
style3 = {'fillOpacity' : '0.02', 'color' : 'red'}

for i in range(len(a)):
    buffer = folium.GeoJson(a[i], style_function=lambda x:style1, show = False).add_to(base)
    
for i in range(len(b)):
    route = folium.GeoJson(b[i], style_function=lambda x:style1, show = False).add_to(base)

base.save(finpath + '0_' + baseJSON + 'MAP.html')

print('end')
endtime = time.time() - starttime
print('time: ' + str(endtime) + 's')