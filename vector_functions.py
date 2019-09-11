# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:22:16 2019

@author: ericv
"""


import os

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

import json
from affine import Affine
from shapely.geometry import Point, Polygon

def coords2gdf(ds, xcoord, ycoord):
    coord_lst = [rasterio.transform.xy(transform = Affine.from_gdal(*ds.GetGeoTransform()), rows = ycoord[i], cols = xcoord[i], offset='ul') for i in range(len(xcoord))]
    gdf_point = gpd.GeoDataFrame(geometry = [Point(x, y) for x, y in coord_lst], crs = {'init': 'epsg:4326'})
    return gdf_point

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    temp = json.loads(gdf.to_json())#['features'][0]['geometry']]
    features = []
    for row in temp['features']:
        features.append(row['geometry'])
    return features

def detected_plants2projected_shp_and_points(img_path, out_path, df, ds, write2file):
    #convert centroids to coords and contours to shape in lat, lon
    df['coords'] = df.centroid.apply(lambda x:rasterio.transform.xy(transform = Affine.from_gdal(*ds.GetGeoTransform()), rows = x[0], cols = x[1], offset='ul'))
    df['geom'] = df.contours.apply(lambda x:rasterio.transform.xy(transform = Affine.from_gdal(*ds.GetGeoTransform()), rows = list(x[:,0,1]), cols = list(x[:,0,0]), offset='ul'))        
    #convert df to gdf
    #for polygon, first reformat into lists of coordinate pairs
    shape_list = []
    for geom in df.geom:
        x_list = geom[0]
        y_list = geom[1]
        coords_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            coords_list.append([x, y])
        shape_list.append(coords_list)
    df['geom2'] = shape_list

    #create points
    gdf_point = gpd.GeoDataFrame(df, geometry = [Point(x, y) for x, y in df.coords], crs = {'init': 'epsg:4326'})
    gdf_point = gdf_point.drop(['contours', 'moment', 'bbox', 'input',
       'centroid', 'coords', 'geom', 'geom2'], axis=1)
    #create polygons
    gdf_poly = gpd.GeoDataFrame(df, geometry = [Polygon(shape) for shape in df.geom2], crs = {'init': 'epsg:4326'})
    gdf_poly = gdf_poly.drop(['contours', 'moment', 'bbox', 'input',
       'centroid', 'coords', 'geom', 'geom2'], axis=1)

    calc_area = gdf_poly.to_crs({'init': 'epsg:28992'})
    gdf_point['area'] = np.asarray(calc_area.geometry.area)

    if write2file == True:
        gdf_point.to_file(os.path.join(out_path, (os.path.basename(img_path)[-16:-4] + '_points.gpkg')), driver = 'GPKG')
        gdf_poly.to_file(os.path.join(out_path, (os.path.basename(img_path)[-16:-4] + '_poly.gpkg')), driver = 'GPKG')
    return gdf_point, gdf_poly

def multi2single(gpdf):
    gpdf = gpdf.drop(['area'], axis = 1)
    gpdf.geometry = gpdf.buffer(0)
    gpdf_dissolved = gpdf.dissolve('prediction')
    gpdf_singlepoly = gpdf_dissolved[gpdf_dissolved.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf_dissolved[gpdf_dissolved.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gpd.GeoDataFrame(row, crs=gpdf_multipoly.crs).T]*len(Series_geometries), ignore_index=True)
        df['geometry']  = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])

    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly

def append_gdfs(gdf1, gdf2):
    index_gdf2 = list(range(len(gdf1), len(gdf1)+len(gdf2)))
    gdf2.index = index_gdf2
    #gdf2.drop(['id'], axis = 1, inplace = True)
    gdf = pd.concat([gdf1, gdf2], sort = False)
    return gdf    
    

    
    
    