# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:04:59 2019
Module for detecting plants using local minima

@author: Arthur
"""
import sys

import ImageAnalysis.image_processing as ip

import numpy as np
from scipy import ndimage
from skimage.transform import downscale_local_mean
import cv2
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString
from affine import Affine
import matplotlib.pyplot as plt

def DetectMinima(img, sigma, neighborhood_size, threshold):
    img_gauss = ndimage.gaussian_filter(img, sigma=sigma)
    img_gauss_min = ndimage.filters.minimum_filter(img_gauss, neighborhood_size)
    img_gauss_max = ndimage.filters.maximum_filter(img_gauss, neighborhood_size)
    minima = (img_gauss == img_gauss_min)
    diff = ((img_gauss_max - img_gauss_min) > threshold)
    minima[diff == 0] = 0
    labeled, num_objects = ndimage.label(minima)
    xy = np.array(ndimage.center_of_mass(img, labeled, range(1, num_objects+1)))
    return np.transpose(xy)
    
def FilterCoordinates(xy, block_size, extra):
    mask = (xy[1,:] >= extra) & (xy[1,:] < block_size+extra)
    mask_arr = np.ma.array(xy[1,:],mask=mask)
    x_arr = xy[1,:][mask_arr.mask == True]-extra
    y_arr = xy[0,:][mask_arr.mask == True]
    return x_arr, y_arr

def HoughLinesP(arr, par_fact, rho=1, theta=np.pi/180, threshold=20,
                min_line_length=40, max_line_gap=5):
    # rho: distance resolution in pixels of the Hough grid
    # theta: angular resolution in radians of the Hough grid
    # threshold: minimum number of votes (intersections in Hough grid cell)
    # min_line_length: minimum number of pixels making up a line
    # max_line_gap: maximum gap in pixels between connectable line segments
    # line_image: creating a blank to draw lines on
    lines = cv2.HoughLinesP(arr, rho, theta, int(round(threshold*par_fact)), np.array([]),
                            int(round(min_line_length*par_fact)),
                            int(round(max_line_gap*par_fact)))
    # Output "lines" is an array containing endpoints of detected line segments
    return lines

def DetectLargeImage(img_path, ds, div_shape, sigma, neighborhood_size, threshold, sigma_grass):
    ysize, xsize, yblocks, xblocks, block_size = div_shape
    
    # Define size extra border for overlapping
    extra = 100
    
    # Initialize lists to store x- and y-coordinates
    xcoord = []
    ycoord = []
    
    # Loop over all the block of image
    for n in range(yblocks):
        for m in range(xblocks):
            try:
                # Read part of image
                print('Block: n={}, m={}'.format(n,m))
                img_extended = ip.ReadImagePartExtra(img_path, div_shape, n, m, extra, ds)
                # Convert to a* channel and Cb channel
                img_a = cv2.cvtColor(img_extended, cv2.COLOR_BGR2Lab)[:,:,1]
                img_gauss = np.uint8(ndimage.gaussian_filter(downscale_local_mean(
                        img_extended[:,extra:-extra,:],(10,10,1)), sigma=(sigma_grass,sigma_grass,0.0)))

                # Get binary image with 0 (grass) and 1 (other)
                img_gauss_Cb_bin = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2Lab)[:,:,2] < 140
                img_gauss = None
                
                # Remove holes in grass
                img_temp = np.zeros((img_gauss_Cb_bin.shape[0]+2,img_gauss_Cb_bin.shape[1]+2),dtype=bool)
                img_temp[1:-1,1:-1] = np.invert(img_gauss_Cb_bin)
                img_gauss_Cb_bin = np.invert(ndimage.morphology.binary_fill_holes(img_temp))[1:-1,1:-1]
                img_temp = None
                
# =============================================================================
#                 img_plot = ip.Otsu(cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,1],rev=True)
#                 grass_plot = resize(img_gauss_Cb_bin,img_plot.shape)
#                 img_p = np.multiply(img_plot, grass_plot)
#                 ip.show_image(img_p)
# =============================================================================
                
                # Detect plants using local minima
                try:
                    xy = DetectMinima(img_a, sigma, neighborhood_size, threshold)
                    x_arr, y_arr = FilterCoordinates(xy, block_size, extra=100)
                except IndexError:
                    print('Algorithm found zero points')
                    continue
                
                img_a = None
                
                #ip.ShowImage(img_gauss_Cb_bin)
                # Sort x_arr and y_arr in unison
                sort_idx = y_arr.argsort()
                y_arr_sort = y_arr[sort_idx]
                x_arr_sort = x_arr[sort_idx]
                
                # Create array with the positions of the coordinates
                pos_arr = np.zeros((block_size//10,block_size//10),dtype=int)
                pos_arr[np.array(y_arr_sort//10,dtype=int),np.array(x_arr_sort//10,dtype=int)] = np.arange(0,len(x_arr),1,dtype=int)+1
                
                # Determine which coordinates belong to plants
                plant_pos = np.multiply(pos_arr,img_gauss_Cb_bin)
                pos = np.nonzero(plant_pos)
                x_plant = x_arr_sort[plant_pos[pos]-1]
                y_plant = y_arr_sort[plant_pos[pos]-1]
                
                xcoord.extend(x_plant+m*block_size)
                ycoord.extend(y_plant+n*block_size)
                
            except ValueError:
                print('Moving on to next block, this block ({}, {}) is not part of the image'.format(n, m))
    
    return np.array(xcoord), np.array(ycoord)

def WriteShapefilePoints(ds, xcoord, ycoord, output_path):
    coord_lst = [rasterio.transform.xy(transform = Affine.from_gdal(*ds.GetGeoTransform()), rows = ycoord[i], cols = xcoord[i], offset='ul') for i in range(len(xcoord))]
    gdf_point = gpd.GeoDataFrame(geometry = [Point(x, y) for x, y in coord_lst], crs = {'init': 'epsg:4326'})
    gdf_point.to_file(output_path +'/Points_localmax.shp')
    
def WriteShapefileLines(ds, xcoord1, ycoord1, xcoord2, ycoord2, output_path):
    coord_lst1 = [rasterio.transform.xy(transform = Affine.from_gdal(*ds.GetGeoTransform()), rows = ycoord1[i], cols = xcoord1[i], offset='ul') for i in range(len(xcoord1))]
    coord_lst2 = [rasterio.transform.xy(transform = Affine.from_gdal(*ds.GetGeoTransform()), rows = ycoord2[i], cols = xcoord2[i], offset='ul') for i in range(len(xcoord2))]
    gdf_line = gpd.GeoDataFrame(geometry = [LineString([(coord_lst1[i][0], coord_lst1[i][1]), (coord_lst2[i][0], coord_lst2[i][1])]) for i in range(len(coord_lst1))], crs = {'init': 'epsg:4326'})
    gdf_line.to_file(output_path + '/Lines.shp')

def PlotPoints(img_path,div_shape, x_plant,y_plant):
    my_dpi = 100
    ds, ysize, xsize, yblocks, xblocks, block_size = div_shape
    img = downscale_local_mean(ip.read_image_specific(img_path, 0, 0, xblocks*block_size, yblocks*block_size), (10, 10, 1))
    figsize = img.shape[1]/my_dpi, img.shape[0]/my_dpi
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.scatter(x_plant,y_plant,s=0.5,color='lime')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    plt.savefig('Points.png',bbox_inches='tight',pad_inches=0,dpi=300)
    plt.close()
    
def PlotLines(img, lines):
    my_dpi = 100
    figsize = img.shape[1]/my_dpi, img.shape[0]/my_dpi
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img,cmap=plt.cm.gray,vmin=0,vmax=1)
    for line in lines:
        for x1,y1,x2,y2 in line:
            ax.plot([x1,x2],[y1,y2],lw=1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    plt.savefig('Lines.png',bbox_inches='tight',pad_inches=0,dpi=my_dpi)
    plt.close()