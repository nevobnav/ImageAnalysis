# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:51:08 2019

@author: ericv
"""

#%%
#Import statements
import os
import sys
import time

import cv2
import numpy as np
import gdal

os.chdir(os.path.dirname(os.path.dirname(sys.argv[0])))
import Vector_functions.vector_functions
import Vector_functions.raster_functions
import Vector_functions.plant_count_functions
import Vector_functions.image_processing as ip
import Vector_functions.detect_plants as dp

import VanBovenProcessing.clip_ortho_2_plot_gdal

#%%
#Specify paths

#Input img_path
img_path = r"D:\VanBovenDrive\VanBoven MT\Archive\c03_termote\De Boomgaard\20190625\1112\Orthomosaic\c03_termote-De Boomgaard-201906251112-GR.tif"
#Output path
out_path = r'D:\800 Operational\c03_termote\De Boomgaard\20190625\1112' 
#Specify path to clip_shp
clip_shp = r"D:\VanBovenDrive\VanBoven MT\Archive\c03_termote\De Boomgaard\20190514\2005\Clip_shape\clip_shape.shp"

#%%
#Set variables for clustering

#Set lower and higher limits to plant size (px) 
min_area = 16
max_area = 1600

#Set no_data_value
no_data_value = 255

#Use small block size to cluster based on colors in local neighbourhood
x_block_size = 512
y_block_size = 512

#Create iterator to process blocks of imagery one by one and be able to process subsets.
it = list(range(0,200000, 1))

#True if you want to generate shapefile output with points and shapes of plants
process_full_image = True
#True if you want to fit the cluster centres iteratively to every image block, false if you fit in once to a random 10% subset of the entire ortho
iterative_fit = False
#True to clip ortho to clip_shape (usefull for removing grass around the fields)
clip_ortho2shp = True
#True to create a tif file of the plant mask of the image.
tif_output = True

#%%
#Set variables for local maxima plant count

#Determine factor for scaling parameters (for different image scales)
xpix,ypix = ip.PixelSize(img_path)
par_fact = 7.68e-5/(xpix*ypix)

#Sigma is smoothing factor higher sigma will result in smoother image and less detected local maxima
sigma = 8.0*par_fact
sigma_grass = 2.0*par_fact
neighborhood_size = 30*par_fact
#Threshold is a parameter specifying when to classify a local maxima as plant/significant. 
#Increasing threshold will result in less detected local maxima. Treshold and sigma interact 
threshold = 2.0

block_size = 3000
#%%
#Set initial cluster centres for clustering algorithm based on sampling in images
cover_lab = cv2.cvtColor(np.array([[[165,159,148]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
cover_init = np.array(cover_lab[0,0,1:3], dtype = np.uint8)
background_init = cv2.cvtColor(np.array([[[120,125,130]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from tif file
background_init = np.array(background_init[0,0,1:3])
green_lab = cv2.cvtColor(np.array([[[87,116,89]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
green_init = np.array(green_lab[0,0,1:3])

#Create init input for clustering algorithm
kmeans_init = np.array([background_init, green_init, cover_init])
#%%
#Run script

if __name__ == '__main__2':
    if clip_ortho2shp == True:
        ds = clip_ortho_2_plot_gdal.clip_ortho2shp_array(img_path, clip_shp)
    else:
        ds = gdal.Open(img_path)
    time_begin = time.time()

    #%% 
    #Run local maxima plant detector
    
    #Get information of image partition
    div_shape = ip.divide_image(img_path, block_size, remove_size=block_size)
    ysize, xsize, yblocks, xblocks, block_size = div_shape
    
    #Detect center of plants using local minima
    xcoord, ycoord = dp.DetectLargeImage(img_path, ds, div_shape, sigma, neighborhood_size, threshold, sigma_grass)
    #Write to shapefile
    gdf2 = vector_functions.coords2gdf(ds, xcoord, ycoord)
    
    #Create array with the positions of the coordinates
    arr_points = np.zeros([yblocks*block_size//10,xblocks*block_size//10], dtype='uint8')
    arr_points[(ycoord/10).astype(int),(xcoord/10).astype(int)] = 1
    arr_points = cv2.dilate(arr_points,np.ones((3,3),np.uint8),iterations = 1)
    
    #Detect lines using Hough Lines Transform
    lines = dp.HoughLinesP(arr_points, par_fact)
    #Write to shapefile
    lines = lines.reshape(lines.shape[0],4) * 10
    dp.WriteShapefileLines(ds, lines[:,0], lines[:,1], lines[:,2], lines[:,3], out_path)
    
    time_end = time.time()
    print('Total time: {}'.format(time_end-time_begin))
    #%%
    #
    plant_pixels, clustering_output = plant_count_functions.cluster_objects(x_block_size, y_block_size, ds, kmeans_init, iterative_fit, it, no_data_value)
    #write clustering output to tif to be able to inspect
    raster_functions.array2tif(img_path, out_path, clustering_output, name_extension = 'clustering_output')
    
    

    
