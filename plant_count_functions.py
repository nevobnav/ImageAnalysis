# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:48:07 2019

@author: ericv
"""

import os
import time
import math


import cv2
import numpy as np
import scipy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd
import random
from ImageAnalysis.raster_functions import resize

def fit_kmeans_on_subset(ds, kmeans_init):
    #get raster
    r = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype = np.uint(8))
    img = np.zeros([r.shape[0],r.shape[1],3], np.uint8)
    img[:,:,2] = r
    r = None
    g = np.array(ds.GetRasterBand(2).ReadAsArray(), dtype = np.uint(8))
    img[:,:,1] = g
    g = None
    b = np.array(ds.GetRasterBand(3).ReadAsArray(), dtype = np.uint(8))
    img[:,:,0] = b
    b = None

    #take random subset of image nog toevoegen maar nu geen internet    
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    Lab_flat = Lab.flatten()
    Lab = None
    
    subset = scipy.random.choice(Lab_flat, size = (len(Lab_flat)//10), replace = False)
    Lab_flat = None
    #get a and b band of subset
    a = subset[:,1]
    b = subset[:,2]
    subset = None

    #stack bands to create final input for clustering algorithm
    Classificatie_Lab = np.column_stack((a, b))
    a = None
    b = None

    tic = time.time()
    #perform kmeans clustering
    kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 50, n_clusters = kmeans_init.shape[0], verbose = 0, precompute_distances = False)
    kmeans.fit(Classificatie_Lab)
    toc = time.time()

    print('Processing took ' + str(toc-tic) + ' seconds')
    return kmeans

def cluster_objects(x_block_size, y_block_size, ds, kmeans_init, iterative_fit, it, no_data_value):

    #fit kmeans to random subset of entire image if False
    if iterative_fit == False:
        kmeans = fit_kmeans_on_subset(ds, kmeans_init)
    #time process
    tic = time.time()

    #get dimensions of image
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize

    #create template for img mask resulting from clustering algorithm
    plant_pixels = np.zeros([ysize, xsize], np.uint8)
    clustering_output = plant_pixels.copy()
    
    #define kernel for morhpological closing operation
    kernel = np.ones((3,3), dtype='uint8')

    #iterate through img using blocks to reduce memory consumption and make function less vulnarable to changes in lighting
    blocks = 0
    for y in range(0, ysize, y_block_size):
        if y > 10:
            y = y - 10 # use -30 pixels overlap to prevent "lines at the edges of blocks in object detection"
        if y + y_block_size < ysize:
            rows = y_block_size
        else:
            rows = ysize - y

        for x in range(0, xsize, x_block_size):
            if x > 10:
                x = x - 10
            blocks += 1
            #if statement for subset
            if blocks in it:
                if x + x_block_size < xsize:
                    cols = x_block_size
                else:
                    cols = xsize - x
                #read bands as array
                r = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                g = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                b = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
                img = np.zeros([b.shape[0],b.shape[1],3], np.uint8)
                img[:,:,0] = b
                img[:,:,1] = g
                img[:,:,2] = r
                r = None
                g = None
                b = None
                #check if block of img has values
                if (img.mean() < 255) and (img.mean() > 0): 
                    #create img mask
                    #use no data value to create mask, make sure it is 255
                    gray = np.ma.masked_equal(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), no_data_value)
                    mask = gray.mask
                    mask_flat = mask.flatten()
    
                    #convert img to CieLAB colorspace
                    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    
                    #get a and b bands
                    a = np.array(img_lab[:,:,1])
                    b2 = np.array(img_lab[:,:,2])
    
                    #flatten the bands for kmeans clustering algorithm
                    a_flat = a.flatten()
                    b2_flat = b2.flatten()
    
                    #stack bands to create final input for clustering algorithm
                    Classificatie_Lab = np.column_stack((a_flat, b2_flat))
    
                    #fit kmeans to data distribution of block if iterative fit == True
                    if iterative_fit == True:
                        kmeans = KMeans(init = kmeans_init, n_jobs = -1, max_iter = 25, n_clusters = kmeans_init.shape[0], verbose = 0)
                        if len(mask_flat) == len(Classificatie_Lab):
                            y_kmeans = kmeans.fit_predict(Classificatie_Lab[mask_flat == False])
                            kmeans_result = np.zeros((a_flat.shape))
                            kmeans_result[mask_flat == False] = y_kmeans
                            y_kmeans = kmeans_result.copy()
                        else:
                            y_kmeans = kmeans.fit_predict(Classificatie_Lab)
                    else:
                        #dont fit on image block, only classify
                        y_kmeans = kmeans.predict(Classificatie_Lab)
                    
                    #get different classes
                    unique, counts = np.unique(y_kmeans, return_counts=True)
    
                    #Get cluster centres
                    centres = kmeans.cluster_centers_
    
                    #get index of the greenest cluster centre
                    get_green = np.argmax(centres[:,1] - centres[:,0])
                    get_background = np.argmin(centres[:,1])
                    get_remaining = np.argmax(centres[:, 0])
    
                    #create binary output, green is one the rest is zero
                    y_kmeans[y_kmeans == get_green] = 10
                    y_kmeans[y_kmeans == get_background] = 5
                    y_kmeans[(y_kmeans < 5)] = 0
    
                    #convert binary output back to 8bit image
                    kmeans_img = y_kmeans
                    kmeans_img = kmeans_img.reshape(img.shape[0:2]).astype(np.uint8)
                    ret,binary_img = cv2.threshold(kmeans_img,9,255,cv2.THRESH_BINARY)
                    clustering_result = kmeans_img * 25
    
                    #optional erode to deal with overlap
                    #binary_img = cv2.erode(binary_img, kernel2, iterations = 1)  
    
                    #close detected shapes
                    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
                    #write img block result back on original sized image template
                    plant_pixels[y:y+rows, x:x+cols] = plant_pixels[y:y+rows, x:x+cols] + closing
                    clustering_output[y:y+rows, x:x+cols] = clustering_result
                    #print('processing of block ' + str(blocks) + ' finished')

    plant_pixels[plant_pixels > 0] = 255        
    toc = time.time()
    print("processing of blocks took "+ str(toc - tic)+" seconds")

    return plant_pixels, clustering_output

def merge_close_points(gdf, list_of_distances):
    #time processing
    tic = time.time()
    
    # filter for valid points, empties are rejected
    gdf = gdf.loc[(gdf.geom_type == 'MultiPoint') | (gdf.geom_type == 'Point')]
              
    #convert to cartesian coordinates
    gpdf = gdf.to_crs({'init': 'epsg:28992'})
    
    #add column to store if a point should be stored or removed
    gpdf['store'] = 0
    
    #reset index
    gpdf.index = gpdf.index.dropna()
    gpdf = gpdf.reset_index(drop = True)
    
    gdf = None
    
    for min_distance in list_of_distances:
        #get coordinates of points
        coord = gpdf.geometry.centroid
        coord = coord.apply(lambda x:x.coords.xy)
        X = np.array(list(coord.apply(lambda x:tuple((x[0][0],x[1][0])))))
        
        #calculate distances to neighbouring points
        knn = NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=8, metric='euclidean', radius=min_distance).fit(X)
        
        #get distances and indices of neighbouring points
        distances, indices = knn.kneighbors(X)
        distances = distances[:,1:]
        indices = indices[:,1:]
        
        #create lists of points to store and to remove    
        remove_list = []
        store_list = []
        
        #check for each point if neighbours are to close and subsequently which one to keep
        for i in gpdf.index:#['index']:
            dists = np.array(distances[i,:])        
            inds = np.array(indices[i,:])
            #get neighbours that are closer than the min_distance
            neighbours = list(inds[np.where(dists < min_distance)])
            if len(neighbours) > 0:
                neighbours.append(i)
                areas = gpdf.loc[neighbours, 'area']
                #add neighbours that are to close to list to remove
                remove_list.append(neighbours)
                if math.isnan(areas.idxmax()) == False:
                    #add the point with the largest are to the list of points to keep
                    store_list.append(areas.idxmax())
                else:
                    store_list.append(math.nan)
        #compare the list with points to remove and points to store and create final list with all points that have to be removed
        remove = np.unique(np.concatenate([np.array(j) for j in remove_list]))
        store = np.unique(np.array(store_list))
        remove_final = remove[np.isin(remove, store) == False]
        #remove points
        gpdf.loc[remove_final, 'store'] = 1
        gpdf = gpdf.loc[gpdf['store'] == 0]
        #reset index again
        gpdf = gpdf.loc[(gpdf.geom_type == 'MultiPoint') | (gpdf.geom_type == 'Point')]
        gpdf.index = gpdf.index.dropna()
        gpdf = gpdf.reset_index(drop = True)
        print(str(min_distance) + ' is processed')
    
    #convert coordinates back to lat lon    
    gpdf = gpdf.to_crs({'init': 'epsg:4326'})
    
    toc = time.time()
    print('removing double counts took ' + str(toc-tic) + ' seconds.')
    return gpdf

def contours2shp(plant_pixels, out_path, min_area, max_area, ds, run_classification = False):

    #Get contours of features
    contours, hierarchy = cv2.findContours(plant_pixels, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #create df with relevant data
    df = pd.DataFrame({'contours': contours})

    contours = None
    hierarchy = None

    df['area'] = df.contours.apply(lambda x:cv2.contourArea(x))
    #filter features, area has to be > 0
    df = df[(df['area'] > min_area) & (df['area'] < max_area)]
    df['moment'] = df.contours.apply(lambda x:cv2.moments(x))
    df['centroid'] = df.moment.apply(lambda x:(int(x['m01']/x['m00']),int(x['m10']/x['m00'])))
    df['bbox'] = df.contours.apply(lambda x:cv2.boundingRect(x))

    #classification is still under construction
    if run_classification == True:
        #read entire img to memory
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize

        #initiate output img
        output = np.zeros([ysize,xsize,3], np.uint8)
        r = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.uint(8))
        output[:,:,0] = r
        r = None
        g = np.array(ds.GetRasterBand(2).ReadAsArray()).astype(np.uint(8))
        output[:,:,1] = g
        g = None
        b = np.array(ds.GetRasterBand(3).ReadAsArray()).astype(np.uint(8))
        output[:,:,2] = b
        b = None
        print('Loaded image into memory')
        
        print('Start with classification of objects')
       
        #create input images for model
        df['input'] = df.bbox.apply(lambda x:output[x[1]-5: x[1]+x[3]+5, x[0]-5:x[0]+x[2]+5])

        #remove img from memory
        output = None

        df = df[df.input.apply(lambda x:x.shape[0]*x.shape[1]) > 0]
        #covnert to hsv for classification and resize 
        df['input'] = df.input.apply(lambda x:resize(cv2.cvtColor(x, cv2.COLOR_BGR2HSV)))
        #resize data to create input tensor for model
        #df['input'] = df.input.apply(lambda x:resize(x))

        model_input = np.asarray(list(df.input.iloc[:]))
        #predict
        tic = time.time()
        try:
            prediction = model.predict(model_input)

            #get prediction result
            pred_final = prediction.argmax(axis=1)
            #add to df
            df['prediction'] = pred_final
        except:
            print('no prediction')

        toc = time.time()
        print('classification of '+str(len(df))+' objects took '+str(toc - tic) + ' seconds')
    if run_classification == False:
        #df['output'] = np.nan
        df['input'] = np.nan
    return df
        
def add_random_area_column(gdf):
    gdf['area'] = 0
    gdf['area'] = gdf['area'].apply(lambda x: random.uniform(x, 0.00001))
    return gdf
    
