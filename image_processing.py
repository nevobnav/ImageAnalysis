# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:33:58 2019
Module for Image Processing

@author: Arthur
"""
import numpy as np
import cv2
import gdal
from skimage import filters
from skimage.transform import resize
import matplotlib.pyplot as plt
import os

def divide_image(ds, block_size, remove_size=50):
    band = ds.GetRasterBand(1)
    xsize = band.XSize
    ysize = band.YSize
    
    xblocks = int(np.ceil(xsize/block_size))
    yblocks = int(np.ceil(ysize/block_size))
    
    # If a block is just a small strip, then remove it
    if (xsize-block_size*(xblocks-1) < remove_size):
        xblocks -= 1
        xsize = block_size*xblocks
    if (ysize-block_size*(yblocks-1) < remove_size):
        yblocks -= 1
        ysize = block_size*yblocks
        
    print('The image is divided in {} blocks ({} x {}) of size {} x {}'.format(yblocks*xblocks,yblocks,xblocks,block_size,block_size))
    return (ysize, xsize, yblocks, xblocks, block_size)

def make_part_plot(img, img_part, factor, n, m, blk):
    img_small = Resize(img, factor)
    cols, rows = img_small.shape[:2]
    img_part[n*blk:n*blk+cols, m*blk:m*blk+rows] = img_small
    return img_part

def make_partition(img_path, block_size, data_path, img_name, remove_size = 1000, factor = 50):
    if not os.path.exists(data_path+'{}\\'.format(img_name)):
        os.makedirs(data_path+'{}\\'.format(img_name))
    div_shape = divide_image(img_path, block_size, remove_size)
    ds, ysize, xsize, yblocks, xblocks, block_size = div_shape
    
    blk = int(np.ceil(block_size/factor))
    img_plot = np.zeros([yblocks*blk,xblocks*blk,3], dtype = np.uint(8))
    for n in range(yblocks):
        for m in range(xblocks):
            try:
                img = read_image_part(img_path, div_shape, n, m)
                img_plot = make_part_plot(img, img_plot, factor, n, m, blk)
            except ValueError:
                print('Moving on to next block, since this block ({}, {}) is not part of the image'.format(n, m))
    plot_partition(img_plot, blk, yblocks, xblocks, data_path+'{}\\'.format(img_name))

def plot_partition(img, block, yblocks, xblocks, data_path):
    fig = plt.figure(figsize = (19, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    scl = min(1250/img.shape[1], 750/img.shape[0])
    img_RGB = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)
    ax.imshow(img_RGB, interpolation='bicubic')
    block_plot_arr = np.arange(0,xblocks*block+1,block)
    for line in block_plot_arr:
        ax.plot([line,line],[0,yblocks*block],color='C1',lw=3*scl)
    block_plot_arr = np.arange(0,yblocks*block+1,block)
    for line in block_plot_arr:
        ax.plot([0,xblocks*block],[line,line],color='C1',lw=3*scl)
    for yb in range(yblocks):
        for xb in range(xblocks):
            ax.text(xb*block+0.1*block,yb*block+0.5*block,'{}, {}'.format(yb,xb),fontsize=10*scl,color='lime')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    plt.savefig(data_path+'Partition.png',bbox_inches = 'tight',pad_inches=0, dpi=300)

def read_image_part(path, div_shape, n, m, ds):
    ysize, xsize, yblocks, xblocks, block_size = div_shape
    
    begin_n = n*block_size
    if n == yblocks-1:
        end_n = ysize
    else:
        end_n = (n+1)*block_size
        
    begin_m = m*block_size
    if m == xblocks-1:
        end_m = xsize
    else:
        end_m = (m+1)*block_size
        
    block_cols = end_m - begin_m
    block_rows = end_n - begin_n
    
    R = np.array(ds.GetRasterBand(1).ReadAsArray(begin_m, begin_n, block_cols, block_rows), dtype = np.uint(8))
    G = np.array(ds.GetRasterBand(2).ReadAsArray(begin_m, begin_n, block_cols, block_rows), dtype = np.uint(8))
    B = np.array(ds.GetRasterBand(3).ReadAsArray(begin_m, begin_n, block_cols, block_rows), dtype = np.uint(8))
    img = np.zeros([B.shape[0],B.shape[1],3], np.uint8)
    img[:,:,0] = B
    img[:,:,1] = G
    img[:,:,2] = R
    
    if np.all(img==255) or np.all(img==0):
        raise ValueError('This block is not part of the image.')
    return img

def ReadImagePartExtra(path, div_shape, n, m, extra, ds):
    """Returns image with extra boundary to the left and right"""
    ysize, xsize, yblocks, xblocks, block_size = div_shape
    
    if m==0:
        img = np.zeros([block_size,block_size+1*extra,3], dtype = np.uint(8))
        img[:,:,2] = np.array(ds.GetRasterBand(1).ReadAsArray(0, n*block_size, block_size+extra, block_size), dtype = np.uint(8))
        img[:,:,1] = np.array(ds.GetRasterBand(2).ReadAsArray(0, n*block_size, block_size+extra, block_size), dtype = np.uint(8))
        img[:,:,0] = np.array(ds.GetRasterBand(3).ReadAsArray(0, n*block_size, block_size+extra, block_size), dtype = np.uint(8))
        ext = np.zeros([block_size,extra,3], dtype = np.uint(8))
        img = np.uint8(np.concatenate((ext,img),axis=1))
    elif m==xblocks-1:
        img = np.zeros([block_size,block_size+1*extra,3], dtype = np.uint(8))
        img[:,:,2] = np.array(ds.GetRasterBand(1).ReadAsArray(m*block_size-extra, n*block_size, block_size+extra, block_size), dtype = np.uint(8))
        img[:,:,1] = np.array(ds.GetRasterBand(2).ReadAsArray(m*block_size-extra, n*block_size, block_size+extra, block_size), dtype = np.uint(8))
        img[:,:,0] = np.array(ds.GetRasterBand(3).ReadAsArray(m*block_size-extra, n*block_size, block_size+extra, block_size), dtype = np.uint(8))
        ext = np.zeros([block_size,extra,3], dtype = np.uint(8))
        img = np.uint8(np.concatenate((img,ext),axis=1))
    else:
        img = np.zeros([block_size,block_size+2*extra,3], dtype = np.uint(8))
        img[:,:,2] = np.array(ds.GetRasterBand(1).ReadAsArray(m*block_size-extra, n*block_size, block_size+2*extra, block_size), dtype = np.uint(8))
        img[:,:,1] = np.array(ds.GetRasterBand(2).ReadAsArray(m*block_size-extra, n*block_size, block_size+2*extra, block_size), dtype = np.uint(8))
        img[:,:,0] = np.array(ds.GetRasterBand(3).ReadAsArray(m*block_size-extra, n*block_size, block_size+2*extra, block_size), dtype = np.uint(8))
    
    # Check if block is really part of image or just an empty block
    if np.all(img==255) or np.all(img==0):
        raise ValueError('This block is not part of the image.')
    return np.uint8(img)

def ReadImageSpecific(path, x, y, cols, rows):
    # The x- and y-coordinates determine the upper left corner of the image
    ds = gdal.Open(path)
    R = np.array(ds.GetRasterBand(1).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
    G = np.array(ds.GetRasterBand(2).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
    B = np.array(ds.GetRasterBand(3).ReadAsArray(x, y, cols, rows), dtype = np.uint(8))
    img = np.zeros([B.shape[0],B.shape[1],3], np.uint8)
    img[:,:,0] = B
    img[:,:,1] = G
    img[:,:,2] = R
    return img

def ShowImage(img, img_title = None):
    "Show image (input: BGR image) with opencv"
    if np.max(img)==1:
        img = np.uint8(img*255)
    else:
        img = np.uint8(img)
    img_resize = cv2.resize(img, (int(img.shape[1]/img.shape[0]*1000), 1000))
    cv2.imshow(img_title, img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
def PlotImage(img):
    "Show image (input: BGR image) with matplotlib"
    dim = len(img.shape)
    if dim==3:
        img = img[:,:,[2,1,0]]
    my_dpi = 100
    figsize = img.shape[1]/my_dpi, img.shape[0]/my_dpi
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    if dim==2:
        ax.imshow(img, cmap=plt.cm.gray)
    else:
        ax.imshow(img)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axis('off')
    plt.show()
    cv2.waitKey(0)
    plt.close()
    
def Resize(img, factor, mask = None):
    "Resize image (and corresponding mask) without smoothing"
    new_shape_rows = img.shape[0] // factor
    new_shape_cols = img.shape[1] // factor
    img_resized = resize(img,(new_shape_rows, new_shape_cols),
                          mode='edge',
                          anti_aliasing=False,
                          anti_aliasing_sigma=None,
                          preserve_range=True,
                          order=0)
    if mask is not None:
        mask_resized = resize(mask,(new_shape_rows, new_shape_cols),
                              mode='edge',
                              anti_aliasing=False,
                              anti_aliasing_sigma=None,
                              preserve_range=True,
                              order=0)
        return img_resized, mask_resized
    return img_resized

def Otsu(img, rev = False):
    "Finds threshold with Otsu's method and returns binary image"
    try:
        val = filters.threshold_otsu(img)
        if rev:
            # Low intensity gets value True (white)
            return img < val
        else:
            # High intensity gets value True (white)
            return img > val
    except ValueError:
        return np.zeros_like(img, dtype=bool)
    
def Normalize(x):
    dim = len(x.shape)
    axis = tuple(range(dim-1))
    x -= x.mean(axis=axis)
    x /= x.std(axis=axis)
    return x

def Normalize2(x):
    dim = len(x.shape)
    axis = tuple(range(dim-1))
    x -= x.mean(axis=axis)
    x /= x.std(axis=axis)
    return x
    
def CalcDistance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 1000 * 6371 * c
    return m

def PixelSize(path):
    src = gdal.Open(path)
    gt = src.GetGeoTransform()
    lon1 = gt[0] 
    lat1 = gt[3] 
    lon2 = gt[0] + gt[1]*src.RasterXSize
    lat2 = gt[3] + gt[4]*src.RasterXSize
    dist = CalcDistance(lat1,lon1,lat2,lon2)
    ysize = dist/src.RasterXSize
    lon2 = gt[0] + gt[2]*src.RasterYSize
    lat2 = gt[3] + gt[5]*src.RasterYSize
    dist = CalcDistance(lat1,lon1,lat2,lon2)
    xsize = dist/src.RasterYSize
    return xsize, ysize