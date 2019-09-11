# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:57:17 2019

@author: ericv
"""

import os
import numpy as np
from PIL import Image
import rasterio

def array2tif(img_path, out_path, array, name_extension):
    src = rasterio.open(img_path)

    # Register GDAL format drivers and configuration options with a
    # context manager.
    with rasterio.Env():
    
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = src.profile
    
        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=rasterio.int8,
            count=1,
            compress='lzw',
            nodata = 0)
    
        with rasterio.open(os.path.join(out_path, os.path.basename(img_path)[:-4] + '_' + name_extension + '.tif'), 'w', **profile) as dst:
            dst.write(array.astype(rasterio.int8), 1)
    return print('array written as geotiff')

def resize(x):
    new_shape = (50,50,3)
    x_resized = np.array(Image.fromarray(x).resize((50,50)))
    #X_train_new = scipy.misc.imresize(x, new_shape)
    x_rescaled = x_resized/255
    return x_rescaled


