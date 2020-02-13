from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import PIL
import PIL.ExifTags
import csv
import re

directory = r"G:\Shared drives\Recordings"
csv_dest = r""

GPSInfo = 34853
Alt = 6

dirs = [f for f in listdir(directory) if isdir(join(directory, f))]
data = []
ddirs = [join(directory, d,f) for d in dirs for f in listdir(join(directory, d)) if isdir(join(directory, d, f)) ]

for i,d in enumerate(ddirs[:1]):
    paths = [f for f in listdir(d) if isfile(join(d, f)) if "JPG" in f]
    pbar2 = tqdm(desc=f'{i + 1}/{len(ddirs) + 1} of directories', total= len(paths), position = 0)
    for p in paths:
        im = PIL.Image.open(join(d, p))        
        a = float(im._getexif()[GPSInfo][Alt][0]/1000)
        if a > 1 and a < 15:
            data.append((((re.compile(r'(c+[0-9]+[0-9]_)').split(d))[-1]).split('\\')[0], join(d, p), "low flight"))
        elif a > 15:
            data.append((((re.compile(r'(c+[0-9]+[0-9]_)').split(d))[-1]).split('\\')[0], join(d, p), "high flight"))
        pbar2.update(1)

with open(csv_dest, "w") as file:
    writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Costumer', 'Absolute path', 'Type of flight'])
    for d in data:
        writer.writerow(list(d))