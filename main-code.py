#import necessary libraries
from pathlib import Path
from PIL import Image
import os, shutil
from os import listdir

#image resizing
from PIL import Image
import numpy as np

#load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot

#Mac OS created a hidden file called .DS_Store which interfered with my data processing. I had to delete this hidden file.
for root, dirs, files in os.walk('/Users/jinchoi725/Desktop/fashionimages'):
    i = 0
    for file in files:
        if file.endswith('.DS_Store'):
            path = os.path.join(root, file)

            print("Deleting: %s" % (path))

            if os.remove(path):
                print("Unable to delete!")
            else:
                print("Deleted...")
                i += 1

print("Files Deleted: %d" % (i))

#import in a list format the fashion image RGB files
Input_dir = '/Users/jinchoi725/Desktop/fashionimages/'
Out_dir = '/Users/jinchoi725/Desktop/gr_fashionimages/'
a = os.listdir('/Users/jinchoi725/Desktop/fashionimages/')

#convert all the RGB files into grayscale files
for i in a:
    try:
        print(i)
        I = Image.open('/Users/jinchoi725/Desktop/fashionimages/'+i)
        L = I.convert('L')
        L.save('/Users/jinchoi725/Desktop/gr_fashionimages/'+i)
    except:
        pass










