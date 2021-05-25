# import necessary libraries
from pathlib import Path
from PIL import Image
import os, shutil
from os import listdir


# image resizing
from PIL import Image
import numpy as np


# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot


# Mac OS created a hidden file called .DS_Store which interfered with my data processing. I had to delete this hidden file.
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


# import in a list format the fashion image RGB files
Input_dir = '/Users/jinchoi725/Desktop/fashionimages/'
Out_dir = '/Users/jinchoi725/Desktop/gr_fashionimages/'
a = os.listdir('/Users/jinchoi725/Desktop/fashionimages/')


# convert all the RGB files into grayscale files. In cases of errors, skip them.
for i in a:
    try:
        print(i)
        I = Image.open('/Users/jinchoi725/Desktop/fashionimages/'+i)
        L = I.convert('L')
        L.save('/Users/jinchoi725/Desktop/gr_fashionimages/'+i)
    except:
        pass

    
# Mac OS created a hidden file called .DS_Store which interfered with my data processing. I had to delete this hidden file, AGAIN.
for root, dirs, files in os.walk('/Users/jinchoi725/Desktop/gr_fashionimages'):
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


# The grayscaled images have to be resized to the same dimension for K-Means clustering.
dim = (80, 60) # This is the dimension that I chose.
X_image_train = []
for fname in listdir(Out_dir):
    fpath = os.path.join(Out_dir, fname)
    im = Image.open(fpath)
    im_resized = im.resize(dim) # This is where the image files are resized
    X_image_train.append(im_resized) # The resized image files are appended to a list object X_image_train.
   

# Using the numpy library, I converted each image file to a 2 dimentional numpy array. 
X_image_array=[]
for x in range(len(X_image_train)):
    X_image=np.array(X_image_train[x],dtype='uint8')
    X_image_array.append(X_image) # The numpy arrays are appended to a list object X_image_array.


# Checking the size of a single numpy array
X_image_array[0].shape
X_image_array[15].shape


# Using np.stack, I stacked the numpy arrays along a new axis. 
# While X_image_array was simply a squence of 2D numpy arrays, the stacked one is one 3D numpy array with three axes.
X_final = np.stack(X_image_array)
X_final


# As a result the shape of X_final is now (44441, 60, 80) with one more value which indicates the added axis (Z).
X_final.shape #(44441, 60, 80)


# A 2D numpy array in the 3D stacked numply array can still be displayed as a visual image.
pyplot.imshow(X_final[0])
pyplot.show()




















