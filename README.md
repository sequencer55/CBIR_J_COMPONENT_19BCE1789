# Video link for the implementation:
https://drive.google.com/file/d/1r7I9rjoRHiSLQBx756aYz7JrEca4FzrZ/view?usp=sharing

# Starting a Kaggle notebook

Go to [Kaggle.com](https://www.kaggle.com/) and make an account and initiate a notebook.



## Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from collections import Counter
import os
print(os.listdir("../input"))
```
## Import training data
```python
train = pd.read_csv("../input/train.csv")
print(train.head())
## Display threshold image
```
## Map of the targets in the dictionary
```python
subcell_locs = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles",
5:  "Nuclear bodies",
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus",
8:  "Peroxisomes",
9:  "Endosomes",
10:  "Lysosomes",
11:  "Intermediate filaments",   
12:  "Actin filaments",
13:  "Focal adhesion sites",   
14:  "Microtubules",
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle",
18:  "Microtubule organizing center",  
19:  "Centrosome",
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions", 
23:  "Mitochondria",
24:  "Aggresome",
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}
```


## Separation Channels
```python
print("The image with ID == 1 has the following labels:", train.loc[1, "Target"])
print("These labels correspond to:")
for location in train.loc[1, "Target"].split():
    print("-", subcell_locs[int(location)])

#reset seaborn style
sns.reset_orig()

#get image id
im_id = train.loc[1, "Id"]

#create custom color maps
cdict1 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict2 = {'red':   ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict3 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0))}

cdict4 = {'red': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

plt.register_cmap(name='greens', data=cdict1)
plt.register_cmap(name='reds', data=cdict2)
plt.register_cmap(name='blues', data=cdict3)
plt.register_cmap(name='yellows', data=cdict4)

#get each image channel as a greyscale image (second argument 0 in imread)
green = cv2.imread('../input/train/{}_green.png'.format(im_id), 0)
red = cv2.imread('../input/train/{}_red.png'.format(im_id), 0)
blue = cv2.imread('../input/train/{}_blue.png'.format(im_id), 0)
yellow = cv2.imread('../input/train/{}_yellow.png'.format(im_id), 0)

#display each channel separately
fig, ax = plt.subplots(nrows = 2, ncols=2, figsize=(15, 15))
ax[0, 0].imshow(green, cmap="greens")
ax[0, 0].set_title("Protein of interest", fontsize=18)
ax[0, 1].imshow(red, cmap="reds")
ax[0, 1].set_title("Microtubules", fontsize=18)
ax[1, 0].imshow(blue, cmap="blues")
ax[1, 0].set_title("Nucleus", fontsize=18)
ax[1, 1].imshow(yellow, cmap="yellows")
ax[1, 1].set_title("Endoplasmic reticulum", fontsize=18)
for i in range(2):
    for j in range(2):
        ax[i, j].set_xticklabels([])
        ax[i, j].set_yticklabels([])
        ax[i, j].tick_params(left=False, bottom=False)
plt.show()
```

## Differentiate the Organelles in the cell images
```python
#stack nucleus and microtubules images
#create blue nucleus and red microtubule images
nuclei = cv2.merge((np.zeros((512, 512),dtype='uint8'), np.zeros((512, 512),dtype='uint8'), blue))
microtub = cv2.merge((red, np.zeros((512, 512),dtype='uint8'), np.zeros((512, 512),dtype='uint8')))

#create ROI
rows, cols, _ = nuclei.shape
roi = microtub[:rows, :cols]

#create a mask of nuclei and invert mask
nuclei_grey = cv2.cvtColor(nuclei, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(nuclei_grey, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

#make area of nuclei in ROI black
red_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

#select only region with nuclei from blue
blue_fg = cv2.bitwise_and(nuclei, nuclei, mask=mask)

#put nuclei in ROI and modify red
dst = cv2.add(red_bg, blue_fg)
microtub[:rows, :cols] = dst

#show result image
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(microtub)
ax.set_title("Nuclei (blue) + microtubules (red)", fontsize=15)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(left=False, bottom=False)
```
## Labelling and targetting the organelles as per our need.
```python
labels_num = [value.split() for value in train['Target']]
labels_num_flat = list(map(int, [item for sublist in labels_num for item in sublist]))
labels = ["" for _ in range(len(labels_num_flat))]
for i in range(len(labels_num_flat)):
    labels[i] = subcell_locs[labels_num_flat[i]]

fig, ax = plt.subplots(figsize=(15, 5))
pd.Series(labels).value_counts().plot('bar', fontsize=14)
```

## Applying threshold on the nucleus image

```python
ret, thresh = cv2.threshold(blue, 0, 255, cv2.THRESH_BINARY)
```
## Displaying the threshold image
```python
fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
ax[0].imshow(thresh, cmap="Greys")
ax[0].set_title("Threshold", fontsize=15)
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].tick_params(left=False, bottom=False)

#morphological opening to remove noise
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
ax[1].imshow(opening, cmap="Greys")
ax[1].set_title("Morphological opening", fontsize=15)
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].tick_params(left=False, bottom=False)
```

## Marker labelling
```python
ret, markers = cv2.connectedComponents(opening)
Map component labels to hue val
label_hue = np.uint8(179 * markers / np.max(markers))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img[label_hue==0] = 0
ax[2].imshow(labeled_img)
ax[2].set_title("Markers", fontsize=15)
ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].tick_params(left=False, bottom=False)
```
## Apply threshold on the endoplasmic reticulum image
```python
ret, thresh1 = cv2.threshold(yellow, 4, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(yellow, 4, 255, cv2.THRESH_TRUNC)
ret, thresh3 = cv2.threshold(yellow, 4, 255, cv2.THRESH_TOZERO)
```

## Display threshold images
```python
fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
ax[0].imshow(thresh1, cmap="Greys")
ax[0].set_title("Binary", fontsize=15)

ax[1].imshow(thresh2, cmap="Greys")
ax[1].set_title("Trunc", fontsize=15)

ax[2].imshow(thresh3, cmap="Greys")
ax[2].set_title("To zero", fontsize=15)
```
## Plotting subplots
```python
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
#morphological opening to remove noise after binary thresholding
kernel = np.ones((5,5),np.uint8)
opening1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
ax[0].imshow(opening1, cmap="Greys")
ax[0].set_title("Morphological opening (binary)", fontsize=15)
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].tick_params(left=False, bottom=False)

#morphological closing after binary thresholding
closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)
ax[1].imshow(closing1, cmap="Greys")
ax[1].set_title("Morphological closing (binary)", fontsize=15)
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].tick_params(left=False, bottom=False)

#morphological opening to remove noise after truncate thresholding
kernel = np.ones((5,5),np.uint8)
opening2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
ax[2].imshow(opening2, cmap="Greys")
ax[2].set_title("Morphological opening (truncate)", fontsize=15)
ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].tick_params(left=False, bottom=False)

#morphological closing after truncate thresholding
closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel)
ax[3].imshow(closing2, cmap="Greys")
ax[3].set_title("Morphological closing (truncate)", fontsize=15)
ax[3].set_xticklabels([])
ax[3].set_yticklabels([])
ax[3].tick_params(left=False, bottom=False)

fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
# Marker labelling for binary thresholding
ret, markers1 = cv2.connectedComponents(closing1)
# Map component labels to hue val
label_hue1 = np.uint8(179 * markers1 / np.max(markers1))
blank_ch1 = 255 * np.ones_like(label_hue1)
labeled_img1 = cv2.merge([label_hue1, blank_ch1, blank_ch1])
# cvt to BGR for display
labeled_img1 = cv2.cvtColor(labeled_img1, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img1[label_hue1==0] = 0
ax[0].imshow(labeled_img1)
ax[0].set_title("Markers (binary)", fontsize=15)
ax[0].set_xticklabels([])
ax[0].set_yticklabels([])
ax[0].tick_params(left=False, bottom=False)

# Marker labelling for truncate thresholding
ret, markers2 = cv2.connectedComponents(closing2)
# Map component labels to hue val
label_hue2 = np.uint8(179 * markers2 / np.max(markers2))
blank_ch2 = 255 * np.ones_like(label_hue2)
labeled_img2 = cv2.merge([label_hue2, blank_ch2, blank_ch2])
# cvt to BGR for display
labeled_img2 = cv2.cvtColor(labeled_img2, cv2.COLOR_HSV2BGR)
# set bg label to black
labeled_img2[label_hue2==0] = 0
ax[1].imshow(labeled_img2)
ax[1].set_title("Markers (truncate)", fontsize=15)
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].tick_params(left=False, bottom=False)
```
## Applying adaptive threshold on endoplasmic reticulum image
```python
y_blur = cv2.medianBlur(yellow, 3)

#apply adaptive thresholding
ret,th1 = cv2.threshold(y_blur, 5,255, cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(y_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)

th3 = cv2.adaptiveThreshold(y_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

#display threshold images
fig, ax = plt.subplots(ncols=3, figsize=(20, 20))
ax[0].imshow(th1, cmap="Greys")
ax[0].set_title("Binary", fontsize=15)

ax[1].imshow(th2, cmap="Greys_r")
ax[1].set_title("Adaptive: mean", fontsize=15)

ax[2].imshow(th3, cmap="Greys_r")
ax[2].set_title("Adaptive: gaussian", fontsize=15)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
