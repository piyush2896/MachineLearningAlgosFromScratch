import cv2
import numpy as np
from kMeans import KMeansClustering
from matplotlib import pyplot as plt

# Read image using openCV
im = cv2.imread('./img.jpg')

# print shape of the image
print(im.shape)

# Convert image to RGB from default BGR
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# resize image
# make sure to resize according to shape of image
im = cv2.resize(im, (im.shape[1]//3, im.shape[0]//3))

# Display image
plt.figure(0)
plt.imshow(im)
plt.show()

# Reshape image into a 2 dimensional matrix
all_pixels = im.reshape(im.shape[0]*im.shape[1], 3)

# Fit it in KMeansClustering
km = KMeansClustering()
km.fit(all_pixels, nIter = 20, k = 7)

# Plt the most prominent k colors in image
i = 1
plt.figure(0)
colors = []
for each_col in km._centers:
	
    plt.subplot(1, 8, i)    
	plt.axis("off")
	
    i+= 1
	
    col = each_col.astype('uint8')
	colors.append(255-col)
	
    a = np.zeros((100, 100, 3))
	a[:, :, :] = col
	
    plt.imshow(255-a)
plt.show()
print(colors)

# New image will be made by changing colors with their labeled colors.
img = np.zeros((im.shape[0]*im.shape[1], 3))
for ix in range(img.shape[0]):
	img[ix] = colors[km._labels[ix]]
img = img.reshape((im.shape[0], im.shape[1], 3))
plt.figure(0)
plt.imshow(img)

# save image
plt.savefig('out_img.jpg', bbox_inches='tight', pad_inches=0)

plt.show()