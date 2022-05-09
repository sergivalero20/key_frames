#!/usr/bin/python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions import (
    entropy, 
    extract, 
    find_locals, 
    find_locals_2,
    FRAMES_LOCATION, 
    IMG_EXT
)
from sklearn.cluster import DBSCAN
#from skimage.filters.rank import entropy
#from matplotlib.pyplot import imread
#from skimage.io import imread
#from skimage.morphology import disk


"""
    When we have video frames then we could follow the next steps:
    1. Calculate image entropy of each frame
    2  Append image entropy values in a list
    3. Find local peaks
    4. Save 
"""

# Extract the frames of the video 
#n_frames = extract()

entropy_list = []
frames_list = []

# We map each frame and calculate its entropy
for i in range(0, 9):
    # We get the image
    img = cv2.imread(f'{FRAMES_LOCATION}Frame{i}{IMG_EXT}')
    # Convert the image color to gray scales
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #entropy_image = entropy(img, disk(5))
    frames_list.append(i+1)
    entropy_list.append(entropy(img))     

# We find local values
#max_peaks = find_locals(entropy_list, "maxima")
#min_peaks = find_locals(entropy_list, "minima")

x_frames, y_peaks = find_locals_2(entropy_list)

#plt.figure("Representation Of The Video")
#plt.plot(frames_list, entropy_list, linestyle='--', marker='o', color='b', label='line with marker')
#plt.plot(frames_list, entropy_list, '--go', label='line with marker')
#plt.xlabel('Frame')
#plt.ylabel('Image Entropy')
#plt.title('Entropy Calculation')
#plt.xticks(frames_list)
#plt.legend(['Entropy value'], loc='lower right')
#plt.grid()
#plt.show()



fig, axs = plt.subplots(1, 2)
axs[0].plot(frames_list, entropy_list, '--go', label='line with marker')
axs[0].set_xticks(frames_list)
axs[0].set_yticks(entropy_list)
axs[0].grid()
axs[0].set_title('Entropy Calculation')
axs[1].plot(x_frames, y_peaks, '--go', label='line with marker')
axs[1].set_xticks(x_frames)
axs[1].set_yticks(y_peaks)
axs[1].grid()
axs[1].set_title('Peaks Selection')

for ax in axs.flat:
    ax.set(xlabel='Frame', ylabel='Image Entropy')

for ax in axs.flat:
    ax.label_outer()


x = np.array([x_frames, y_peaks])

# Cluster identify
#clusters = DBSCAN(eps=2, min_samples=2).fit(x)
#labels = clusters.labels_
clusters = DBSCAN(eps=2, min_samples=2).fit_predict(x)
plt.show()