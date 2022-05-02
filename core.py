#!/usr/bin/python3
import cv2
import matplotlib.pyplot as plt
from functions import (
    entropy, 
    extract, 
    find_locals, 
    FRAMES_LOCATION, 
    IMG_EXT
)
#from skimage.filters.rank import entropy
from matplotlib.pyplot import imread
#from skimage.io import imread
#from skimage.morphology import disk


"""
    When we have video frames then we could follow the next steps:
    1. Calculate image entropy of each frame
    2  Append image entropy values in a list
    3. Find local peaks
    4. Save 
"""

# Now let us import the image we will be working with.

entropy_list = []
frames_list = []

for i in range(0, 5):
    # We get the image
    img = cv2.imread(f'{FRAMES_LOCATION}Frame{i}{IMG_EXT}')
    # Convert the image color to gray scales
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #entropy_image = entropy(img, disk(5))
    frames_list.append(i+1)
    entropy_list.append(entropy(img))     

#plt.figure(num=None, figsize=(8,62))
plt.figure("Representation Of The Video")
plt.plot(frames_list,entropy_list, "o")
plt.xlabel('Frames')
plt.ylabel('Entropy')
plt.title('Frames with its entropy values')
plt.xticks(frames_list)

#print(entropy_list)
max_peaks = (find_locals(entropy_list, "maxima"))
min_peaks = (find_locals(entropy_list, "minima"))
plt.legend(['Entropy value'], loc='lower right')
plt.grid()
plt.show()

