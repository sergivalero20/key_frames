import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
#from imageio import imread, imshow
from skimage import data

# Constants related to the video
VIDEO_PATH = '/home/javargas/2022-04-06 10-29-48.mkv'
FRAMES_LOCATION = '/home/javargas/Escritorio/images/'
IMG_EXT = '.jpg'

def extract():
    # Cacth the video
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Calculate the number of frames
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Extracting frames from video...")

    # Iterate on each frame
    for i in range(0, length):
        ret, frame = cap.read()
        
        # This condition prevents from infinite looping
        # incase video ends.
        if ret == False:
            break
        
        # Save Frame by Frame into a specific path using imwrite method
        cv2.imwrite(FRAMES_LOCATION+'Frame'+str(i)+IMG_EXT, frame)
        i += 1

def find_locals(array, tipo):

    new_array = np.array(array)
    if tipo=="maxima":
        # The next line return indices
        #argrelextrema(x, np.greater)
        result = new_array[argrelextrema(new_array, np.greater)[0]]
    elif tipo=="minima":
        # The next line return indices
        #argrelextrema(x, np.less)
        result = new_array[argrelextrema(new_array, np.less)[0]]
    return list(result)

# Return the entropy of a image
def entropy(im):
    # Compute normalized histogram -> p(g)
    p = np.array([(im==v).sum() for v in range(256)])
    p = p/p.sum()
    # Compute e = -sum(p(g)*log2(p(g)))
    e = -(p[p>0]*np.log2(p[p>0])).sum()
    
    return e
   