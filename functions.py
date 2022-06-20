import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
#from imageio import imread, imshow
#from skimage import data
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# Constants related to the video
VIDEO_PATH = '/home/javargas/2022-04-06 10-29-48.mkv'
FRAMES_LOCATION = '/home/javargas/Escritorio/images/'
IMG_EXT = '.jpg'

def get_entropy(n_frames:int=9):

    entropy_list = []
    frames_list = []
    
    for i in range(0, n_frames):
        # We get the image
        img = cv2.imread(f'{FRAMES_LOCATION}Frame{i}{IMG_EXT}')
        # Convert the image color to gray scales
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #entropy_image = entropy(img, disk(5))
        frames_list.append(i+1)
        entropy_list.append(entropy(img)) 
    
    return (frames_list, entropy_list)

def extract(video_path:str=VIDEO_PATH):
    """
    Function to extract frames from a video, save them to a specific path 
    and return the total number of frames.
    """
    # Cacth the video
    cap = cv2.VideoCapture(video_path)

    # Calculate the number of frames
    numbers_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Extracting frames from video...")

    # Iterate on each frame
    for i in range(0, numbers_of_frames):
        ret, frame = cap.read()
        
        # This condition prevents from infinite looping
        # incase video ends.
        if ret == False:
            break
        
        # Save Frame by Frame into a specific path using imwrite method
        cv2.imwrite(FRAMES_LOCATION+'Frame'+str(i)+IMG_EXT, frame)
        
    return numbers_of_frames

# Return the entropy of a image
def entropy(im):
    """
    Function that return the image entropy
    """
    # Compute normalized histogram -> p(g)
    p = np.array([(im==v).sum() for v in range(256)])
    p = p/p.sum()
    # Compute e = -sum(p(g)*log2(p(g)))
    e = -(p[p>0]*np.log2(p[p>0])).sum()
    
    return e

def find_locals(array:list):
    """
    Function that find minimum and maximum local peaks. It returns a list that
    contain the different values.
    """
    frame_entropy = dict(zip(list(range(1, len(array)+1)), array))

    new_array = np.array(array)    
   
    local_max = new_array[argrelextrema(new_array, np.greater)[0]]
    local_min = new_array[argrelextrema(new_array, np.less)[0]]
    Pext = list(local_max) + list(local_min)

    frames = []
    entropies = []

    for k, v in frame_entropy.items():
        if v in Pext:
            frames.append(k)
            entropies.append(v)

    return frames, entropies
   
def first_plot(frames_list:list, entropy_list:list):
    
    plt.figure("Representation Of The Video")
    plt.plot(frames_list, entropy_list, linestyle='--', marker='o', color='b', label='line with marker')
    plt.plot(frames_list, entropy_list, '--go', label='line with marker')
    plt.xlabel('Frame')
    plt.ylabel('Image Entropy')
    plt.title('Entropy Calculation')
    plt.xticks(frames_list)
    plt.legend(['Entropy value'], loc='lower right')
    plt.grid()
    plt.show()
    
    return 
    
def local_points_plot(frames_list:list, entropy_list:list, x_frames:list, y_peaks:list):

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
        
    return 

def normalize_data():
    
    return

def tranning_data():
    
    return 

def estimate_parameters():
    
    minPts = 4
    
    return 

def algorithm():
    x = np.array([x_frames, y_peaks])

    # Cluster identify
    #clusters = DBSCAN(eps=2, min_samples=2).fit(x)
    #labels = clusters.labels_
    clusters = DBSCAN(eps=2, min_samples=2).fit_predict(x)
    plt.show()