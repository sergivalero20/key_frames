import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import shutil
import os

from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# Constants related to the video
#VIDEO_PATH = '/home/sergio/Documents/tesis/key_frames/datasets/HandGesture/1/video1.avi'
FRAMES_LOCATION = '/home/sergio/Documents/tesis/images/'
KEY_FRAMES_LOC = '/home/sergio/Documents/tesis/final_result/'
IMG_EXT = '.jpg'

def get_directory(video_path:str)->str:
    
    if '/' in video_path:        
        dir = video_path.split('/')
        dir = list(filter(lambda x: x!='', dir))
        dir = '/'+'/'.join(dir[:-2])
        return dir
    return None

def get_entropy(n_frames:int, video_path:str)->list:
    
    """Returns a list that contain entropy values 

    Args:
        n_frames (int): An integer that represent the number of frames
        video_path (str): A string that represent the absolute path of the video

    Returns:
        list: A list that contain entropy values for each frame
    """

    frames_dir = os.path.join(video_path, 'Frames')
    entropy_list = []
    
    for i in range(1, n_frames + 1):
        # We get the image
        img = cv2.imread(f'{frames_dir}/Frame{i}{IMG_EXT}')
        # Convert the image color to gray scales
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #entropy_image = entropy(img, disk(5))
        entropy_list.append(entropy(img)) 
    
    return entropy_list

def extract(video_path:str)->int:
    
    """Returns the total numbers of frames 

    Args:
        video_path (str): A string that represent the absolute path of the video

    Returns:
        int: Integer that represent the total of frames of the video
    """
    # Cacth the video
    cap = cv2.VideoCapture(video_path)

    # Calculate the number of frames
    numbers_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Extracting frames from video...")
    frames_dir = os.path.join(get_directory(video_path), 'Frames')
    os.mkdir(frames_dir)
    # Iterate on each frame
    for i in range(1, numbers_of_frames + 1):
        ret, frame = cap.read()
        
        # This condition prevents from infinite looping
        # incase video ends.
        if ret == False:
            break
        
        # Save Frame by Frame into a specific path using imwrite method
        cv2.imwrite(frames_dir+'/Frame'+str(i)+IMG_EXT, frame)
        
    return numbers_of_frames

# Return the entropy of a image
def entropy(im:cv2.cvtColor)->float:
    
    """Returns the entropy of an image 

    Args:
        im (cv2.cvtColor): An image

    Returns:
        float: A float that represents the image entropy
    """
    # Compute normalized histogram -> p(g)
    p = np.array([(im==v).sum() for v in range(256)])
    p = p/p.sum()
    # Compute e = -sum(p(g)*log2(p(g)))
    e = -(p[p>0]*np.log2(p[p>0])).sum()
    
    return e

def find_locals(entropy_values:list):
    
    """Compute the minimum and maximum locals points 

    Args:
        entropy_values (list): A list that contain entropy values

    Returns:
        A Tuple: it returns a tuple that contain frames and entropy values lists
    """
    
    #frame_entropy = dict(zip(list(range(1, len(entropy_values)+1)), entropy_values))
    frame_entropy = {i+1:v for i, v in enumerate(entropy_values)}

    new_array = np.array(entropy_values)    
   
    local_max = new_array[argrelextrema(new_array, np.greater)[0]]
    local_min = new_array[argrelextrema(new_array, np.less)[0]]
    Pext = list(local_max) + list(local_min)

    frames = []
    entropies = []

    for k, v in frame_entropy.items():
        if v in Pext:
            frames.append(k)
            entropies.append(v)

    return (frames, entropies)

def list_to_df(frames:list, entropy:list):
    
    """Returns a DataFrame

    Args:
        frames (list): A list that contain i frames
        entropy (list): A list that contain entropy values

    Returns:
        DataFrame: A DataFrame that contain frames and entropy values in order
    """

    frames_entropy = []
    
    for i, value in enumerate(frames):
        frames_entropy.append((value, entropy[i]))
    
    df = pd.DataFrame(frames_entropy, columns=('Frame', 'Entropy'))
    return df

def get_max_curvature(distances:list)->float:
    
    """Returns a float

    Args:
        distances (list): A list that contains the average distance between each point in the data set

    Returns:
        float: A float that represents epsilon value for the algorithm 
    """
    
    step = 2
    result = []
    
    for index, value in enumerate(distances):
        count, acc = 0.0, 0.0
        for step1 in range(1,step+1):
            if index-step1>0:
                temp_v = distances[index-step1]
                count += 1
                acc += abs(distances[index] - temp_v)
        for step1 in range(1,step+1):
            if index+step1<len(distances):
                temp_v = distances[index+step1]
                count+=1
                acc += abs(distances[index] - temp_v)
        result.append(acc/count)
    idx = result.index(max(result))
    eps = distances[idx]
    print(f"This is the value of eps: {eps}")
    return eps

def estimate_value(df:pd.core.frame.DataFrame)->float:
    
    """Returns a list

    Args:
        df (DataFrame): A DataFrame that represents all the points

    Returns:
        float: A float that represents epsilon value for the algorithm 
    """
    
    # creating an object of the NearestNeighbors class
    neighb = NearestNeighbors(n_neighbors=2)
    
    # fitting the data to the object
    nbrs=neighb.fit(df)
    
    # finding the nearest neighbours
    distances,indices=nbrs.kneighbors(df)
    
    # Sort the distances results
    distances = np.sort(distances, axis = 0)
    distances = distances[:, 1]    
    eps = get_max_curvature(distances)
    
    plt.rcParams['figure.figsize'] = (5,3)
    plt.plot(distances)
    plt.show()
    
    return eps
    
   
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

def algorithm(df:pd.core.frame.DataFrame, Eps:float, n_points:int=4)->list:
    
    """Returns a list

    Args:
        df (DataFrame): A DataFrame that represents all the points
        Eps (Float): Epsilon that represents the size of area to consider a dense region
        n_points (int): The total number of points in a dense region

    Returns:
        list: A list that containts all the cluster find in the data
    """
    
    dbscan = DBSCAN(eps = Eps, min_samples = n_points).fit(df)
    labels = dbscan.labels_
    return labels
    plt.figure("DBSCAN Algorithm")
    plt.scatter(df.Frame, df.Entropy, c = labels, cmap= "plasma")
    plt.xlabel("Frame")
    plt.ylabel("Entropy")
    plt.xticks(list(range(1,239+1)))
    plt.tick_params(axis='x', which='minor', labelsize='small', labelcolor='m', rotation=30)
    #plt.setp(plt.xticks(list(range(1,239+1))), horizontalalignment='right')
    
    plt.show()
    
def divide_clusters(clusters:list)->dict: 
    
    """Returns a Dict

    Args:
        clusters (list): A list that contain all cluster without segmentation

    Returns:
        Dict: A dict that contain clusters number as key and a points list as value
    """
    
    clusters_points = {}
    
    for i, label in enumerate(clusters):
        # We take all points except outliers
        if label!=-1:
            if label in clusters_points:
                clusters_points[label].append(i)
            else:
                clusters_points[label] = [i]
    print(f"This are the clusters found it: {clusters_points}")
    return clusters_points   

def get_den_point(positions:list, df)->int:
    
    """Returns a list

    Args:
        positions (list): A list that contain points

    Returns:
        Int: A integer that represents the most dense point in a cluster
    """
    
    densities = []
    
    for i, current_point in enumerate(positions):
        a = (df.Frame[current_point], df.Entropy[current_point])
        density = 0
        for j, next_point in enumerate(positions):
            if j!=i:
                b = (df.Frame[next_point], df.Entropy[next_point])
                # Compute the Euclidean distance between two points a and b.
                density += math.dist(a, b)
        densities.append((current_point, density))
    densities = sorted(densities, key=lambda d: d[1])
    return densities[0][0]

def find_key_frames(df, clusters:list)->list:
    
    """Returns a list

    Args:
        df (DataFrame): A DataFrame that contain all points 
        clusters (list): A list that contain all cluster without segmentation

    Returns:
        list: A list that contain all the key frames
    """
    
    clusters_points = divide_clusters(clusters)
    key_frames = []
    
    for points in clusters_points.values():
        key_frames.append(df.Frame[get_den_point(points, df)])
        
    return sorted(key_frames)

def move_key_frames(key_frames:list, video_path:str)->str:

    """Returns the path of the key frames

    Args: 
        key_frames (list): A list that contains all the key frames
        video_path (str): A string that represent the absolute path of the video 

    Returns:
        str: A string that represents the path of the key frames
    """
    key_frames_dir = os.path.join(video_path, 'Key_Frames')
    os.mkdir(key_frames_dir)
    frames_dir = os.path.join(video_path, 'Frames')
    
    for key_frame in key_frames:
        src_path = f'{frames_dir}/Frame{key_frame}{IMG_EXT}'
        dst_path = f'{key_frames_dir}/Frame{key_frame}{IMG_EXT}'
        shutil.move(src_path, dst_path)
    return key_frames_dir