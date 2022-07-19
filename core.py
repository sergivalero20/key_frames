#!/usr/bin/python3

# Databases: Cambridge Hand Gesture, Northwesten University Hand Gesture , Hand Gesture

import time
import sys

from functions import (
            algorithm,
            estimate_value,
            extract, 
            find_locals, 
            find_key_frames,
            first_plot,
            get_directory,
            get_entropy,
            list_to_df,
            local_points_plot,
            move_key_frames
            )

def execute_algorithm(video_path):
    #start_time = time.time()
    dir_path = get_directory(video_path)
    # 1) Let's extract the frames of the video 
    n_frames = extract(video_path)
    #print("--- {} seconds ---".format(time.time() - start_time))
    # 2) Let's get frames and entropy values
    entropy_list = get_entropy(n_frames, dir_path)   

    # Let's plot  
    # first_plot(frames_list, entropy_list)  

    # 3) Let's find min and max local points
    x_frames, y_peaks = find_locals(entropy_list)

    # 4) Let's obtain the DataFrame 
    df = list_to_df(x_frames, y_peaks)

    # Let's plot peaks 
    # local_points_plot(frames_list, entropy_list, x_frames, y_peaks)

    # 5) Let's estimate algorithm parameters
    Eps = estimate_value(df)

    # 6) Let's get cluster
    clusters = algorithm(df, Eps)

    # 7) Finally get key frames
    key_frames = find_key_frames(df, clusters)

    # 8) Move key frames to other path
    move_key_frames(key_frames, dir_path)

    print(f"Final result: {key_frames}")

if __name__ == '__main__':

    video_path = sys.argv[1]
    execute_algorithm(video_path)

