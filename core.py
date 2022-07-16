#!/usr/bin/python3

# Databases: Cambridge Hand Gesture, Northwesten University Hand Gesture , Hand Gesture

from functions import (
            algorithm,
            estimate_values,
            extract, 
            find_locals, 
            find_key_frames,
            first_plot,
            get_entropy,
            list_to_df,
            local_points_plot
            )


# Let's extract the frames of the video 
n_frames = extract()

# Let's get frames and entropy values
entropy_list = get_entropy(n_frames)   

# Let's plot  
# first_plot(frames_list, entropy_list)  

# Let's find min and max local points
x_frames, y_peaks = find_locals(entropy_list)

# Let's obtain the DataFrame 
df = list_to_df(x_frames, y_peaks)

# Let's plot peaks 
# local_points_plot(frames_list, entropy_list, x_frames, y_peaks)

# Let's estimate algorithm parameters
Eps = estimate_values(df)

# Let's get cluster
clusters = algorithm(df, Eps)

# Finally get key frames
key_frames = find_key_frames(df, clusters)

print(f"Final result: {key_frames}")

