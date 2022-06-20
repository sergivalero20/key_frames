#!/usr/bin/python3

from functions import (
            extract, 
            find_locals, 
            first_plot,
            get_entropy,
            local_points_plot
            )


# Let's extract the frames of the video 
n_frames = extract()

# Let's get frames and entropy values
frames_list, entropy_list = get_entropy(n_frames)   

# Let's plot  
first_plot(frames_list, entropy_list)  

# Let's find min and max local points
x_frames, y_peaks = find_locals(entropy_list)

# Let's plot peaks 
local_points_plot(frames_list, entropy_list, x_frames, y_peaks)

