import cv2

# Constants related to the video
VIDEO_PATH = '/home/javargas/2022-04-06 10-29-48.mkv'
FRAMES_LOCATION = '/home/javargas/Escritorio/images/'
IMG_EXT = '.jpg'

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