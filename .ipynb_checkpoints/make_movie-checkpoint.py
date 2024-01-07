import cv2
import os
import re

# helper function to perform sort
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

def make_movie(run_ID): # Convert nuclei locations into stopmotion, can be changed to make stopmotion of error progression which may be useful
    image_folder = f'{run_ID}/{run_ID}_nuc_locs'
    video_name = f'{run_ID}/{run_ID}_stopmotion.mp4'
    
    images = [img for img in os.listdir(image_folder) if img.startswith("nuc_locs")]
    images.sort(key=num_sort) 

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 5, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()