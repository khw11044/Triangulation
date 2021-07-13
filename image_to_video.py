import os
import argparse

import cv2
import numpy as np
from glob import glob


parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--video_name', default='plt_triangulation', type=str,
                    help='videos or image files')
args = parser.parse_args()
root=args.video_name


print(root)
pathOutput = 'output2.mp4'
fps = 15

def get_frames(video_name):
    images = glob(os.path.join(video_name, '*.png*'))
    images = sorted(images, key=lambda x: x.split('/')[-1].split('.')[0])
        # key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for img in images:
        frame = cv2.imread(img)
        yield frame

def main():
    name = root.split('/')[-1].split('.')[0]
    frame_array = []
    for frame in get_frames(root):
        height, width, layers = frame.shape 
        size = (width, height)
        frame_array.append(frame)

        # image resize
        #cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(name, 480, 640)
        # image show
        cv2.imshow(name, frame)
        cv2.waitKey(40)
    out = cv2.VideoWriter(pathOutput, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

if __name__=="__main__":
    main()