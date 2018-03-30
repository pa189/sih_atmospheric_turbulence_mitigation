import cv2
import math


vidcap = cv2.VideoCapture('1.mp4')
framerate = vidcap.get(5)
print(framerate)
success,image = vidcap.read()
count = 0
success = True
while vidcap.isOpened():
  frameId = vidcap.get(1)
  success,image = vidcap.read()
  if success != True:
    break
  if frameId % math.floor(framerate) == 0:
    print('Read a new frame: ', success)
    cv2.imwrite("frame%d.png" % count, image)     # save frame as JPEG file
  count += 1