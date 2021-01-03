import glob   
import cv2
import os
import numpy as np
path = '/home/basu/Desktop/study materials/handwritten digit recognition/progrmas and samples and outputs/resizedthresh/*.jpg'
from pathlib import Path
files=glob.glob(path) 
i=-1
for file in files:
	i=i+1     
	#print(file)
	img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
	image=cv2.imread(file,0)
	n_white_pix = np.sum(img == 255)
	count = cv2.countNonZero(image)
	print('Number of white pixels '+str(i)+'.jpg:', n_white_pix)
	print('Number of black pixels '+str(i)+'.jpg:',count)
	ratio=count/n_white_pix
	ratio=round(ratio,5)
	print('Number of white pixels '+str(i)+'.jpg:',ratio)
cv2.waitKey(0)
