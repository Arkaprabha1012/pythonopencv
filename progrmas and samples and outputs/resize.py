import glob   
import cv2
import os
import shutil
import numpy as np
path = '/home/basu/Desktop/study materials/handwritten digit recognition/progrmas and samples and outputs/cutted_pics/*.png'
from pathlib import Path
files=glob.glob(path) 
i=-1
dirpath = Path('resized')
if dirpath.exists() and dirpath.is_dir():
	shutil.rmtree(dirpath)
	os.mkdir('resized')
	
else:
	os.mkdir('resized')

dirpath = Path('resizedthresh')
if dirpath.exists() and dirpath.is_dir():
	shutil.rmtree(dirpath)
	os.mkdir('resizedthresh')
	
else:
	os.mkdir('resizedthresh')

os.chdir('cutted_pics')


for file in files:
	i=i+1     
	#print(file)
	img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
	os.chdir('..')
	print('Original Dimensions : ',img.shape)
 
	scale_percent = 10 # percent of original size
	#width = int(img.shape[1] * scale_percent / 100)
	#height = int(img.shape[0] * scale_percent / 100)
	width=60
	height=60
	dim = (width, height)
	# resize image
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	print('Resized Dimensions : ',resized.shape)
	os.chdir('resized')  
	#cv2.imshow("Resized image", resized)
	
	kernel = np.ones((5,5), np.uint8) 
	resized= cv2.dilate(resized, kernel, iterations=1)
	ret,resizedbin=cv2.threshold(resized,127,255,cv2.THRESH_BINARY)
	cv2.imwrite(str(i)+'.jpg',resized)
	os.chdir("..")
	os.chdir('resizedthresh')
	cv2.imwrite(str(i)+'Bin.jpg',resizedbin)
	
cv2.waitKey(0)
