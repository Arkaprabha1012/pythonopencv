import cv2 
import numpy as np
import os
from pathlib import Path
import shutil
#CRACKED IT ON JAN 21
dirpath = Path('cutted_pics')
filename=input("enter the filename:")
image = cv2.imread(filename)
img2=image.copy()
height=img2.shape[0]
width=img2.shape[1]
channels=img2.shape[2]
normalizedImg = np.zeros((800, 800))
cv2.imshow("main image",img2)
img2 = cv2.normalize(img2,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
#cv2.imshow("normalised image",img2)
blurred1=cv2.pyrMeanShiftFiltering(img2,21,111)
#cv2.imshow("pyrmeans image",blurred1)
blurred=cv2.medianBlur(img2,5)
#cv2.imshow("medianblur image",blurred)
blurred=cv2.medianBlur(blurred1,3)
#cv2.imshow("pyrmeans medianblur image",blurred1)
gray=cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 10, 250)
#cv2.imshow("canny",edged)
(cnts,hi1) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
region,hi=cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i= 0
matrix=[]
j=-1
#path for output cutted images
if dirpath.exists() and dirpath.is_dir():
	shutil.rmtree(dirpath)
	os.mkdir('cutted_pics')
	
else:
	os.mkdir('cutted_pics')
os.chdir('cutted_pics')
digit=[]
for i in region:
	#x,y,w,h =matrix[i]
	x,y,w,h = cv2.boundingRect(i)	
	digit.append(i)
	imag = cv2.rectangle(blurred,(x,y),(x+w,y+h),(0,0,255),2)	
	new_img=image[y:y+h,x:x+w]
	
	cv2.imwrite(str(i) +'.png', new_img)
cv2.imshow("countous box",imag)
cv2.waitKey(0)
