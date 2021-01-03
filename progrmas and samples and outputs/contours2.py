import cv2 
import numpy as np
import os
from pathlib import Path
import shutil
import imutils
#CRACKED IT ON JAN 21
dirpath = Path('cutted_pics')
filename=input("enter the filename:")
image = cv2.imread(filename)
image = imutils.resize(image, width=600)
img2=image.copy()
height=img2.shape[0]
width=img2.shape[1]
channels=img2.shape[2]
normalizedImg = np.zeros((800, 800))
img2 = cv2.normalize(img2,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
blurred=cv2.pyrMeanShiftFiltering(img2,21,111)
blurred=cv2.medianBlur(blurred,5)
gray=cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(image, 10, 250)
cv2.imshow("main image",img2)
(cnts,hi1) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
region,hi=cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i= 0
matrix=[]
j=-1
for c in cnts:
	x,y,w,h = cv2.boundingRect(c)
	if ((h<6 and w<6) or(w==width) or(h==height)):
		continue
	i+=1
	j=j+1
	c1=c		
	x,y,w,h = cv2.boundingRect(c1)	
	matrix.append([])	
	matrix[j].append(int(x))
	matrix[j].append(int(y))
	matrix[j].append(int(w))
	matrix[j].append(int(h))
r=[]
for i in range(0,len(matrix)):
	for j in range(0,len(matrix)):
		if((matrix[i][0]<matrix[j][0]) or (matrix[i][0]==matrix[j][0])):
			xi=matrix[i][0]
			yi=matrix[i][1]
			wi=matrix[i][2]
			hi=matrix[i][3]
			xj=matrix[j][0]
			yj=matrix[j][1]
			wj=matrix[j][2]
			hj=matrix[j][3]
			if((yi<yj) or (yi==yj)):
				if(((xi+wi)>(xj+wj) or (xi+wi)==(xj+wj)) and (((yi+hi)>(yj+hj)))):
					r.append(j)
				elif((yi+hi)==(yj+hj) and (xi+wi)>(xj+wj)):
					r.append(j)
# Python code to remove duplicate elements 
def Remove(duplicate): 
	final_list = [] 
	for num in duplicate: 
		if num not in final_list: 
			final_list.append(num) 
	return final_list
r=Remove(r)
for e in sorted(r, reverse=True):
    del matrix[e]
print(matrix)
#path for output cutted images
if dirpath.exists() and dirpath.is_dir():
	shutil.rmtree(dirpath)
	os.mkdir('cutted_pics')
	
else:
	os.mkdir('cutted_pics')
os.chdir('cutted_pics')
for i in range(0,len(matrix)):
	x,y,w,h =matrix[i]
	image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
	new_img=image[y:y+h,x:x+w]
	
	cv2.imwrite(str(i) +'.png', new_img)
cv2.imshow("countous box",image)
cv2.waitKey(0)
