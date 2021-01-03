import cv2
import numpy as np
import os
from pathlib import Path
import shutil
#CRACKED IT ON JAN 21
dirpath = Path('cutted_pics')
img = cv2.imread('sample11.jpg')
cv2.imshow("main image",img)
normalizedImg = np.zeros((800, 800))
cv2.imshow("main image",img)
height=img.shape[0]
width=img.shape[1]
img2 = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
# Gaussian Blurring
blur = cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow("blur",blur)
# Apply Sobelx in high output datatype 'float32'
# and then converting back to 8-bit to prevent overflow
sobelx_64 = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)
absx_64 = np.absolute(sobelx_64)
sobelx_8u1 = absx_64/absx_64.max()*255
sobelx_8u = np.uint8(sobelx_8u1)
 
# Similarly for Sobely
sobely_64 = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
absy_64 = np.absolute(sobely_64)
sobely_8u1 = absy_64/absy_64.max()*255
sobely_8u = np.uint8(sobely_8u1)
 
# From gradients calculate the magnitude and changing
# it to 8-bit (Optional)
mag = np.hypot(sobelx_8u, sobely_8u)
mag = mag/mag.max()*255
mag = np.uint8(mag)
 
# Find the direction and change it to degree
theta = np.arctan2(sobely_64, sobelx_64)
angle = np.rad2deg(theta)
#cv2.imshow("theta",theta)
#cv2.imshow("angle",angle)
M, N = mag.shape
Non_max = np.zeros((M,N), dtype= np.uint8)
 
for i in range(1,M-1):
    for j in range(1,N-1):
       # Horizontal 0
        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
            b = mag[i, j+1]
            c = mag[i, j-1]
        # Diagonal 45
        elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
            b = mag[i+1, j+1]
            c = mag[i-1, j-1]
        # Vertical 90
        elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
            b = mag[i+1, j]
            c = mag[i-1, j]
        # Diagonal 135
        elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
            b = mag[i+1, j-1]
            c = mag[i-1, j+1]           
            
        # Non-max Suppression
        if (mag[i,j] >= b) and (mag[i,j] >= c):
            Non_max[i,j] = mag[i,j]
        else:
            Non_max[i,j] = 0
# Set high and low threshold
highThreshold = 21
lowThreshold = 15
 
M, N = Non_max.shape
out = np.zeros((M,N), dtype= np.uint8)
 
# If edge intensity is greater than 'High' it is a sure-edge
# below 'low' threshold, it is a sure non-edge
strong_i, strong_j = np.where(Non_max >= highThreshold)
zeros_i, zeros_j = np.where(Non_max < lowThreshold)
 
# weak edges
weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))
 
# Set same intensity value for all edge pixels
out[strong_i, strong_j] = 255
out[zeros_i, zeros_j ] = 0
out[weak_i, weak_j] = 75
M, N = out.shape
for i in range(1, M-1):
    for j in range(1, N-1):
        if (out[i,j] == 75):
            if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
                out[i, j] = 255
            else:
                out[i, j] = 0
cv2.imshow("final",out)
(cnts,hi1) = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
region,hi=cv2.findContours(out.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
digit=[]
for i in range(0,len(matrix)):
	x,y,w,h =matrix[i]
	digit.append(i)
	imag = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)	
	new_img=img[y:y+h,x:x+w]
	
	cv2.imwrite(str(i) +'.png', new_img)
cv2.imshow("countous box",imag)
cv2.waitKey(0)
cv2.waitKey(0)
