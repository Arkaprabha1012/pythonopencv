import numpy as np
import cv2
from PIL import Image
name=input("enter the filename:")
img1=cv2.imread(name)
img2=img1.copy()
imgray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
(thresh,bwINVimage)=cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)
(thresh,bwimage)=cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
ret,bw=cv2.threshold(imgray,127,255,0)
con,th=cv2.findContours(bwINVimage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours,hier=cv2.findContours(bwimage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow('image gray',imgray)
#cv2.drawContours(img1, contours, -1, (0,255,0), 3)
#cnt = contours[]
#cv2.drawContours(img, [cnt], 0, (0,255,0), 5)

cv2.imshow('image main',img1)
#cv2.imshow('image bw',bwimage)
i=0
cnt1=con
ep1=0.1*cv2.arcLength(cnt1,True)
ap1=cv2.approxPolyDP(cnt1,ep1,True)
image_crop=imgray[ap1[0,0,1]:ap1[2,0,1],ap1[0,0,0]:ap1[2,0,0]]
cv2.imshow("crop",image_crop)	
print("no of shapes {0}".format(len(contours)))
for cnt in contours:
	i=i+1
	rect=cv2.minAreaRect(cnt)
	box=cv2.boxPoints(rect)
	box=np.int0(box)
	image=cv2.drawContours(img2, [box],-1, (0,179,44), 2)
	#print("area",i,"=",cv2.contourArea(cnt))
	#print("arclength",i,"=",(cv2.arcLength(cnt,True)))
	epsilon=0.005*cv2.arcLength(cnt,True)
	approx=cv2.approxPolyDP(cnt,epsilon,True)
	appimage=cv2.drawContours(img1, [approx],-1, (0,179,44), 2)
cv2.imshow("contours box",image)
cv2.imshow("approx",appimage)
cannyimg=cv2.Canny(bwimage,100,200)
cv2.imshow("canny image",cannyimg)
#cv2.imshow('bwinvimage',bwINVimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
