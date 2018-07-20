# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 20:41:47 2018

@author: Rishabh Sharma
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

StarBuck = cv2.imread("Starbucks-New-Logo.png")
gray = cv2.cvtColor(StarBuck,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

kp, des = sift.detectAndCompute(gray,None)

Ref_imge = cv2.imread("REFERENCE.jpg")
gray_ref = cv2.cvtColor(Ref_imge,cv2.COLOR_BGR2GRAY)

kp1,des2 = sift.detectAndCompute(gray_ref,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des,des2, k=2)

good = []
MIN_MATCH_COUNT = 10
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    Ref_imge = cv2.polylines(Ref_imge,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
    

cv2.imshow('image',Ref_imge)
cv2.imwrite('image.png',Ref_imge)
