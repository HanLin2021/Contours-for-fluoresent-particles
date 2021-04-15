# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:12:06 2021

@author: Han
"""
import cv2
import numpy as np
import matplotlib
# from matplotlib.pyplot import imsho
from matplotlib import pyplot as plt
from PIL import Image
import os
# im = Image.open(infile)
#        print "new filename : " + outfile
#        out = im.convert("RGB")
#        out.save(outfile, "JPEG", quality=90)

def analyze_imag(RGB):
    mask=RGB
    thresh_min,thresh_max = 165,255
    ret,thresh = cv2.threshold(mask,thresh_min,thresh_max,0)
    thresh_bw = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(thresh_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    p_=[]
    for c in contours:
        if(cv2.arcLength(c,True)>350):
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(mask,(x,y),(x+w,y+h),(255,0,0),2)      
            p_.append(cv2.boundingRect(c))
    p_=np.transpose(p_)
    plt.figure(0,figsize=(5,5),dpi=500)
    plt.imshow(mask)
    # plt.title("max: %d" % (p_[2]),size=15)
    plt.text(100, 150, max(p_[2]), fontsize='large',color='red')
    plt.show()
 
    # plt.hist(p_[2])
    # plt.show()
    return max(p_[2])

for fname in os.listdir():
    print(fname)
    if('T0' in fname):    
        im = cv2.imread(fname)
        # plt.imshow(im)
        # plt.show()
        w=analyze_imag(im)
        print(w)
