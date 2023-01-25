#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 08:01:32 2022

@author: mocquais
"""

import numpy as np
import cv2 as cv2

def max_radius(circles):
    max_r = 0
    for circle in circles:
        max_r = max(max_r, circle[2])
    return max_r

def img_treatment(img):
    dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    #norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    bin_img = cv2.threshold(diff_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return bin_img

def create_matrix(circles):
    
    radius = max_radius(circles[0].tolist())
    
    circles_y = circles[0].tolist()
    circles_y.sort(key = lambda x: x[1])
    circles_x = circles[0].tolist()
    circles_x.sort(key = lambda x: x[0])
    
    y_bound = 0
    line_counter = -1
    column_counter = -1
    x_bound = 0
    
    matrix_coordinates = {}
    
    for circle in circles_y:
        if circle[1] > y_bound :
            line_counter += 1
        y_bound = circle[1] + 2*radius
        
        matrix_coordinates[str(circle)] = {"ligne":line_counter,"column":-1}
        
    for circle in circles_x:
        if circle[0] > x_bound :
            column_counter += 1
        x_bound = circle[0] + 2*radius
        
        matrix_coordinates[str(circle)]["column"] = column_counter
        
    matrix = np.array([[[0.0,0.0,0.0]]*(column_counter+1)]*(line_counter+1))
    
    for circle in matrix_coordinates:
        matrix[matrix_coordinates[circle]["ligne"]][matrix_coordinates[circle]["column"]] = circle.strip("][").split(", ")
    
    return matrix


def detect_circles(img):
    print(img)
    bin_img = img_treatment(img)
    circles = cv2.HoughCircles(bin_img,cv2.HOUGH_GRADIENT,1.5,55, param1=200,param2=18,minRadius=8,maxRadius=19)
    
    return create_matrix(circles)
    


        
