#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 08:01:32 2022

@author: mocquais
"""

from keras.datasets import mnist
from pathlib import Path
from random import randrange
import numpy as np
import cv2 as cv2
import imutils as imutils

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

def dataset_creator(dataset_type,mnist_data,mnist_labels):

    numbers = {key:[] for key in [1,2,3,4,5,6,7,8]}
    
    for i in range(len(mnist_labels)):
        label = mnist_labels[i]
        if label != 0 and label != 9:
            mnist_data[i][mnist_data[i] == 255] = 254
            img = cv2.threshold(
                 mnist_data[i],
                100,255,cv2.THRESH_TOZERO
            )[1]
            img = cv2.bitwise_not(img) 
            img[img == 255] = 0
            img_finale = cv2.copyMakeBorder(img, 26, 26, 26, 26, cv2.BORDER_CONSTANT, 0)
            numbers[label].append(img_finale)
    
    circles = []
    
    
    
    for i in range(1,16):
        circles.append(
            imutils.resize(cv2.threshold(
                 cv2.imread('Circles/'+ str(i) +'.png',cv2.IMREAD_GRAYSCALE),
                135,255,cv2.THRESH_TOZERO_INV
            )[1],width=80)       
        )
        
    backgrounds = []
        
    for i in range(1,4):
        backgrounds.append(
            cv2.resize(
                cv2.imread('Backgrounds/'+ str(i) +'.jpg',cv2.IMREAD_GRAYSCALE), 
                (1960,1960)
            )       
        )
        
    puzzles = []
    
    
    for p in Path('./100/' + dataset_type).glob('*.has'):
        puzzles.append(p.read_text().splitlines()[1::])
    
    i = 1
    
    vw_pad = np.full((80, 40, 3), 255, dtype = "uint8")
    hw_pad = np.full((40, 1960, 3), 255, dtype = "uint8")
    vb_pad = np.zeros((80, 40, 1), dtype = "uint8")
    hb_pad = np.zeros((40, 1960, 1), dtype = "uint8")
    
    for puzzle in puzzles :
        completed_grid_HD = [hb_pad]
        completed_grid_clean = [hw_pad]
        textual = []
        for line in puzzle:
            textual_line = []
            split_line = line.split()
            completed_line_HD = [vb_pad]
            completed_line_clean = [vw_pad]
            for digit in split_line:
                textual_line.append(int(digit))
                img_clean = np.zeros([80,80,3],dtype=np.uint8)
                img_clean.fill(255)
                if digit != '0':
                    img_HD = numbers[int(digit)][randrange(len(numbers[int(digit)]))] + circles[randrange(len(circles))]
                    
                    cv2.circle(img_clean, (40,40), 35, 0, 1)

                    cv2.putText(img_clean,digit, 
                        (30,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,
                        (0,0,0),
                        2,
                        2)
                else :
                    img_HD = np.zeros((80, 80, 1), dtype = "uint8")
                completed_line_HD.append(img_HD)
                completed_line_clean.append(img_clean)
                completed_line_HD.append(vb_pad)
                completed_line_clean.append(vw_pad)
            textual.append(textual_line)
            completed_grid_HD.append(cv2.hconcat(completed_line_HD))
            completed_grid_clean.append(cv2.hconcat(completed_line_clean))
            completed_grid_HD.append(hb_pad)
            completed_grid_clean.append(hw_pad)
            
        cv2.imwrite("Dataset/" + dataset_type + "/Clean/" + str(i) + ".jpg", cv2.vconcat(completed_grid_clean) )    
        
        final_image_HD = cv2.vconcat(completed_grid_HD) 
        
        final_image_HD[final_image_HD == 0] = backgrounds[randrange(len(backgrounds))][np.where(final_image_HD == 0)]
        
        cv2.imwrite("Dataset/" + dataset_type + "/HandDrawn/" + str(i) + ".jpg", final_image_HD)
        
        textual = np.array(textual) 
        textual = textual[~np.all(textual == 0, axis=1)]
        textual = textual[:, ~np.all(textual == 0, axis=0)]

        
        np.savetxt("Dataset/" + dataset_type + "/Textual/" + str(i) + ".txt", textual, fmt='%s')
        
        i+=1

dataset_creator("Train",train_X, train_Y) 
dataset_creator("Test",test_X, test_Y) 
        
    


    

    

