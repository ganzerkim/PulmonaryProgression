# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:40:31 2020

@author: MG
"""

import pydicom # for reading dicom files
import os # for doing directory operations 
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import matplotlib.pyplot as plt

data_dir = 'C:/Users/MG/Desktop/OSIC_pre/train/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('C:/Users/MG/Desktop/OSIC_pre/train.csv', index_col=0)
labels_df.head()

for patient in patients[:3]:
    label = labels_df.get_value(patient, 'Weeks')
    path = data_dir + patient
    
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
       
    print(len(slices),label)
    print(slices[0])
    print(slices[0].pixel_array.shape, len(slices))
    plt.imshow(slices[0].pixel_array)

import cv2
import numpy as np

IMG_PX_SIZE = 150

for patient in patients[:1]:
    label = labels_df.get_value(patient, 'Weeks')
    path = data_dir + patient
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    fig = plt.figure()
    for num,each_slice in enumerate(slices[:12]):
        y = fig.add_subplot(3,4,num+1)
        new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))
        y.imshow(new_img)
    plt.show()
    
    
    