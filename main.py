# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:28:21 2019

@author: sanjith
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#to load an object
from joblib import load
classifier=load('heart.joblib')

choice=1
while choice:
    val=[]
    values=[]
    print('\tPlease enter the values asked below\n\n')
    values.append(int(input('Age : ')))
    values.append(int(input('Sex : ')))
    values.append(int(input('Chest Pain (0-3) : ')))
    values.append(int(input('Resting bp : ')))
    values.append(int(input('Serum Cholestrol : ')))
    values.append(int(input('Blood sugar (0/1) : ')))
    values.append(int(input('Resting ecg (0/1) : ')))
    values.append(int(input('Max heart rate : ')))
    values.append(int(input('Exercise induced Angina(0/1) : ')))
    values.append(float(input('ST depression induced by exercise : ')))
    values.append(int(input('The slope of the peak exercise ST segment(0-2) : ')))
    values.append(int(input('number of major vessels (0-3) colored by flourosopy : ')))
    values.append(int(input('thal(3 = normal; 6 = fixed defect; 7 = reversable defect) : ')))
    val.append(values)
    res=classifier.predict(val)
    if res[0]:
        print('No Heart Disease')
    else:
        print('Heart Diseased')
    choice = int(input('press 0 to stop'))