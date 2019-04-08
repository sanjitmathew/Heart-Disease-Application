# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:28:24 2019

@author: sanjith
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:, 13].values

#Splitting datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Classification

from sklearn.svm import SVC
classifier = SVC(C=7,kernel= 'linear',random_state=0)   #kernel='rbf' for kernelsvm
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Valuation

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
std=accuracy.std()
accuracy.mean()

from sklearn.model_selection import GridSearchCV
parameters=[
        { 'C' : [7,8,9,10,11,12]  },
        ]
grid_search = GridSearchCV(estimator=classifier,
                           scoring='accuracy',
                           param_grid = parameters,
                           cv=10,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_params = grid_search.best_params_
