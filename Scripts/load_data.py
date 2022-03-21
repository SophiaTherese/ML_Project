#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:35:11 2022

@author: Sophia Wesche s173828, Simone Engelbrecht s174276, Aidana Nursultanova s212994
"""

# From exercise 1.5.1
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Load the Glass csv data using the Pandas library
filename = '../Data/glass.data'

# List of attribute names, does not appear in data file and is thus added manually
attributeNames_all = np.array(['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type of glass'])
df = pd.read_csv(filename, names=attributeNames_all)

#print("Missing values:", df.isnull().values.any())

# Pandas returns a dataframe (df), which we will convert to numpy arrays for this project.
raw_data = df.values  

# Notice that raw_data both contains the information we want to store in an array
# X (RI and chemical composition data) and the information that we wish to store 
# in y (the class labels, that is the glass type).

# We start by making the data matrix X by indexing into data.
# We know that the attributes (excluding ID) are stored in column 1-10 from inspecting 
# the file.
cols = range(1, 10) 
X = raw_data[:, cols]
attributeNames = attributeNames_all.tolist()[1:-1]
# A closer look at the attributes:
#print(df[attributeNames].describe().transpose())

# Store the class indices present in the dataset, manually add labels
y = raw_data[:,-1].astype(int) # -1 takes the last column


classNames = np.unique(y)
classLabels = ['building_windows_float_processed',
               'building_windows_non_float_processed',
               'vehicle_windows_float_processed',
               'containers',
               'tableware',
               'headlamps']



# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)






