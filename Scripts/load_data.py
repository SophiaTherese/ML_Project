#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:35:11 2022

@author: Sophia
"""

# From exercise 1.5.1
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Load the Iris csv data using the Pandas library
filename = '../Data/glass.data'

# List of attribute names, does not appear in data file and is thus added manually
attributeNames = np.array(['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type of glass'])
df = pd.read_csv(filename, names=attributeNames)
print(df[['Ba', 'Type of glass']])
print(df.loc[df['Ba'] > 0])

print("Missing values:", df.isnull().values.any())
print(df[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].describe().transpose())
# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df.values  

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(0, 10) 
X = raw_data[:, cols]

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
    
y = raw_data[:,-1] # -1 takes the last column
# classLabels = ['navn på den første', ..]
# Then determine which classes are in the data by finding the set of 
# unique class labels 
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






