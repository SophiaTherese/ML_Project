#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:33:04 2022

@author: Sophia
"""

from load_data import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




#Jeg ved godt at du siger man ikke skal definere det sådan her, men gør det
#indtil vi får ryttet op. 
cols = range(1, 10) 
X = raw_data[:, cols]
attributeNames = np.array(['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
N, M = X.shape



#I følgende benytter jeg: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas


#rs = np.random.RandomState(0)
#df = pd.DataFrame(rs.rand(10, 10))
df = pd.DataFrame(X)
corr = df.corr()

#Defining range:
r = np.arange(1,M+1)



#corr.style.background_gradient(cmap='BrBG_r').set_precision(2)
# 'RdBu_r', 'BrBG_r', & PuOr_r 'coolwarm' are other good diverging colormaps


#Når følgende plot laves kan jeg ikke ændre størrelsen, hvilket gør det meget utydeligt. 
#Desuden legenden ikke fra 0 til 1, men ved ikke hvordan man fixer det.
# Fill diagonal and upper half with NaNs
f = plt.figure(figsize=(50, 30))
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr[mask] = np.nan
(corr 
 .style
 .background_gradient(cmap='BrBG_r', axis=None, vmin=-1, vmax=1)
 .highlight_null(null_color='#FFFFFF')  # Color NaNs grey
 .set_precision(2))

plt.matshow(corr)
plt.xticks(r-1, attributeNames, fontsize=8, rotation=45)
plt.yticks(r-1, attributeNames,fontsize=8, rotation=45)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=6)
plt.title('Correlation Matrix', fontsize=12);




#Dette plot virker fint
z = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=z.number)
#plt.xticks(range(df.select_dtypes(['number']).shape[1]), attributeNames.object, fontsize=16, rotation=45)
plt.xticks(r-1, attributeNames, fontsize=20, rotation=45)
#plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
plt.yticks(r-1, attributeNames,fontsize=20, rotation=45)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.title('Correlation Matrix', fontsize=24);


