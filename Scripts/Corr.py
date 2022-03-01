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


#I følgende benytter jeg: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas

#rs = np.random.RandomState(0)
#df = pd.DataFrame(rs.rand(10, 10))
df = pd.DataFrame(X)
corr = df.corr()

#Defining range:
r = np.arange(1,M+1)

print(corr)

#corr.style.background_gradient(cmap='BrBG_r').set_precision(2)
# 'RdBu_r', 'BrBG_r', & PuOr_r 'coolwarm' are other good diverging colormaps


#Når følgende plot laves kan jeg ikke ændre størrelsen, hvilket gør det meget utydeligt. 
#Desuden legenden ikke fra 0 til 1, men ved ikke hvordan man fixer det.
# Fill diagonal and upper half with NaNs
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr[mask] = np.nan

f = plt.figure(figsize=(19, 15))
plt.matshow(corr, fignum=f.number, cmap='PiYG', vmin=-1, vmax=1)
plt.xticks(r-1, attributeNames, fontsize=20, rotation=45)
plt.yticks(r-1, attributeNames,fontsize=20, rotation=45)
cb = plt.colorbar(format = '%.1f', label='Correlation', ticks=np.arange(-1,1.1,0.2))
cb.ax.tick_params(labelsize=20)
plt.title('Correlation Matrix', fontsize=24);
plt.show()

#Dette plot virker fint
z = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=z.number, cmap='pink')
#plt.xticks(range(df.select_dtypes(['number']).shape[1]), attributeNames.object, fontsize=16, rotation=45)
plt.xticks(r-1, attributeNames, fontsize=20, rotation=45)
#plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
plt.yticks(r-1, attributeNames,fontsize=20, rotation=45)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.title('Correlation Matrix', fontsize=24);
plt.show()



