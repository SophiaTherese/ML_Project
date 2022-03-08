#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:33:04 2022

@author: Sophia Wesche s173828, Simone Engelbrecht s174276, Aidana Nursultanova s212994
"""

from load_data import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 
df = pd.DataFrame(X)
corr = df.corr()

# Defining range:
r = np.arange(M)

# Fill diagonal and upper half with NaNs
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr[mask] = np.nan

f = plt.figure(figsize=(19, 15))
plt.matshow(corr, fignum=f.number, cmap='PiYG', vmin=-1, vmax=1)
for (y, x), value in np.ndenumerate(corr):
    if not np.isnan(value):
        plt.text(x, y, f"{value:.2f}", va="center", ha="center", fontsize=20)
plt.xticks(r, attributeNames, fontsize=20, rotation=45)
plt.yticks(r, attributeNames,fontsize=20, rotation=45)
cb = plt.colorbar(format = '%.1f', ticks=np.arange(-1,1.1,0.2))
cb.set_label(label='Correlation',size=20)
cb.ax.tick_params(labelsize=20)
plt.title('Correlation Matrix', fontsize=24);
plt.show()
