#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:33:04 2022

@author: Sophia
"""

from load_data import *
from PCA import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



rs = np.random.RandomState(0)
#df = pd.DataFrame(rs.rand(10, 10))
df = pd.DataFrame(X_PCA)


corr = df.corr()


f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
#plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);



#corr.style.background_gradient(cmap='BrBG_r').set_precision(2)
# 'RdBu_r', 'BrBG_r', & PuOr_r 'coolwarm' are other good diverging colormaps



# Fill diagonal and upper half with NaNs
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr[mask] = np.nan
(corr 
 .style
 .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
 .highlight_null(null_color='#f1f1f1')  # Color NaNs grey
 .set_precision(2))

plt.matshow(corr)

plt.show()