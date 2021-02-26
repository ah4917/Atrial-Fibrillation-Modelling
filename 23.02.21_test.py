# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:54:49 2021

@author: ahadj
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit
import copy
from scipy.optimize import curve_fit
import pandas as pd
from skimage.feature import peak_local_max
__import__=('22.02.21_code')
#%%
smltn2202=RunState(50,5000,50,50,5,1,0.5,3,50)
#%%
MovieNodes(smltn2202,None)
#%%
s=miss_fire2(smltn2202,20,5,5)
print(s[1])
#%%
it=50
epsilon=np.arange(0,0.56,0.02)
#%%
n_dataall=[]
n_dataallstd=[]
for e in epsilon:
    print(e)
    n_data=[]
    for r in range(10):
        n=0
        for i in range(it):
            smltnpd=RunState(40,1250,50,50,5,1,e,3,50)
            timeL=miss_fire2(smltnpd,50,5,5)[1]
            if timeL>0:
                n+=1
        n_data.append(n/it)
    n_dataall.append(np.mean(n_data))
    n_dataallstd.append(np.std(n_data)/np.sqrt(10))

#%%
#np.savetxt('phasediag',n_dataall)
#np.savetxt('phasediagstd',n_dataallstd)
n1=np.loadtxt('phasediag')
n1s=np.loadtxt('phasediagstd')
n1=n1.tolist()
n1s=n1s.tolist()
ntot=n1+n_dataall
ntots=n1s+n_dataallstd
#%%
#np.savetxt('phasediag',ntot)
#np.savetxt('phasediagstd',ntots)
ntot=np.loadtxt('phasediag')
ntots=np.loadtxt('phasediagstd')
#%%
#n_dataallstdH=[1 if i==0 else i for i in n_dataallstd]
plt.xlabel(r'Epsilon, $\epsilon$')
plt.ylabel('Probability of AF to be initated')
#plt.plot(epsilon[:-11],n_dataall,'o')
plt.errorbar(epsilon,ntot,yerr=ntots,fmt='.',capsize=5)
plt.savefig('phasediagram',dpi=500)
#%%


    