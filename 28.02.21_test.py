# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:48:39 2021

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
__import__=('27.02.21_code')
#%%
smltn2802=RunState(100,3000,20,20,5,1,0.28,1.2,22)
vectors2802=Resultant_Vectors(smltn2802,outv=True)
vc=condvelavg(smltn2802)
print(vc)
#%%
MovieNodes(smltn2802,None)
#%%
tM=miss_fire2(smltn2802,20,5,5)[1]
print(tM)
#%%
xdim=20
res=20
b=xdim/res
pxv2802=PixelatedVectors(smltn2802,vectors2802,res,res)
px2802=Pixelation(smltn2802,res,res)
pxn2802=Pixelation_noise(smltn2802,res,res,35)
pxm2802M=generate_macro_vectors_weightedGt(px2802,0.8,2)
pxm2802Mn=generate_macro_vectors_weightedGt(pxn2802,0.8,2)
#%%
t=3
MoviePixels(pxn2802,t)
MoviePixels(px2802,t)
#%%
c=pxn2802[2][0]
time=np.arange(0,len(smltn2802[0]))
plt.plot(time,px2802[2][0])
plt.plot(time,c)
ll=[]
for t in range(1,len(time)-1):
    top=max(c)
    #print(top)
    if c[t]>c[t-1] and c[t]>c[t+1] and c[t]>0.7*top:
        ll.append(t) 
#%%
tc=4
#VectorMovieNodes(vectors2402,smltn2402[1],1)
VectorMovie(pxv2802[0],pxv2802[1],tc)
VectorMovie(pxm2802M,pxv2802[1],tc)
#VectorMovie(pxm2802Mn,pxv2802[1],tc)
#%%
#sa=dot_product_averageA(smltn2802,20,20,0,display=True)
sa=dot_product_averageA(smltn2802,20,20,0,display=True)
sd=dot_product_averageA(smltn2802,20,20,macro=True,threshold_c=0.8,threshold_t=2,display=True)
sdn=dot_product_averageAn(smltn2802,20,20,50,macro=True,threshold_c=0.8,threshold_t=2,display=True)
#%%
xdim=20
#res=np.arange(1,1.05,0.1)
res=[0.03,0.05,0.08,0.11,0.14,0.16,0.23,0.26,0.33,0.38,0.42,0.48]
c_a=[]
success_e=[]
distance_e=[]
for i in range(5):
    success=[0]*len(res)
    distance_all=[]
    c=0
    for it in range(50):
        print(it)
        smltn2802R=RunState(100,3000,20,20,5,1,0.28,1.2,22)
        #microloc=dot_product_averageA(smltn2802R,20,20,0,display=True)
        microloc=dot_product_averageA(smltn2802R,20,20,0,data=True)
        distance_i=[]
        if type(microloc)==int:
            continue
        else:
            c+=1
            for r in range(len(res)):
                grid_length=int(np.sqrt(res[r]*0.9*3000))
                #print(grid_length)
                b=xdim/grid_length
                pathl=1.2*5
                #pathlr=pathl/b
                #macroloc=dot_product_averageA(smltn2802R,grid_length,grid_length,macro=True,threshold_c=0.8,threshold_t=2,display=True)
                macroloc=dot_product_averageA(smltn2802R,grid_length,grid_length,macro=True,threshold_c=0.8,threshold_t=2,data=True)
                macrolocmic=macroloc*b
                
                distance=np.sqrt((microloc[0]-macrolocmic[0])**2+(microloc[1]-macrolocmic[1])**2)
                distance_i.append(distance)
                if distance<=pathl:
                    success[r]+=1
        distance_all.append(distance_i)
    success_e.append(success)
    distance_e.append(distance_all)
    c_a.append(c)
#%%
#np.savetxt('success0203M',success_e)
#np.save('distance0203M',distance_e)
#distancealls=distanceall[:50]
#%%following data performed at 1360 nodes for 40 iterations
success1=np.loadtxt('success0103')
success2=np.loadtxt('success0203')
success_total=np.concatenate((success1, success2))
res=[0.03,0.05,0.08,0.11,0.16,0.21,0.26,0.33,0.51,0.6,0.73]
#res=[0.03,0.05,0.08,0.11,0.14,0.16,0.23,0.26,0.33,0.38,0.42,0.48]
#success_total=success1
#%%
#success=[0]*len(res)
#for it in range(50):
#    for r in range(len(res)):
#        grid_length=int(np.sqrt(res[r]*xdim**2))
#        b=xdim/grid_length
#        pathl=1.2*5
#        pathlr=pathl/b
#        if distancealls[it][r]<=pathlr:
#            success[r]+=1
#%%
success_eg=[[] for i in range(len(res))]
for i in range(len(success_total)):
    for j in range(len(success_total[i])):
        success_eg[j].append(success_total[i][j])
success_egm=np.array([np.mean(i) for i in success_eg])*(100/40)
success_egstd=np.array([np.std(i) for i in success_eg])/np.sqrt(6)*(100/40)
#%%
#res=[0.03,0.05,0.08,0.11,0.16,0.21,0.26,0.33,0.51,0.6,0.73]
#res=[0.03,0.05,0.08,0.11,0.14,0.16,0.23,0.26,0.33,0.38,0.42,0.48]
plt.errorbar(np.array(res)*100,success_egm,yerr=success_egstd,fmt='k.',ecolor='k',capsize=5)
plt.plot(np.array(res)*100,success_egm,'kx')
plt.ylabel('Percentage of success (%)')
plt.xlabel('Resolution (%)')
plt.xlim(0,100)
plt.ylim(0,100)

#plt.savefig('ressuccnew',bbox_inches='tight',dpi=2000)
#%%following data performed for 3000 nodes at 50 iterations
res2=[0.03,0.05,0.08,0.11,0.14,0.16,0.23,0.26,0.33,0.38,0.42,0.48]
success_total2=np.loadtxt('success0203M')
#%%
success_eg2=[[] for i in range(len(res2))]
for i in range(len(success_total2)):
    for j in range(len(success_total2[i])):
        success_eg2[j].append(success_total2[i][j])
success_egm2=np.array([np.mean(i) for i in success_eg2])*(100/50)
success_egstd2=np.array([np.std(i) for i in success_eg2])/np.sqrt(5)*(100/50)
#%%
plt.errorbar(np.array(res2)*100,success_egm2,yerr=success_egstd2,fmt='k.',ecolor='k',capsize=5)
plt.plot(np.array(res2)*100,success_egm2,'kx')
plt.ylabel('Percentage of success (%)')
plt.xlabel('Resolution (%)')
plt.xlim(0,100)
plt.ylim(0,100)
#%%plotting the distance graph for the 1st distribution
distance1=np.load('distance0103.npy')
distance2=np.load('distance0203.npy')
distance_total=np.concatenate((distance1,distance2))
res=[0.03,0.05,0.08,0.11,0.16,0.21,0.26,0.33,0.51,0.6,0.73]
#%%
distance_ra=[[] for i in range(len(res))]
for r in range(len(distance_total)):
    for i in range(len(distance_total[r])):
        for re in range(len(distance_total[r][i])):
            distance_ra[re].append(distance_total[r][i][re])
distance_egm2=np.array([np.mean(i) for i in distance_ra])/(1.2*5)
distance_egstd2=np.array([np.std(i) for i in distance_ra])/(np.sqrt(len(distance_ra[0]))*1.2*5)
#%%
plt.errorbar(np.array(res)*100,distance_egm2,yerr=distance_egstd2,fmt='k.',ecolor='k',capsize=5)
plt.errorbar(np.array(res[:-3])*100,dist_exp,yerr=dist_expstd,fmt='.',ecolor='b',capsize=5)
plt.plot(np.array(res)*100,distance_egm2,'kx')
plt.ylabel('Rescaled Distance to Pathlength Ratio')
plt.xlabel('Resolution (%)')
#plt.savefig('resgr',dpi=1000)
#%%
grid_size=[int(np.sqrt(i*0.9*1360)) for i in res]
b=[math.ceil(20/i) for i in grid_size[:-3]]
dist_exp=[]
dist_expstd=[]
for bi in b:
    coor=[[i,j] for j in range(bi) for i in range(bi)]
    dist_a=[]
    for c in range(0,len(coor)):
        dist_a.append(np.sqrt((coor[0][0]-coor[c][0])**2+(coor[0][1]-coor[c][1])**2))
    dist_exp.append(np.mean(dist_a))
    dist_expstd.append(np.std(dist_a)/np.sqrt(len(coor)))
    
    
