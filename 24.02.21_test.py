# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:57:19 2021

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
#%%will be adjusting the macro method so that it takes into consideration the 
#the timeseries of each cell
def generate_macro_vectors_weightedG(data,threshold):
    #data output of pixelation
    charge_time = data[0]
    time_series=data[2]
    time_length = len(charge_time)
    x_dim = len(charge_time[0])
    y_dim = len(charge_time[0][0])
    
    def normalise_vector(vector):
        if vector[0]==0 and vector[1]==0:
            return vector
        else:
            return vector/np.sqrt(vector[0]**2 + vector[1]**2)
    
    def peakloc(time_series,x,y,time,x_dim):
        time_length=len(time_series[0])
        lpos=y*x_dim+x
        ts=time_series[lpos]
        peak_t=-2

        if time>time_length-3:
            peak_t=-2
        else:
            if ts[time+1]>ts[time] and ts[time+1]>ts[time+2]:
                peak_t=time+1
        return peak_t
    
    def generate_vector(charge_time,time_series,time,threshold,i,j):
        #do NOT confuse i and j
        #i corresponds to y values and j to x values
        neighbours = [-1]*8
        centralcell=charge_time[time][i][j]
        if time==0:
            peakc=-2
        else:
            peakc=peakloc(time_series,j,i,time-1,x_dim)
        if j == x_dim -1:
            neighbours[1]=-5
            neighbours[2]=-5
            neighbours[3]=-5
        else:            
            neighbours[2]=peakloc(time_series,j+1,i,time,x_dim)
        if j==0:
            neighbours[5]=-5
            neighbours[6]=-5
            neighbours[7]=-5
        else:
            neighbours[6]=peakloc(time_series,j-1,i,time,x_dim)        
        if i == 0:
            if neighbours[0]==-1:
                neighbours[0]=peakloc(time_series,j,y_dim-1,time,x_dim)  
            if neighbours[1]==-1:
                neighbours[1]=peakloc(time_series,j+1,y_dim-1,time,x_dim)  
            if neighbours[7]==-1:
                neighbours[7]=peakloc(time_series,j-1,y_dim-1,time,x_dim)  
        else:
            if neighbours[0]==-1:
                neighbours[0]=peakloc(time_series,j,i-1,time,x_dim)  
            if neighbours[1]==-1:
                neighbours[1]=peakloc(time_series,j+1,i-1,time,x_dim)  
            if neighbours[7]==-1:
                neighbours[7]=peakloc(time_series,j-1,i-1,time,x_dim)  
        
        if i == y_dim-1:
            if neighbours[3]==-1:
                neighbours[3]=peakloc(time_series,j+1,0,time,x_dim)  
            if neighbours[4]==-1:
                neighbours[4]=peakloc(time_series,j,0,time,x_dim)  
            if neighbours[5]==-1:
                neighbours[5]=peakloc(time_series,j-1,0,time,x_dim)  
        else:
            if neighbours[3]==-1:
                neighbours[3]=peakloc(time_series,j+1,i+1,time,x_dim)  
            if neighbours[4]==-1:
                neighbours[4]=peakloc(time_series,j,i+1,time,x_dim)  
            if neighbours[5]==-1:
                neighbours[5]=peakloc(time_series,j-1,i+1,time,x_dim)  
        #if neighbour value is -5 then the point corresponds to a boundary
        #if neighbour value is -2 then the point corresponds to no peak
        #if neighbour value is +ve then the point corresponds to a peak
        #need to 'clean' neighbour data and choose min peak_t +ve value 
        if j==1 and i==1 and time==0:
            print(neighbours)
        neighboursc=[i for i in neighbours if i>0]#to remove negative values
        if len(neighboursc)==0 or peakc<0:
            total_charge=0
        elif peakc>0:
            peak_tc=min(neighboursc)
            neighboursnew=[1 if i==peak_tc else 0 for i in neighbours]
            neighbours=neighboursnew
            total_charge=sum(neighbours)
        if j==1 and i==1 and time==0:
            print(neighbours)
        if total_charge != 0:
            vector = [0,0]
            res=1/np.sqrt(2)#to resolve diagonals
            vector[0] += res*neighbours[1]/total_charge
            vector[0] += neighbours[2]/total_charge
            vector[0] += res*neighbours[3]/total_charge
            vector[0] += -res*neighbours[5]/total_charge
            vector[0] += -neighbours[6]/total_charge
            vector[0] += -res*neighbours[7]/total_charge
            vector[1] += res*neighbours[7]/total_charge
            vector[1] += neighbours[0]/total_charge
            vector[1] += res*neighbours[1]/total_charge
            vector[1] += -res*neighbours[3]/total_charge
            vector[1] += -neighbours[4]/total_charge
            vector[1] += -res*neighbours[5]/total_charge
            vector = normalise_vector(vector)
        else:
            vector = [0,0]
        
        if total_charge < threshold:
            vector = [0,0]

        return vector
        
            
    vector_store = []
    for time in range(time_length):
        store_single = []
        for i in range(y_dim):
            for j in range(x_dim):
                store_single.append(generate_vector(charge_time,time_series,time,threshold,i,j))
        vector_store.append(store_single)
    
    return vector_store
#%%
smltn2402=RunState(40,2500,50,50,5,1,0.25,4,20)
vectors2402=Resultant_Vectors(smltn2402,outv=True)
vc=condvelavg(smltn2402)
#%%
MovieNodes(smltn2402,None)
#%%
xdim=50
res=50
b=xdim/res
pxv2402=PixelatedVectors(smltn2402,vectors2402,res,res)
px2402=Pixelation(smltn2402,res,res)
pxm2402M=generate_macro_vectors_weightedG(px2402,0)
#%%
MoviePixels(px2402,1)
#%%
time=np.arange(0,len(smltn2402[0]))
plt.plot(time,px2402[2][0])
plt.plot(time,px2402[2][11])
ll=[]
for t in range(1,len(time)-1):
    if px2402[2][70][t]>px2402[2][70][t-1] and px2402[2][70][t]>px2402[2][70][t+1]:
        ll.append(t)
#%%
tc=1
#VectorMovieNodes(vectors2402,smltn2402[1],1)
VectorMovie(pxv2402[0],pxv2402[1],tc)
VectorMovie(pxm2402M,pxv2402[1],tc) 
#%%
st=dot_product_averageA(smltn2402,25,25,macro=True,threshold=0,display=True)
sa=dot_product_averageA(smltn2402,25,25,0,display=True)
#%%james improvements
def generate_macro_vectors_weightedN_AGAIN(data,threshold,zero_threshold):
    #data output of pixelation
    charge_time = data[0]
    time_length = len(charge_time)
    x_dim = len(charge_time[0])
    y_dim = len(charge_time[0][0])
    
    def normalise_vector(vector):
        if vector[0]==0 and vector[1]==0:
            return vector
        else:
            return vector/np.sqrt(vector[0]**2 + vector[1]**2)
    
    def generate_vector(charge_time,time,threshold,i,j):
        
        if charge_time[time][i][j] > threshold:
            neighbours = [-1]*8
            
            if j == x_dim -1:
                neighbours[1]=0
                neighbours[2]=0
                neighbours[3]=0
            else:            
                neighbours[2]=charge_time[time][i][j+1]        
            if j==0:
                neighbours[5]=0
                neighbours[6]=0
                neighbours[7]=0
            else:
                neighbours[6]=charge_time[time][i][j-1]
            
            if i == 0:
                if neighbours[0]<0:
                    neighbours[0]=charge_time[time][y_dim-1][j]
                if neighbours[1]<0:
                    neighbours[1]=charge_time[time][y_dim-1][j+1]
                if neighbours[7]<0:
                    neighbours[7]=charge_time[time][y_dim-1][j-1]
            else:
                if neighbours[0]<0:
                    neighbours[0]=charge_time[time][i-1][j]
                if neighbours[1]<0:
                    neighbours[1]=charge_time[time][i-1][j+1]
                if neighbours[7]<0:
                    neighbours[7]=charge_time[time][i-1][j-1]
            
            if i == y_dim-1:
                if neighbours[3]<0:
                    neighbours[3]=charge_time[time][0][j+1]
                if neighbours[4]<0:
                    neighbours[4]=charge_time[time][0][j]
                if neighbours[5]<0:
                    neighbours[5]=charge_time[time][0][j-1]
            else:
                if neighbours[3]<0:
                    neighbours[3]=charge_time[time][i+1][j+1]
                if neighbours[4]<0:
                    neighbours[4]=charge_time[time][i+1][j]
                if neighbours[5]<0:
                    neighbours[5]=charge_time[time][i+1][j-1]        
            
            total_charge = sum(neighbours)
        
            for i in range(len(neighbours)):
                if neighbours[i] > zero_threshold:
                    neighbours[i] = 0
        else:
            total_charge = 0
        
        if total_charge != 0:
            vector = [0,0]
            res=1/2#to resolve diagonals
            vector[0] += res*neighbours[1]/total_charge
            vector[0] += neighbours[2]/total_charge
            vector[0] += res*neighbours[3]/total_charge
            vector[0] += -res*neighbours[5]/total_charge
            vector[0] += -neighbours[6]/total_charge
            vector[0] += -res*neighbours[7]/total_charge
            
            vector[1] += res*neighbours[7]/total_charge
            vector[1] += neighbours[0]/total_charge
            vector[1] += res*neighbours[1]/total_charge
            vector[1] += -res*neighbours[3]/total_charge
            vector[1] += -neighbours[4]/total_charge
            vector[1] += -res*neighbours[5]/total_charge
            
            vector = normalise_vector(vector)
        else:
            vector = [0,0]
        
        if total_charge < threshold:
            vector = [0,0]
        return vector
            
    vector_store = []
    for time in range(time_length):
        store_single = []
        for i in range(y_dim):
            for j in range(x_dim):
                store_single.append(generate_vector(charge_time,time,threshold,i,j))
        vector_store.append(store_single)
    
    return vector_store  
#%%
x = RunState(100,5000,80,80,6,1,0.35,3,15)






