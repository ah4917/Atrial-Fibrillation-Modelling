# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:16:31 2021

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
#%%timeseries testing
def generate_macro_vectors_weightedGt(data,threshold_c,threshold_t):
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
    
    peak_data=[]
    for ts in time_series:
        peak_data_i=[]
        top=max(ts)
        for t in range(len(ts)):
            if t==0:
                if ts[t]>ts[t+1] and ts[t]>threshold_c*top:
                    peak_data_i.append(t)
            elif t==len(ts)-1:
                if ts[t]>ts[t-1] and ts[t]>threshold_c*top:
                    peak_data_i.append(t)
            else:
                if ts[t]>ts[t-1] and ts[t]>ts[t+1] and ts[t]>0.7*top:
                    peak_data_i.append(t)
        peak_data.append(peak_data_i)

    def ctol(x,y,x_dim):
        return y*x_dim+x
    
    def generate_vector(peak_data,time,i,j,threshold_t):
        #do NOT confuse i and j
        #i corresponds to y values and j to x values
        neighbours = [-1]*8
        centralcell=peak_data[ctol(j,i,x_dim)]
        
        if j == x_dim -1:
            neighbours[1]=0
            neighbours[2]=0
            neighbours[3]=0
        else:            
            neighbours[2]=peak_data[ctol(j+1,i,x_dim)]
        if j==0:
            neighbours[5]=0
            neighbours[6]=0
            neighbours[7]=0
        else:
            neighbours[6]=peak_data[ctol(j-1,i,x_dim)]
        if i == 0:
            if neighbours[0]==-1:
                neighbours[0]=peak_data[ctol(j,y_dim-1,x_dim)]
            if neighbours[1]==-1:
                neighbours[1]=peak_data[ctol(j+1,y_dim-1,x_dim)]
            if neighbours[7]==-1:
                neighbours[7]=peak_data[ctol(j-1,y_dim-1,x_dim)]
        else:
            if neighbours[0]==-1:
                neighbours[0]=peak_data[ctol(j,i-1,x_dim)]
            if neighbours[1]==-1:
                neighbours[1]=peak_data[ctol(j+1,i-1,x_dim)]
            if neighbours[7]==-1:
                neighbours[7]=peak_data[ctol(j-1,i-1,x_dim)]
        if i == y_dim-1:
            if neighbours[3]==-1:
                neighbours[3]=peak_data[ctol(j+1,0,x_dim)]
            if neighbours[4]==-1:
                neighbours[4]=peak_data[ctol(j,0,x_dim)]
            if neighbours[5]==-1:
                neighbours[5]=peak_data[ctol(j-1,0,x_dim)]
        else:
            if neighbours[3]==-1:
                neighbours[3]=peak_data[ctol(j+1,i+1,x_dim)]
            if neighbours[4]==-1:
                neighbours[4]=peak_data[ctol(j,i+1,x_dim)]                
            if neighbours[5]==-1:
                neighbours[5]=peak_data[ctol(j-1,i+1,x_dim)]
        #if neighbour value is -5 then the point corresponds to a boundary
        #if neighbour value is -2 then the point corresponds to no peak
        #if neighbour value is +ve then the point corresponds to a peak
        #need to 'clean' neighbour data and choose min peak_t +ve value 
        
        if time in centralcell:
            peak_cc=time
            for n in range(len(neighbours)):
                if type(neighbours[n])==list:
                    for ni in neighbours[n]:
                        if ni-peak_cc>0 and ni-peak_cc<threshold_t:
                            neighbours[n]=ni-peak_cc
                if type(neighbours[n])==list:
                    neighbours[n]=0
            neighbourstt=[max(neighbours)+1 if s==0 else s for s in neighbours]
            choose=min(neighbourstt)
            neighbours=[1 if i==choose else 0 for i in neighbours]
            total_charge=sum(neighbours)
        else:
            total_charge=0        
        
        if total_charge != 0:
            vector = [0,0]
            res=1/np.sqrt(2)#to resolve diagonals
            vector[0] += res*neighbours[1]/total_charge
            vector[0] += neighbours[2]/total_charge
            vector[0] += res*neighbours[3]/total_charge
            vector[0] += -res*neighbours[5]/total_charge
            vector[0] += -neighbours[6]/total_charge
            vector[0] += -res*neighbours[7]/total_charge
            vector[1] += -res*neighbours[7]/total_charge
            vector[1] += -neighbours[0]/total_charge
            vector[1] += -res*neighbours[1]/total_charge
            vector[1] += res*neighbours[3]/total_charge
            vector[1] += neighbours[4]/total_charge
            vector[1] += res*neighbours[5]/total_charge
            vector = normalise_vector(vector)
        else:
            vector = [0,0]
    
        return vector
                    
    vector_store = []
    for time in range(time_length):
        store_single = []
        for i in range(y_dim):
            for j in range(x_dim):
                store_single.append(generate_vector(peak_data,time,i,j,threshold_t))
        vector_store.append(store_single)
    
    return vector_store
#%%
smltn2402=RunState(20,4000,40,40,5,1,0.32,2,20)
vectors2402=Resultant_Vectors(smltn2402,outv=True)
vc=condvelavg(smltn2402)
#%%
MovieNodes(smltn2402,None)
#%%
xdim=20
res=20
b=xdim/res
pxv2402=PixelatedVectors(smltn2402,vectors2402,res,res)
px2402=Pixelation(smltn2402,res,res)
pxm2402M=generate_macro_vectors_weightedGt(px2402,0,5)
#%%
tM=miss_fire2(smltn2402,20,5,5)[1]
print(tM)
#%%
MoviePixels(px2402,26)
#%%
tc=None
#VectorMovieNodes(vectors2402,smltn2402[1],1)
VectorMovie(pxv2402[0],pxv2402[1],tc)
VectorMovie(pxm2402M,pxv2402[1],tc)
#%%
c=0
time=np.arange(0,len(smltn2402[0]))
#plt.plot(time,px2402[2][1])
plt.plot(time,px2402[2][c])
ll=[]
for t in range(1,len(time)-1):
    top=max(px2402[2][c])
    #print(top)
    if px2402[2][c][t]>px2402[2][c][t-1] and px2402[2][c][t]>px2402[2][c][t+1] and px2402[2][c][t]>0.7*top:
        ll.append(t) 
#%%testing the section of code for the surrounding cells
time=2
centralcell=[2]
neighbours=[[1],[2,4],[4,10],[7],0]
threshold_t=9
j=0
i=0
if time in centralcell:
    peak_cc=time
    for n in range(len(neighbours)):
        if type(neighbours[n])==list:
            for ni in neighbours[n]:
                if ni-peak_cc>0 and ni-peak_cc<threshold_t:
                    neighbours[n]=ni-peak_cc
        if type(neighbours[n])==list:
            neighbours[n]=0
    print(neighbours)
    neighbourstt=[max(neighbours)+1 if s==0 else s for s in neighbours]
    choose=min(neighbourstt)
    neighbours=[1 if i==choose else 0 for i in neighbours]
    print(neighbours)
    total_charge=1
#%%
sa=dot_product_averageA(smltn2402,30,30,0,animate=True)
sd=dot_product_averageA(smltn2402,30,30,macro=True,threshold_c=0.7,threshold_t=2,animate=True)
#%%
sa=dot_product_averageA(smltn2402,30,30,0,display=True)
sd=dot_product_averageA(smltn2402,30,30,macro=True,threshold_c=0.7,threshold_t=2,display=True)
#%%
def Pixelation_noise(cc,x_grid_size,y_grid_size,sigma):
    #prepares pixelation of nodes based on resolution requested
    #cc is output of RunState
    x_size=cc[8][2]
    y_size=cc[8][3]
    tau=cc[8][4]
    grid_coor = []
    for j in range(int(y_grid_size)):
        for i in range(int(x_grid_size)):
            grid_coor.append([i,j])
    grid_container = []
    timeseries=[]#contains time-series for each cell
    for i in range(len(grid_coor)):
        grid_container.append([])
        timeseries.append([])
    for i in range(len(cc[1])):
        grid_coor_state = cc[1][i][0]//(x_size/x_grid_size),cc[1][i][1]//(y_size/y_grid_size) 
        grid_container[int(grid_coor_state[1]*(x_grid_size) + grid_coor_state[0] )].append(i)
    allgridvalues=[]
    allgridvalues_noise=[]
    for i in range(len(cc[0])):
        grid_sum = np.zeros([int(y_grid_size),int(x_grid_size)])
        grid_sum_noise = np.zeros([int(y_grid_size),int(x_grid_size)])
        for cell in range(len(grid_container)):
            sum_c = 0
            for node in range(len(grid_container[cell])):
                sum_c += cc[0][i][grid_container[cell][node]]
            grid_sum[grid_coor[cell][1]][grid_coor[cell][0]]=sum_c
            grid_sum_noise[grid_coor[cell][1]][grid_coor[cell][0]]= random.gauss(sum_c,sigma)
            timeseries[cell].append(sum_c)
        allgridvalues.append(grid_sum)
        allgridvalues_noise.append(grid_sum_noise)
    nodespc=[]#nodespercell(determining cell with max number of nodes)
    for i in range(len(grid_container)):
        nodespc.append(len(grid_container[i]))
    maxcellcolor=np.mean(nodespc)*(tau+1)#determining max value possible 
    #in grid_sum,required to set the color scale    
    return allgridvalues_noise,int(maxcellcolor) ,timeseries,grid_container,grid_coor
#%%
smltn2802=RunState(70,4000,20,20,5,1,0.3,1.2,22)
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
pxn2802=Pixelation_noise(smltn2802,res,res,25)
pxm2802M=generate_macro_vectors_weightedGt(px2802,0.8,2)
pxm2802Mn=generate_macro_vectors_weightedGt(pxn2802,0.8,2)
#%%
MoviePixels(pxn2802,None)
#MoviePixels(px2802,None)
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
tc=None
#VectorMovieNodes(vectors2402,smltn2402[1],1)
#VectorMovie(pxv2802[0],pxv2802[1],tc)
VectorMovie(pxm2802M,pxv2802[1],tc)
VectorMovie(pxm2802Mn,pxv2802[1],tc)
#%%
sa=dot_product_averageA(smltn2802,20,20,0,display=True)
sd=dot_product_averageA(smltn2802,20,20,macro=True,threshold_c=0.8,threshold_t=2,display=True)
sd=dot_product_averageAn(smltn2802,20,20,25,macro=True,threshold_c=0.8,threshold_t=2,display=True)