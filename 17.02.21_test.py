# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:16:51 2021

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
__import__=('13.02.21_code')
#%%
def generate_macro_vectors_weighted(data,threshold):
    #data output of pixelation
    charge_time = data[0]
    time_length = len(charge_time)
    x_dim = len(charge_time[0])
    y_dim = len(charge_time[0][0])
    print(time_length,x_dim,y_dim)
    
    def normalise_vector(vector):
        if vector[0]==0 and vector[1]==0:
            return vector
        else:
            return vector/np.sqrt(vector[0]**2 + vector[1]**2)
    
    def generate_vector(charge_time,time,threshold,i,j):
        neighbours = []
        
        if i == x_dim -1:
            neighbours.append(0)
        else:
            neighbours.append(charge_time[time][i+1][j])
        
        if i==0:
            neighbours.append(0)
        else:
            neighbours.append(charge_time[time][i-1][j])
        
        if j == y_dim-1:
            neighbours.append(charge_time[time][i][0])
        else:
            neighbours.append(charge_time[time][i][j+1])
        
        if j == 0:
            neighbours.append(charge_time[time][i][y_dim-1])
        else:
            neighbours.append(charge_time[time][i][j-1])
        
        
        total_charge = sum(neighbours)
        
        if total_charge != 0:
            vector = [0,0]
            vector[0] += neighbours[0]/total_charge
            vector[0] += -1*neighbours[1]/total_charge
            vector[1] += neighbours[2]/total_charge
            vector[1] += -1*neighbours[3]/total_charge
        
            vector = normalise_vector(vector)
         
        else:
            vector = [0,0]
        
        if total_charge < threshold:
            vector = [0,0]
        return vector
            
    vector_store = []
    for time in range(time_length):
        store_single = []
        for i in range(x_dim):
            for j in range(y_dim):
                store_single.append(generate_vector(charge_time,time,threshold,i,j))
        vector_store.append(store_single)
    
    return vector_store
#%%
smltn1702=RunState(200,5000,100,100,5,1,0.24,3,20)
vectors1702=Resultant_Vectors(smltn1702,outv=True)
pxv1702=PixelatedVectors(smltn1702,vectors1702,50,50)
#%%
#writervideo=animation.FFMpegFileWriter()
writergif=animation.PillowWriter()
MovieNodes(smltn1702,None).save('nodes.gif',writer=writergif)
#%%
px1702=Pixelation(smltn1702,50,50)
#%%
pxm1702=generate_macro_vectors_weighted(px1702,28)
#%%
plt.plot(np.arange(0,200),px1702[2][20])
#%%
#VectorMovie(pxv1702[0],pxv1702[1],None).save('vectorsmicro.gif',writer=writergif)
VectorMovie(pxm1702,pxv1702[1],None).save('vectorsmacro.gif',writer=writergif)
#%%
xdim=50
tau=5
vcond=2
tM=24
#%%
connections=[]
for t in range(tau+1):
    conn_t=ConnectionArray(xdim**2,xdim,xdim,pxv1702[1],t*vcond)
    connections.append(conn_t)
#%%following is for the microscopic
q_allon=[np.array([0.0]*xdim**2) for i in range(len(smltn1702[0])-5)]
q_alltot=np.array([0.0]*xdim**2)
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn1702[0])-5):
        print(t)
        q_all=focal_quality_indicator3(pxv1702[0],pxv1702[1],connections,t,2,5)
        q_all=[0 if i<0 else i for i in q_all]    
        q_alltot+=np.array(q_all)/(len(smltn1702[0])-5-tM)        
        if t<10:
            for ti in range(0,t+1):
                q_allon[ti]+=np.array(q_all)/10
        else:    
            for ti in range(t-9,t+1):
                q_allon[ti]+=np.array(q_all)/10
#%%
q_allon=q_allon[tM:-9]
#%%
matrixtot=np.zeros([xdim,xdim])
for i in range(len(q_alltot)):
    y=i//xdim
    x=i%xdim
    matrixtot[y][x]=q_alltot[i]
matrixl=[]
x_tot=[]
y_tot=[]
for f in range(len(q_allon)):
    matrixli=np.zeros([xdim,xdim])
    for i in range(len(q_allon[f])):
        y=i//xdim
        x=i%xdim
        matrixli[y][x]=q_allon[f][i]
    matrixl.append(matrixli)
    coordinates = peak_local_max(matrixli,min_distance = 10,threshold_abs=0.4, exclude_border = False)
    xloci=[i[1] for i in coordinates]
    yloci=[i[0] for i in coordinates]
    x_tot.append(xloci)
    y_tot.append(yloci)
maxvalue=max([np.max(i) for i in matrixl])
minvalue=min([np.min(i) for i in matrixl])
fig = plt.figure()
ims=[] 
ims_s=[]   
for i in range(len(matrixl)):
    im=plt.imshow(matrixl[i],interpolation='none',cmap='jet',vmin=minvalue,vmax=maxvalue,animated=True)
    ims.append([im])
    ims_si=plt.scatter(x_tot[i],y_tot[i],c='k', marker = 'o',s=50)
    ims_s.append([ims_si])
plt.title('Pixelated Grid')
fig.colorbar(im)
plt.gca().invert_yaxis()
ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000) 
ani2 = animation.ArtistAnimation(fig, ims_s, interval=200, repeat_delay=1000) 
#%%
coordinates = peak_local_max(matrixtot, min_distance = 10,threshold_abs=0, exclude_border = False)
xloci=[i[1] for i in coordinates]
yloci=[i[0] for i in coordinates]
#%%
fig=plt.figure()
s=plt.imshow(matrixtot,interpolation='none',cmap='jet',animated=True)
plt.scatter(xloci,yloci,c='k',marker='o')
fig.colorbar(s)
plt.gca().invert_yaxis()
#plt.savefig('totmicro',dpi=1000)
#%%following is for the macroscopic
q_allonM=[np.array([0.0]*xdim**2) for i in range(len(smltn1702[0])-5)]
q_alltotM=np.array([0.0]*xdim**2)
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn1702[0])-5):
        print(t)
        q_all=focal_quality_indicator3(pxm1702,pxv1702[1],connections,t,2,5)
        q_all=[0 if i<0 else i for i in q_all]    
        q_alltotM+=np.array(q_all)/(len(smltn1702[0])-5-tM)        
        if t<10:
            for ti in range(0,t+1):
                q_allonM[ti]+=np.array(q_all)/10
        else:    
            for ti in range(t-9,t+1):
                q_allonM[ti]+=np.array(q_all)/10
#%%
q_allonM=q_allonM[tM:-9]
#%%
matrixtotM=np.zeros([xdim,xdim])
for i in range(len(q_alltotM)):
    y=i//xdim
    x=i%xdim
    matrixtotM[y][x]=q_alltotM[i]
matrixlM=[]
x_totM=[]
y_totM=[]
for f in range(len(q_allonM)):
    matrixli=np.zeros([xdim,xdim])
    for i in range(len(q_allonM[f])):
        y=i//xdim
        x=i%xdim
        matrixli[y][x]=q_allonM[f][i]
    matrixlM.append(matrixli)
    coordinates = peak_local_max(matrixli,min_distance = 10,threshold_abs=0.2, exclude_border = False)
    xloci=[i[1] for i in coordinates]
    yloci=[i[0] for i in coordinates]
    x_totM.append(xloci)
    y_totM.append(yloci)
maxvalueM=max([np.max(i) for i in matrixlM])
minvalueM=min([np.min(i) for i in matrixlM])
fig = plt.figure()
ims=[] 
ims_s=[]   
for i in range(len(matrixlM)):
    im=plt.imshow(matrixlM[i],interpolation='none',cmap='jet',vmin=minvalueM,vmax=maxvalueM,animated=True)
    ims.append([im])
    ims_si=plt.scatter(x_totM[i],y_totM[i],c='k', marker = 'o',s=50)
    ims_s.append([ims_si])
plt.title('Pixelated Grid')
fig.colorbar(im)
plt.gca().invert_yaxis()
aniM = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000) 
ani2M = animation.ArtistAnimation(fig, ims_s, interval=200, repeat_delay=1000) 
#%%
coordinates = peak_local_max(matrixtotM, min_distance = 10,threshold_abs=0, exclude_border = False)
xlociM=[i[1] for i in coordinates]
ylociM=[i[0] for i in coordinates]
#%%
fig=plt.figure()
s=plt.imshow(matrixtotM,interpolation='none',cmap='jet',animated=True)
plt.scatter(xlociM,ylociM,c='k',marker='o')
fig.colorbar(s)
plt.gca().invert_yaxis()
plt.savefig('totmacro',dpi=1000)
