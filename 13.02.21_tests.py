# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:32:39 2021

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
__import__=('18.01.21_code')
#%%over next few lines will attempt to make the dot product method faster
smltn1302=RunState(80,1000,32,32,5,1,0.35,3,20)
vectors1302=Resultant_Vectors(smltn1302,outv=True)
pxv1302=PixelatedVectors(smltn1302,vectors1302,32,32)
#%%
VectorMovie(pxv1302[0],pxv1302[1],None)
#%%
xdim=32
tM=TimeLoc(pxv1302[0],pxv1302[1],smltn1302)
print(tM)
#%%visualing top dot product values
locall=[]
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn1302[0])-5):
        print(t)
        q_all=[]
        for i in range(32**2):
            q=focal_quality_indicator(pxv1302[0],pxv1302[1],pxv1302[1][i],t,2,5)
            q_all.append(q)
        locall.append(max(q_all))
#%%
time=np.arange(tM,len(smltn1302[0])-5)
plt.plot(time,locall)
#%%visualising the field to compare with focal_quality_indicator3
focal_quality_indicator(pxv1302[0],pxv1302[1],pxv1302[1][495],30,2,5)
#%%
ss=focal_quality_indicator3(pxv1302[0],pxv1302[1],30,2,5)
#%%obtaining data for all dot product values
locall=[0]*xdim**2
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn1302[0])-5):
        print(t)
        q_all=[]
        for i in range(32**2):
            q=focal_quality_indicator(pxv1302[0],pxv1302[1],pxv1302[1][i],t,2,5)
            locall[i]+=q
#%%
matrix=np.zeros([32,32])
for i in range(len(locall)):
    y=i//xdim
    x=i%xdim
    matrix[y][x]=locall[i]
fig=plt.figure()
s=plt.imshow(matrix,interpolation='none',cmap='jet',animated=True)
plt.gca().invert_yaxis()
fig.colorbar(s)
#%%testing new function with connections
conn=ConnectionArray(xdim**2,32,32,pxv1302[1],10)
q_allcum=np.array([0.0]*xdim**2)
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn1302[0])-5):
        print(t)
        q_all=focal_quality_indicator2(pxv1302[0],pxv1302[1],conn,t,2,5)
        q_allcum+=np.array(q_all)
#%%
matrix2=np.zeros([32,32])
for i in range(len(q_allcum)):
    y=i//xdim
    x=i%xdim
    matrix2[y][x]=q_allcum[i]
fig=plt.figure()
s=plt.imshow(matrix2,interpolation='none',cmap='jet',animated=True)
plt.gca().invert_yaxis()
fig.colorbar(s)
#%%
coordinates = peak_local_max(matrix2, min_distance = 5, exclude_border = True)

print(coordinates)
x = []
y = []
for i in range(len(coordinates)):
    x.append(coordinates[i][1])
    y.append(coordinates[i][0])
    
plt.imshow(matrix2,interpolation='none',cmap='jet',animated=True)
plt.scatter(x,y, marker = 'x')
plt.gca().invert_yaxis()
plt.show()
#%%now repeating with improved dot product that considers radius in timesteps
q_allcum2=np.array([0.0]*xdim**2)
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn1302[0])-5):
        print(t)
        q_all=focal_quality_indicator3(pxv1302[0],pxv1302[1],t,2,5)
        q_allcum2+=np.array(q_all)
#%%
matrix3=np.zeros([32,32])
for i in range(len(q_allcum)):
    y=i//xdim
    x=i%xdim
    matrix3[y][x]=q_allcum2[i]
fig=plt.figure()
s=plt.imshow(matrix3,interpolation='none',cmap='jet',animated=True)
plt.gca().invert_yaxis()
fig.colorbar(s)
#%%
coordinates = peak_local_max(matrix3, min_distance = 5, exclude_border = False)
print(coordinates)
x = []
y = []
for i in range(len(coordinates)):
    x.append(coordinates[i][1])
    y.append(coordinates[i][0])
    
plt.imshow(matrix3,interpolation='none',cmap='jet',animated=True)
plt.scatter(x,y, marker = 'x')
plt.gca().invert_yaxis()
plt.show()
#%%will need to test this for larger system to investigate if there could be 2 coexisting 
#focal points and for longer to study the session in windows
smltn1602=RunState(200,5000,100,100,5,1,0.24,3,20)
vectors1602=Resultant_Vectors(smltn1602,outv=True)
pxv1602m=PixelatedVectors(smltn1602,vectors1602,100,100)
#%%
xdim=50
pxv1602=PixelatedVectors(smltn1602,vectors1602,xdim,xdim)
#%%
#MovieNodes(smltn1602,None)
VectorMovie(pxv1602[0],pxv1602[1],36)
#%%
xdim=50
tau=5
vcond=2
tM=TimeLoc(pxv1602m[0],pxv1602m[1],smltn1602)
print(tM[0])
plt.plot(np.arange(0,len(smltn1602[0])),tM[1],'x')
#%%
#must prepare the connections first
connections=[]
for t in range(tau+1):
    conn_t=ConnectionArray(xdim**2,xdim,xdim,pxv1602[1],t*vcond)
    connections.append(conn_t)
#%%
tM=36
#q_allon=[np.array([0.0]*xdim**2) for i in range(len(smltn1602[0])//10)]
q_allon=[np.array([0.0]*xdim**2) for i in range(len(smltn1602[0])-5)]
#-15 arises from rolling average of 10 and the tau 5(due to dot product method using timesteps forward in time)
q_alltot=np.array([0.0]*xdim**2)
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn1602[0])-5):
        print(t)
        q_all=focal_quality_indicator3(pxv1602[0],pxv1602[1],connections,t,2,5)
        q_all=[0 if i<0 else i for i in q_all]    
        q_alltot+=np.array(q_all)/(len(smltn1602[0])-5-tM)        
        #l=t//10        
        #q_allon[l]+=np.array(q_all)/10
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
ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000) 
ani2 = animation.ArtistAnimation(fig, ims_s, interval=100, repeat_delay=1000) 
ani3=[ani,ani2]
#writergif=animation.PillowWriter()
#ani3.save('videoofslice.gif',writer=writergif)
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
#plt.savefig('avgwholenozero',dpi=1000)
#%%
density=np.arange(0.2,1.2,0.2)
vc_all=[]
vc_allstd=[]
for d in density:
    print(d)
    nodes=int(d*100**2)
    vc=[]
    for i in range(20):
        smltn17=RunState(50,nodes,100,100,5,1,0.24,3,20)
        
        vc_i=condvelavg(smltn17)
        vc.append(vc_i)
    vc_all.append(np.mean(vc))
    vc_allstd.append(np.std(vc))
#%%
plt.plot(density,5/np.array(vc_all))