# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:55:04 2021

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
smltn2802=RunState(100,6000,20,20,5,1,0.25,1.2,22)
vectors2802=Resultant_Vectors(smltn2802,outv=True)
vc=condvelavg(smltn2802)
print(vc)
#%%
tM=miss_fire2(smltn2802,20,5,5)[1]
print(tM)
#%%
MovieNodes(smltn2802,None)
#%%
xdim=20
res=10
b=xdim/res
pxv2802=PixelatedVectors(smltn2802,vectors2802,res,res)
px2802=Pixelation(smltn2802,res,res)
pxn2802=Pixelation_noise(smltn2802,res,res,25)
pxm2802M=generate_macro_vectors_weightedGt(px2802,0.8,2)
pxm2802Mn=generate_macro_vectors_weightedGt(pxn2802,0.8,2)
#%%
MoviePixels(px2802,None)
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
def dot_product_averageAF(runstate,x_dim,y_dim,macro = False,threshold_c=0,threshold_t=0,animate=False,display=False,data=False):
    x = runstate
    tau=x[8][4]
    period=x[8][8]
    x_dim_m=x[8][2]
    b=x_dim_m/x_dim
    vectors = Resultant_Vectors(x,outv=True)
    if macro == False:
        vectors_pixelated = PixelatedVectors(x,vectors,x_dim,y_dim)
    else:
        pixels = Pixelation(x,x_dim,y_dim)
        vectors_pixelated = [generate_macro_vectors_weightedGt(pixels,threshold_c,threshold_t),pixels[4]]

    tM =miss_fire2(x,period,tau,tau)[1]
    vcond = 1.2
    connections=[]
    for t in range(tau+1):
        conn_t=ConnectionArray(x_dim**2,x_dim,x_dim,vectors_pixelated[1],t*vcond)
        connections.append(conn_t)

    q_allon=[np.array([0.0]*x_dim**2) for i in range(len(x[0])-5)]
    q_alltot=np.array([0.0]*x_dim**2)

    for t in range(tM,len(x[0])-5):
        #print(t)
        q_all=focal_quality_indicator3(vectors_pixelated[0],vectors_pixelated[1],connections,t,2,5)
        q_alltot+=np.array(q_all)/(len(x[0])-5-tM)        
        if t<10:
            for ti in range(0,t+1):
                q_allon[ti]+=np.array(q_all)/10
        else:    
            for ti in range(t-9,t+1):
                q_allon[ti]+=np.array(q_all)/10

    q_allon=q_allon[tM:-9]

    matrixtot=np.zeros([x_dim,x_dim])    
    for i in range(len(q_alltot)):
        y=i//x_dim
        x=i%x_dim
        matrixtot[y][x]=q_alltot[i]
    maxvalueNt=max([np.max(i) for i in matrixtot])
    minvalueNt=min([np.min(i) for i in matrixtot])
        
    coordinates = peak_local_max(matrixtot, min_distance = 10,threshold_abs=0.99*maxvalueNt, exclude_border = False)
    xloct=[i[1] for i in coordinates]
    yloct=[i[0] for i in coordinates]
    
    matrixl=[]
    for f in range(len(q_allon)):
        matrixli=np.zeros([x_dim,x_dim])
        for i in range(len(q_allon[f])):
            y=i//x_dim
            x=i%x_dim
            matrixli[y][x]=q_allon[f][i]  
        matrixl.append(matrixli)
    maxvalueN=max([np.max(i) for i in matrixl])
    minvalueN=min([np.min(i) for i in matrixl])
    
    x_tot=[]
    y_tot=[]
    for s in range(len(matrixl)):
        coordinates = peak_local_max(matrixl[s],min_distance = 10,threshold_abs=0.7*maxvalueN, exclude_border = False)
        xloci=[i[1] for i in coordinates]
        yloci=[i[0] for i in coordinates]
        x_tot.append(xloci)
        y_tot.append(yloci) 
    
    rolling=[matrixl,x_tot,y_tot,maxvalueN,minvalueN]
    totalavg=[matrixtot,xloct,yloct]
    
    if animate==True:
        #fig=plt.figure()
        fig,ax=plt.subplots()
        ims=[] 
        ims_s=[]   
        for i in range(len(matrixl)):
            im=plt.imshow(matrixl[i],interpolation='none',cmap='jet',vmin=minvalueN,vmax=maxvalueN,animated=True)
            ims.append([im])
            ims_si=plt.scatter(x_tot[i],y_tot[i],c='w', marker = 'x',s=50)
            ims_s.append([ims_si])
        
        fig.colorbar(im)
        xaxisp=np.arange(0,x_dim,2)
        yaxisp=np.arange(0,y_dim,2)
        ax.set_xticks(xaxisp)
        ax.set_yticks(yaxisp)
        ax.set_xticklabels([int(i*b) for i in xaxisp])
        ax.set_yticklabels([int(i*b) for i in yaxisp]) 
        plt.xlabel('Rescaled Distance in the x Direction')
        plt.ylabel('Rescaled Distance in the y Direction')
        plt.gca().invert_yaxis()
        
        aniM = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000) 
        ani2M = animation.ArtistAnimation(fig, ims_s, interval=200, repeat_delay=1000)
        return aniM,ani2M,rolling,totalavg
    
    if display==True:
        fig,ax=plt.subplots()
        plt.rcParams.update({'font.size':15})
        s=plt.imshow(matrixtot,interpolation='none',cmap='jet',vmin=minvalueNt,vmax=maxvalueNt,animated=True)
        plt.scatter(xloct,yloct,c='w',marker='x',s=400)
        fig.colorbar(s)
        plt.gca().invert_yaxis()

        xaxisp=np.arange(0,x_dim,2)
        yaxisp=np.arange(0,y_dim,2)
        ax.set_xticks(xaxisp)
        ax.set_yticks(yaxisp)
        ax.set_xticklabels([int(i*b) for i in xaxisp])
        ax.set_yticklabels([int(i*b) for i in yaxisp]) 
        plt.xlabel('Rescaled Distance in the x Direction')
        plt.ylabel('Rescaled Distance in the y Direction')
        plt.tight_layout()
        return fig,rolling,totalavg
    if data==True:
        if tM==0:
            return 0
        else:
            return np.array([xloct[0],yloct[0]])
#%%
#sa=dot_product_averageA(smltn2802,20,20,0,display=True)
#sa=dot_product_averageA(smltn2802,20,20,0,display=True)
#sd=dot_product_averageAF(smltn2802,20,20,macro=True,threshold_c=0.8,threshold_t=2,display=True)
#plt.savefig('hrn',bbox_inches='tight',dpi=2000)
#sa=dot_product_averageAF(smltn2802,10,10,macro=True,threshold_c=0.8,threshold_t=2,display=True)
#plt.savefig('lrn',bbox_inches='tight',dpi=2000)
sa=dot_product_averageAF(smltn2802,20,20,macro=True,threshold_c=0.8,threshold_t=2,animate=True)
#writergif=animation.PillowWriter()
#sa.save('animationoffocalpoint_poster.gif',writer=writergif)
#%%
x=np.arange(10,110,10)
y=[0,12,44,62,82,71,74,73,72,74]
plt.scatter(x,y,c='k',marker='x')
plt.xlabel('Resolution (%)')
plt.ylabel('Percentage of success (%)')
#plt.savefig('perresnew',bbox_inches='tight',dpi=2000)
#%%
def Vectormovie_plus(vectordata,points,frame,pixeldata):
        #vectordata must be the vector of either nodes or pixels
    #with their respective points
    #if frame==None then it returns the whole movie otherwise specify
    #the frame you need to visualise
    Allgridvalues=pixeldata[0]
    X=[i[0] for i in points]
    Y=[i[1] for i in points]
    U=[0]*len(points)
    V=[0]*len(points)
    
    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X,Y,U, V, pivot='tail', angles='xy', scale_units='xy',scale=0.9, color = 'r')
    
    def update_quiver(num,Q):
        U=[0]*len(points)
        V=[0]*len(points)
        Q.set_UVC(U,V)
        for i in range(len(points)):
            U[i]=vectordata[num][i][0]
            V[i]=vectordata[num][i][1]
            
        Q.set_UVC(U,V)
        plt.imshow(Allgridvalues[num],interpolation='none',cmap=plt.cm.binary,vmin=0,vmax=pixeldata[1]*0.6,animated=True)
        #plt.colorbar()
        #MoviePixels(pixeldata,frame)
        return Q,
    
    if frame==None:
        anim1 = animation.FuncAnimation(fig, update_quiver,frames=len(vectordata),fargs=(Q,), interval=500, blit=True)
        fig.tight_layout()
        return anim1
    else:
        #MoviePixels(pixeldata,frame)
        anim1=update_quiver(frame,Q)
        plt.gca().invert_yaxis()
        x_dim=10
        y_dim=10
        xaxisp=np.arange(0,x_dim,1)
        yaxisp=np.arange(0,y_dim,1)
        ax.set_xticks(xaxisp)
        ax.set_yticks(yaxisp)
        ax.set_xticklabels([int(i*b) for i in xaxisp])
        ax.set_yticklabels([int(i*b) for i in yaxisp])
        
        plt.xlabel("Rescaled Distance in the x Direction",fontsize  =15)
        plt.ylabel("Rescaled Distance in the y Direction",fontsize  =15)
        #plt.savefig('srvector',bbox_inches='tight',dpi = 2000)
#%%
Vectormovie_plus(pxv2802[0],pxv2802[1],8,px2802)