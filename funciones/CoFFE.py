import scipy.optimize as opt
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dtes
import copy
from scipy.io.idl import readsav
from tqdm import tqdm
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import datetime


def CoFFE(data,p):
    

    nMagnetograms=np.shape(data)[2]
    param1=np.empty([nMagnetograms,5])
    param2=np.empty([nMagnetograms,5])

    for i in tqdm(range(nMagnetograms-1,-1,-1)):

        movi=data[:,:,i]    
        arr=copy.deepcopy(movi[:,:])

        ydata1 = copy.deepcopy(arr)
        ydata1[ydata1<0] = 0

        ydata2 = copy.deepcopy(arr)
        ydata2[ydata2>0] = 0
        
        sz=np.shape(movi)
        
        x = np.linspace(0, sz[1]-1, sz[1])
        y = np.linspace(0, sz[0]-1, sz[0])
        x, y = np.meshgrid(x, y)

        xdata_tuple=(x,y)
        xdata = np.vstack((x.ravel(),y.ravel()))

        noise_sigma1=np.ones(np.shape(arr))
        noise_sigma2=np.ones(np.shape(arr))

        nmax=np.unravel_index(np.argmax(ydata1, axis=None), np.shape(arr))
        nmin=np.unravel_index(np.argmin(ydata2, axis=None), np.shape(arr))

        if i == nMagnetograms-1:
            initial_guess1 = (np.amax(ydata1),nmax[1],nmax[0],10,0)
            initial_guess2 = (np.amin(ydata2),nmin[1],nmin[0],10,0)
        else:
            initial_guess1 = popt1
            initial_guess2 = popt2

        bounds1=((0.5*initial_guess1[0], initial_guess1[1]-10, initial_guess1[2]-10, 0,-0.5*np.pi), 
                 (1.5*initial_guess1[0], initial_guess1[1]+10, initial_guess1[2]+10, 0.5*sz[0],0.5*np.pi))
        bounds2=((1.5*initial_guess2[0], initial_guess2[1]-10, initial_guess2[2]-10, 0,-0.5*np.pi), 
                 (0.5*initial_guess2[0], initial_guess2[1]+10, initial_guess2[2]+10, 0.5*sz[0],0.5*np.pi))


    #    print(initial_guess1)
    #    print(bounds1)
        try:
            popt1, pcov1 = opt.curve_fit(twoD_Gaussian, xdata, ydata1.ravel(),method='trf', p0=initial_guess1,bounds=bounds1, 
                                         sigma=10*noise_sigma1.ravel(), absolute_sigma=True)
            popt2, pcov2 = opt.curve_fit(twoD_Gaussian, xdata, ydata2.ravel(),method='trf', p0=initial_guess2,bounds=bounds2, 
                                         sigma=10*noise_sigma2.ravel(), absolute_sigma=True)

            h=0

            while h < 8:
        #        print(h)

                tilt1=np.arctan((popt1[2]-popt2[2])/(popt1[1]-popt2[1]))

         #       print(h,tilt1*180/np.pi)
                if tilt1 < 0:
                    loc1=y-(-1/np.tan(tilt1))*(x-popt1[1]+p*popt1[3]) - popt1[2]
                    loc2=y-(-1/np.tan(tilt1))*(x-popt2[1]-p*popt2[3]) - popt2[2]
                else:
                    loc1=y-(1/np.tan(tilt1))*(x-popt1[1]+p*popt1[3]) - popt1[2]
                    loc2=y-(1/np.tan(tilt1))*(x-popt2[1]-p*popt2[3]) - popt2[2]

                loc1[loc1 > 0] = 1
                loc1[loc1 < 0] = 9999999

                loc2[loc2 > 0] = 9999999
                loc2[loc2 < 0] = 1

                noise_sigma1 = 10*loc1
                noise_sigma2 = 10*loc2


                popt1, pcov1 = opt.curve_fit(twoD_Gaussian, xdata, ydata1.ravel(),method='trf',p0=popt1,bounds=bounds1, 
                                             sigma=noise_sigma1.ravel(), absolute_sigma=True,maxfev=5000)
                popt2, pcov2 = opt.curve_fit(twoD_Gaussian, xdata, ydata2.ravel(),method='trf',p0=popt2,bounds=bounds2, 
                                             sigma=noise_sigma2.ravel(), absolute_sigma=True,maxfev=5000)                 

                h=h+1

            bounds1=((0.95*popt1[0], popt1[1]-30, popt1[2]-30, 0.9*popt1[3],popt1[4]-0.01), (1.05*popt1[0], popt1[1]+30, popt1[2]+30, 1.1*popt1[3],popt1[4]+0.01))
            bounds2=((1.05*popt2[0], popt2[1]-30, popt2[2]-30, 0.9*popt2[3],popt2[4]-0.01), (0.95*popt2[0], popt2[1]+30, popt2[2]+30, 1.1*popt2[3],popt2[4]+0.01) )

            popt1, pcov1 = opt.curve_fit(twoD_Gaussian, xdata, ydata1.ravel(),method='trf',p0=popt1,bounds=bounds1, 
                                         sigma=10*np.ones(np.shape(arr)).ravel(), absolute_sigma=True,maxfev=5000)
            popt2, pcov2 = opt.curve_fit(twoD_Gaussian, xdata, ydata2.ravel(),method='trf',p0=popt2,bounds=bounds2, 
                                         sigma=10*np.ones(np.shape(arr)).ravel(), absolute_sigma=True,maxfev=5000)


            param1[i-nMagnetograms,:]=popt1
            param2[i-nMagnetograms,:]=popt2
            
        except:
            param1[i-nMagnetograms,:]=initial_guess1
            param2[i-nMagnetograms,:]=initial_guess2
            


    tiltC = np.arctan((param1[:,2]-param2[:,2])/(param1[:,1]-param2[:,1]))*180/np.pi
    tiltM=[]
    for i in range(0,nMagnetograms,1):

        movi = data[:,:,i]
    #    movi[v.yl[i]:v.yu[i],v.xl[i]:v.xu[i]]=s.movie[i,v.yl[i]:v.yu[i],v.xl[i]:v.xu[i]]
        tiltM=np.append(tiltM,Barys_tilt2(xdata_tuple,movi[:,:]))
        
    return tiltC,tiltM


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma,theta):    
    (x,y)=xdata_tuple
    xo = float(xo)
    yo = float(yo)    
    a = 1/(2*sigma**2) 
    g = amplitude*np.exp( - (a*((x-xo)**2 + (y-yo)**2)))
    
    zt = g*np.cos(theta) - ((x-xo) )*np.sin(-theta)
    
    
    return zt.ravel()

def Barys_tilt2(xdata_tuple,mapa):
    thr=1
    (x,y)=xdata_tuple
    xp=np.sum((mapa[mapa > thr])*x[mapa>thr])/np.sum(mapa[mapa>thr])
    yp=np.sum((mapa[mapa > thr])*y[mapa>thr])/np.sum(mapa[mapa>thr])
    
    xn=np.sum((mapa[mapa < -thr])*x[mapa<-thr])/np.sum(mapa[mapa<-thr])    
    yn=np.sum((mapa[mapa < -thr])*y[mapa<-thr])/np.sum(mapa[mapa<-thr])
    
    tb = np.arctan((yp-yn)/(xp-xn))*180/np.pi
    
    return tb