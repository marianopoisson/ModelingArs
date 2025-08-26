import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io.idl import readsav
import csv
# import math

# PyMC 4.0 imports
import pymc as pm
import aesara.tensor as at 
import aesara
import arviz as az
import pymc.sampling_jax
import pytensor.tensor as pt
import pytensor

#import pymc3 as pm
#import theano.tensor as tt
#import theano as T
#from pymc3 import Model, Normal, Slice, sample, traceplot
#from pymc3.distributions import Interpolated
from scipy import stats

def set_ranges(datad):
    sz2,sz1,nMagnetograms = (np.shape(datad))
    
    xp1=[barys(datad[:,:,j],10)[0] for j in range(nMagnetograms)]
    yp1=[barys(datad[:,:,j],10)[1] for j in range(nMagnetograms)]
    xn1=[barys(datad[:,:,j],10)[2] for j in range(nMagnetograms)]
    yn1=[barys(datad[:,:,j],10)[3] for j in range(nMagnetograms)]
    
    tilt1=[np.arctan((yn1[j]-yp1[j])/(xn1[j]-xp1[j])) for j in range(nMagnetograms)]
    sar1=[np.sqrt( (xp1[j]-xn1[j])**2 + (yp1[j]-yn1[j])**2) for j in range(nMagnetograms)]
    
    flux1=[MFLUX(datad[:,:,j],10)[2] for j in range(nMagnetograms)]
    fluxn1=[MFLUX(datad[:,:,j],10)[1] for j in range(nMagnetograms)]
    fluxp1=[MFLUX(datad[:,:,j],10)[0] for j in range(nMagnetograms)]
    ep1=[elong(datad[:,:,j],10)[0] for j in range(nMagnetograms)]
    en1=[elong(datad[:,:,j],10)[1] for j in range(nMagnetograms)]
    spmax1=[elong(datad[:,:,j],10)[2] for j in range(nMagnetograms)]
    spmin1=[elong(datad[:,:,j],10)[3] for j in range(nMagnetograms)]
    snmax1=[elong(datad[:,:,j],10)[4] for j in range(nMagnetograms)]
    snmin1=[elong(datad[:,:,j],10)[5] for j in range(nMagnetograms)]


    tiltmax=tilt1[np.where(flux1 == np.max(flux1))[0][0]]
    
    xc1=[barys(np.abs(datad[:,:,j]),10)[0] for j in range(nMagnetograms)]
    yc1=[barys(np.abs(datad[:,:,j]),10)[1] for j in range(nMagnetograms)]
    
    nmax=np.where(flux1 == np.max(flux1))[0][0]
    Bmax=[-np.min(datad),np.max(datad)]
    
    if np.mean([xp1[j]-xn1[j] for j in range(nMagnetograms)]) < 0:
        hem=1
        amax=[snmin1[nmax],snmax1[nmax]]
    else:
        hem=-1
        amax=[spmin1[nmax],spmax1[nmax]]

    ranges={}


    ranges['a']=[0.5*amax[0],1.5*amax[1]]
    ranges['R']=[0.5*max(sar1),1.5*max(sar1)]
    ranges['axf']=[0.5*np.max(flux1),1.5*np.max(flux1)]
    ranges['alpha']=[tiltmax-1.5,tiltmax+1.5]
    ranges['xc']=[min(xc1)-5,max(xc1)+5]
    ranges['yc']=[min(yc1)-5,max(yc1)+5]
    ranges['hem']=hem


    return ranges

def barys(data,thr):

  pixsize=1.98*725E5 # 1.98 segundos de arco y 725 km en cm
  sz2,sz1 = np.shape(data)
  x=np.linspace(0,sz1-1,sz1)
  y=np.linspace(0,sz2-1,sz2)
  xv, yv = np.meshgrid(x, y)

  xp=np.sum(xv[data>thr]*data[data>thr])/np.sum(data[data>thr])
  xn=np.sum(xv[data<-thr]*data[data<-thr])/np.sum(data[data<-thr])

  yp=np.sum(yv[data>thr]*data[data>thr])/np.sum(data[data>thr])
  yn=np.sum(yv[data<-thr]*data[data<-thr])/np.sum(data[data<-thr])

  tl=np.arctan((yp-yn)/(xp-xn))
  sar=np.sqrt((xp-xn)**2+(yp-yn)**2)

  return xp,yp,xn,yn

def tilt(data,thr):

  pixsize=1.98*725E5 # 1.98 segundos de arco y 725 km en cm
  sz2,sz1 = np.shape(data)
  x=np.linspace(0,sz1-1,sz1)
  y=np.linspace(0,sz2-1,sz2)
  xv, yv = np.meshgrid(x, y)

  xp=np.sum(xv[data>thr]*data[data>thr])/np.sum(data[data>thr])
  xn=np.sum(xv[data<-thr]*data[data<-thr])/np.sum(data[data<-thr])

  yp=np.sum(yv[data>thr]*data[data>thr])/np.sum(data[data>thr])
  yn=np.sum(yv[data<-thr]*data[data<-thr])/np.sum(data[data<-thr])

  tl=np.arctan((yp-yn)/(xp-xn))
  sar=np.sqrt((xp-xn)**2+(yp-yn)**2)

  return tl,sar*pixsize

def sizes(data,thr):

  pixsize=1.98*725E5 # 1.98 segundos de arco y 725 km en cm
  sz2,sz1 = np.shape(data)
  x=np.linspace(0,sz1-1,sz1)
  y=np.linspace(0,sz2-1,sz2)
  xv, yv = np.meshgrid(x, y)

  xp=np.sum(xv[data>thr]*data[data>thr])/np.sum(data[data>thr])
  xn=np.sum(xv[data<-thr]*data[data<-thr])/np.sum(data[data<-thr])

  yp=np.sum(yv[data>thr]*data[data>thr])/np.sum(data[data>thr])
  yn=np.sum(yv[data<-thr]*data[data<-thr])/np.sum(data[data<-thr])

  msp=np.sum(np.sqrt((yv[data>thr]-yp)**2+(xv[data>thr]-xp)**2)*data[data>thr])/np.sum(data[data>thr])
  msn=np.sum(np.sqrt((yv[data<-thr]-yn)**2+(xv[data<-thr]-xn)**2)*data[data<-thr])/np.sum(data[data<-thr])

#  tl=np.arctan((yp-yn)/(xp-xn))
  sar=np.sqrt((xp-xn)**2+(yp-yn)**2)

  return msp,msn,sar

def sizes2(data,thr,xp=None,yp=None,xn=None,yn=None):

  pixsize=1.98*725E5 # 1.98 segundos de arco y 725 km en cm
  sz2,sz1 = np.shape(data)
  x=np.linspace(0,sz1-1,sz1)
  y=np.linspace(0,sz2-1,sz2)
  xv, yv = np.meshgrid(x, y)


  msp=np.sum(np.sqrt((yv[data>thr]-yp)**2+(xv[data>thr]-xp)**2)*data[data>thr])/np.sum(data[data>thr])
  msn=np.sum(np.sqrt((yv[data<-thr]-yn)**2+(xv[data<-thr]-xn)**2)*data[data<-thr])/np.sum(data[data<-thr])

#  tl=np.arctan((yp-yn)/(xp-xn))
  sar=np.sqrt((xp-xn)**2+(yp-yn)**2)

  return msp,msn,sar

def elong(data,thr):

  pixsize=1.98*725E5 # 1.98 segundos de arco y 725 km en cm
  sz2,sz1 = np.shape(data)
  x=np.linspace(0,sz1-1,sz1)
  y=np.linspace(0,sz2-1,sz2)
  xv, yv = np.meshgrid(x, y)

  xp=np.sum(xv[data>thr]*data[data>thr])/np.sum(data[data>thr])
  xn=np.sum(xv[data<-thr]*data[data<-thr])/np.sum(data[data<-thr])

  yp=np.sum(yv[data>thr]*data[data>thr])/np.sum(data[data>thr])
  yn=np.sum(yv[data<-thr]*data[data<-thr])/np.sum(data[data<-thr])

  sar=np.sqrt((xp-xn)**2+(yp-yn)**2)
  flp=np.sum(data[data>thr])
  fln=np.sum(abs(data[data<-thr]))

  x2p=np.sum(abs(data[data>thr])*(xv[data>thr]-xp)**2)/flp
  x2n=np.sum(abs(data[data<-thr])*(xv[data<-thr]-xn)**2)/fln

  y2p=np.sum(abs(data[data>thr])*(yv[data>thr]-yp)**2)/flp
  y2n=np.sum(abs(data[data<-thr])*(yv[data<-thr]-yn)**2)/fln

  xyp=np.sum(abs(data[data>thr])*(xv[data>thr]-xp)*(yv[data>thr]-yp))/flp
  xyn=np.sum(abs(data[data<-thr])*(xv[data<-thr]-xn)*(yv[data<-thr]-yn))/fln

  spmax=np.sqrt(0.5*(x2p+y2p+np.sqrt((x2p-y2p)**2+4*xyp**2)))
  spmin=np.sqrt(0.5*(x2p+y2p-np.sqrt((x2p-y2p)**2+4*xyp**2)))

  snmax=np.sqrt(0.5*(x2n+y2n+np.sqrt((x2n-y2n)**2+4*xyn**2)))
  snmin=np.sqrt(0.5*(x2n+y2n-np.sqrt((x2n-y2n)**2+4*xyn**2)))

  Ep=(spmax-spmin)/sar
  En=(snmax-snmin)/sar

  return Ep,En,spmax,spmin,snmax,snmin

def PILOOP(data):

  sz2,sz1 = np.shape(data)
  x=np.linspace(0,sz1-1,sz1)
  y=np.linspace(0,sz2-1,sz2)
  xx, yy = np.meshgrid(x, y)

#  print('.................. mag #'+str(mag)+' .........................')
  with pm.Model() as model:
    xs=pm.Uniform('xs',20,50)
    ys=pm.Uniform('ys',20,50)
    theta=pm.Data('theta',0)
    ts=pm.Categorical('ts',p=[0.5,0.5])
    hem=pm.Deterministic('hem',2*ts-1)

    

  def PIL(xs=xs,ys=ys,theta=theta,hem=hem):
    pil1=(xx-xs)*pt.cos(theta*np.pi/180.)+(yy-ys)*pt.sin(theta*np.pi/180.)
    mask1=pt.set_subtensor(pt.ones_like(pil1)[(pil1>0).nonzero()],0.0)
    mask2=pt.set_subtensor(pt.ones_like(pil1)[(pil1<0).nonzero()],0.0)

    return hem*(mask2-mask1)*np.abs(data)


  with model:
    sharedData = pytensor.shared((data))
    model_PIL_pm = pm.Deterministic('model_PIL_pm', PIL())
    observations = pm.Normal( "obs",  mu=model_PIL_pm, tau=1/10, observed=sharedData)  
    tracea = pm.sample(init='adapt_diag',tune=2000,return_inferencedata=True,
                      cores=4,chains=4)


  xs1=tracea.posterior['xs'].mean().values
  ys1=tracea.posterior['ys'].mean().values
  mask3=pt.set_subtensor(pt.ones_like(pt._shared(data))[(xx<np.abs(xs1-10)).nonzero()],0.0)
  mask4=pt.set_subtensor(pt.ones_like(pt._shared(data))[(yy<np.abs(ys1-10)).nonzero()],0.0)
  mask3b=pt.set_subtensor(pt.ones_like(pt._shared(data))[(xx>np.abs(xs1+10)).nonzero()],0.0)
  mask4b=pt.set_subtensor(pt.ones_like(pt._shared(data))[(yy>np.abs(ys1+10)).nonzero()],0.0)

  mask5=mask3*mask4*mask3b*mask4b

  apa=np.ravel(mask5.eval())

  ind=(pd.Series(np.abs(apa) == 1)).to_numpy().nonzero()

  sx=np.unravel_index(ind[0],(sz2,sz1))[1]
  sy=np.unravel_index(ind[0],(sz2,sz1))[0]
  epa=np.ravel(data)[ind[0]]

  with pm.Model() as model:
    xs=pm.Uniform('xs',xs1-2,xs1+2)
    ys=pm.Uniform('ys',ys1-2,ys1+2)
    theta1=pm.Uniform('theta1',-89,89)
    hem=pm.ConstantData('hem',tracea.posterior['hem'].mean().values)


  def PIL(xs=xs,ys=ys,theta=theta,hem=hem):
    pil1=(sx-xs)*pt.cos(theta*np.pi/180.)+(sy-ys)*pt.sin(theta*np.pi/180.)
    mask1=pt.set_subtensor(pt.ones_like(pil1)[(pil1>0).nonzero()],0.0)
    mask2=pt.set_subtensor(pt.ones_like(pil1)[(pil1<0).nonzero()],0.0)

    return hem*(mask2-mask1)*np.abs(epa)

  with model:
    sharedData = pytensor.shared(epa)
    model_PIL_pm = pm.Deterministic('model_PIL_pm', PIL())
    observations = pm.Normal( "obs",  mu=model_PIL_pm, tau=1/10, observed=sharedData)  
    trace = pm.sample(init='adapt_diag',tune=2000,return_inferencedata=True,
                      cores=4,chains=4)

    


  return trace

def MFLUX(data,thr):

  pixsize=1.98*725E5 # 1.98 segundos de arco y 725 km en cm
  pixarea=pixsize*pixsize
  datap=np.zeros_like(data)
  datan=np.zeros_like(data)
  datap[data>thr] = data[data>thr]
  datan[data<-thr] = data[data<-thr] 
  pos = np.sum(datap)*pixarea
  neg=np.sum(np.abs(datan))*pixarea
  total=(pos+neg)/2.

  return pos,neg,total

