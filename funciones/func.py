import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io.idl import readsav
import csv
# import math

#import pymc3 as pm
#import theano.tensor as tt
#import theano as T
#from pymc3 import Model, Normal, Slice, sample, traceplot
#from pymc3.distributions import Interpolated
from scipy import stats


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
    xs=pm.DiscreteUniform('xs',20,50)
    ys=pm.DiscreteUniform('ys',20,50)
    theta=pm.Data('theta',0) 
    hem=pm.DiscreteUniform('hem',-1,1)

    

  def PIL(xs=xs,ys=ys,theta=theta,hem=hem):
    pil1=(xx-xs)*tt.cos(theta*np.pi/180.)+(yy-ys)*tt.sin(theta*np.pi/180.)
    mask1=tt.set_subtensor(tt.ones_like(pil1)[(pil1>0).nonzero()],0.0)
    mask2=tt.set_subtensor(tt.ones_like(pil1)[(pil1<0).nonzero()],0.0)

    return hem*(mask2-mask1)*np.abs(data)


  with model:
    sharedData = T.shared((data))
    model_PIL_pm = pm.Deterministic('model_PIL_pm', PIL())
    observations = pm.Normal( "obs",  mu=model_PIL_pm, tau=1/10, observed=sharedData)  
    tracea = pm.sample(init='adapt_diag',tune=2000,return_inferencedata=True,
                      cores=4,chains=4)


  xs1=tracea.posterior['xs'].mean().values
  ys1=tracea.posterior['ys'].mean().values
  mask3=tt.set_subtensor(tt.ones_like(tt._shared(data))[(xx<np.abs(xs1-10)).nonzero()],0.0)
  mask4=tt.set_subtensor(tt.ones_like(tt._shared(data))[(yy<np.abs(ys1-10)).nonzero()],0.0)
  mask3b=tt.set_subtensor(tt.ones_like(tt._shared(data))[(xx>np.abs(xs1+10)).nonzero()],0.0)
  mask4b=tt.set_subtensor(tt.ones_like(tt._shared(data))[(yy>np.abs(ys1+10)).nonzero()],0.0)

  mask5=mask3*mask4*mask3b*mask4b

  apa=np.ravel(mask5.eval())

  ind=(pd.Series(np.abs(apa) == 1)).to_numpy().nonzero()

  sx=np.unravel_index(ind[0],(sz2,sz1))[1]
  sy=np.unravel_index(ind[0],(sz2,sz1))[0]
  epa=np.ravel(data)[ind[0]]

  with pm.Model() as model:
    xs=pm.DiscreteUniform('xs',xs1-2,xs1+2)
    ys=pm.DiscreteUniform('ys',ys1-2,ys1+2)
    theta1=pm.DiscreteUniform('theta1',-89*10,89*10)
    theta=pm.Deterministic('theta',theta1/10)
    hem=pm.Data('hem',-1)


  def PIL(xs=xs,ys=ys,theta=theta,hem=hem):
    pil1=(sx-xs)*np.cos(theta*np.pi/180.)+(sy-ys)*np.sin(theta*np.pi/180.)
    mask1=tt.set_subtensor(tt.ones_like(pil1)[(pil1>0).nonzero()],0.0)
    mask2=tt.set_subtensor(tt.ones_like(pil1)[(pil1<0).nonzero()],0.0)

    return hem*(mask2-mask1)*np.abs(epa)

  with model:
    sharedData = T.shared(epa)
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

