import numpy as np
import pandas as pd
# import math

#import pymc3 as pm
#import theano.tensor as tt
#import theano as T


def modelmag(xv,yv,params=None,hem=None):
    
#   global xv,yv

    a=params['a']
    B0=params['axf']/(np.pi*a**2)
    R=params['R']
    da=params['da']
    xc=params['xc']
    yc=params['yc']
    alpha=params['alpha']
    N0=params['N0']

    f=0
    g=0

    x=(xv-xc+0.5)*np.cos(alpha) + (yv-yc+0.5)*np.sin(alpha)
    y=-(xv-xc+0.5)*np.sin(alpha) + (yv-yc+0.5)*np.cos(alpha)
    xr=np.sqrt(x**2+((1-da)*(R+a))**2) - R
    rho=np.sqrt((xr)**2 + (y)**2)    
    u=np.sqrt(x**2 + ((1-da)*(R+a))**2)
    costh=xr/rho
    Nt=N0*(1+f*(rho/a)**2)
    mag = (x*(1+g*(costh))- 2*Nt*(1-da)*(R+a)*y/u)*(-1)*B0*np.exp((-1)*(rho/a)**2)/u
    mag0 = x*(-1)*B0*np.exp((-1)*(rho/a)**2)/u

    return hem*mag,hem*mag0


def modelmagAx(xv,yv,params=None,hem=None):
    
#   global xv,yv

    a=params['a']
    B0=params['axf']/(np.pi*a**2)
    R=params['R']
    da=params['da']
    xc=params['xc']
    yc=params['yc']
    alpha=params['alpha']
    N0=params['N0']

    f=0
    g=0

    x=(xv-xc+0.5)*np.cos(alpha) + (yv-yc+0.5)*np.sin(alpha)
    y=-(xv-xc+0.5)*np.sin(alpha) + (yv-yc+0.5)*np.cos(alpha)
    xr=np.sqrt(x**2+((1-da)*(R+a))**2) - R
    rho=np.sqrt((xr)**2 + (y)**2)    
    u=np.sqrt(x**2 + ((1-da)*(R+a))**2)
    costh=xr/rho
    Nt=N0*(1+f*(rho/a)**2)
    Nt=0
    mag = (x*(1+g*(costh))- 2*Nt*(1-da)*(R+a)*y/u)*(-1)*B0*np.exp((-1)*(rho/a)**2)/u
    mag0 = x*(-1)*B0*np.exp((-1)*(rho/a)**2)/u

    return hem*mag,hem*mag0

def modelmagf(xv,yv,params=None,hem=None):
    
#   global xv,yv

    a=params['a']
    B0=params['axf']/(np.pi*a**2)
    R=params['R']
    da=params['da']
    xc=params['xc']
    yc=params['yc']
    alpha=params['alpha']
    N0=params['N0']
    f=params['f']
    g=0

    x=(xv-xc+0.5)*np.cos(alpha) + (yv-yc+0.5)*np.sin(alpha)
    y=-(xv-xc+0.5)*np.sin(alpha) + (yv-yc+0.5)*np.cos(alpha)
    xr=np.sqrt(x**2+((1-da)*(R+a))**2) - R
    rho=np.sqrt((xr)**2 + (y)**2)    
    u=np.sqrt(x**2 + ((1-da)*(R+a))**2)
    costh=xr/rho
    Nt=N0*(1+f*(rho/a)**2)
    mag = (x*(1+g*(costh))- 2*Nt*(1-da)*(R+a)*y/u)*(-1)*B0*np.exp((-1)*(rho/a)**2)/u


    return hem*mag

def modelmag2(xv,yv,params=None,hem=None):
    
#   global xv,yv

    a=params['a']
    B0=params['axf']/(np.pi*a**2)
    R=params['R']
    da=params['da']
    xc=params['xc']
    yc=params['yc']
    alpha=params['alpha']
    N0=params['N0']
    ar=params['ar']
    f=0
    g=0

    x=(xv-xc+0.5)*np.cos(alpha) + (yv-yc+0.5)*np.sin(alpha)
    y=-(xv-xc+0.5)*np.sin(alpha) + (yv-yc+0.5)*np.cos(alpha)
    xr=np.sqrt(x**2+((1-da)*(R+a))**2) - R
    rho=np.sqrt((xr)**2 + (y)**2)    
    u=np.sqrt(x**2 + ((1-da)*(R+a))**2)
    costh=xr/rho
    Nt=N0*(1+f*(rho/a)**2)
    
            
    ss=(x- 2*Nt*(1-da)*(R+a)*y/u)
    ap=a-(ss/np.abs(ss))*ar*da  #np.arcsin(ff)
    mag = (x- 2*Nt*(1-da)*(R+a)*y/u)*((B0*a**2)/ap**2)*(-1)*np.exp((-1)*(rho/ap)**2)/u


    return hem*mag