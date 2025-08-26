


def MODELA():
    with pm.Model() as modelA:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl,shape=nMagnetograms2)
    #     a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2,shape=nMagnetograms2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10,shape=nMagnetograms2)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
        da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
    #    xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
    #    yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                   
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl,shape=nMagnetograms2)
     #   R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)

    def modelmagA(a=a,R=R,B0=B0,N0=N0,da=da,xc=xc,yc=yc,alpha=alpha):

        f=0
        g=0

        a0=a[sz[0]]
        R0=R[sz[0]]
        B00=B0[sz[0]]
        da0=da[sz[0]]
        N00=N0[sz[0]]

        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N00*(1+f*(rho/a0)**2)
        mag = (x*(1+g*(costh))- 2*Nt*(1-da0)*(R0+a0)*y/u)*(-1)*B00*pt.exp((-1)*(rho/a0)**2)/u

        return hem*mag

    with modelA:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagA())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceA = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})
    
    return traceA


def MODELB():
    with pm.Model() as modelB:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl)
    #     a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
        da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
    #    xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
    #    yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                   
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl)
     #   R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)

    def modelmagB(a=a,R=R,B0=B0,Nt=N0,da=da,xc=xc,yc=yc,alpha=alpha):

        f=0
        g=0

        a0=a
        R0=R
        B00=B0
        da0=da[sz[0]]

        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N0*(1+f*(rho/a0)**2)
        mag = (x*(1+g*(costh))- 2*Nt*(1-da0)*(R0+a0)*y/u)*(-1)*B00*pt.exp((-1)*(rho/a0)**2)/u

        return hem*mag

    with modelB:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagB())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceB = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})
    
    return traceB


def MODELC():
    with pm.Model() as modelC:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl)
      #  a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
        da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
    #    xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
    #    yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl)
      #  R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
      #  f=pm.Uniform('f',-1,1)
       # ar=pm.ConstantData('ar',value=0)    
        ar=pm.Uniform('ar',0,2)    
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)


    def modelmagC(a=a,R=R,B0=B0,da=da,N0=N0,xc=xc,yc=yc,alpha=alpha,ar=ar):

        f=0
        g=0

        a0=a
        R0=R
        B00=B0
        da0=da[sz[0]]

        ff=(1-da0)*(R0+a0)/R0
        ff=pt.clip(ff,0,1)
       # ff[ff>1]=1


        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N0*(1+f*(rho/a0)**2)

        ss=(x- 2*Nt*(1-da0)*(R0+a0)*y/u)
        ap=a0-(ss/np.abs(ss))*ar*da0  #np.arcsin(ff)
        mag = (x- 2*Nt*(1-da0)*(R0+a0)*y/u)*((B00*a0**2)/ap**2)*(-1)*pt.exp((-1)*(rho/ap)**2)/u

        return hem*mag

    with modelC:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagC())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceC = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})

    return traceC

def MODELD():
    
    with pm.Model() as modelD:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl)
    #     a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
        da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
    #    xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
    #    yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                   
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl)
        f = pm.Uniform('f',-1,1)
     #   R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)

    def modelmagD(a=a,R=R,B0=B0,Nt=N0,da=da,xc=xc,yc=yc,alpha=alpha,f=f):


        g=0

        a0=a
        R0=R
        B00=B0
        da0=da[sz[0]]

        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N0*(1+f*(rho/a0)**2)
        mag = (x*(1+g*(costh))- 2*Nt*(1-da0)*(R0+a0)*y/u)*(-1)*B00*pt.exp((-1)*(rho/a0)**2)/u

        return hem*mag

    with modelD:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagD())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceD = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})
    
    return traceD


def MODELE():
    with pm.Model() as modelE:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl)
      #  a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
        da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
     #   xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
     #   yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl)
      #  R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
        f=pm.Uniform('f',-1,1)
       # ar=pm.ConstantData('ar',value=0)    
        ar=pm.Uniform('ar',0,2)    
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)


    def modelmagE(a=a,R=R,B0=B0,da=da,f=f,N0=N0,xc=xc,yc=yc,alpha=alpha,ar=ar):


        g=0

        a0=a
        R0=R
        B00=B0
        da0=da[sz[0]]

        ff=(1-da0)*(R0+a0)/R0
        ff=pt.clip(ff,0,1)
       # ff[ff>1]=1


        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N0*(1+f*(rho/a0)**2)

        ss=(x- 2*Nt*(1-da0)*(R0+a0)*y/u)
        ap=a0-(ss/np.abs(ss))*ar*da0  #np.arcsin(ff)
        mag = (x- 2*Nt*(1-da0)*(R0+a0)*y/u)*((B00*a0**2)/ap**2)*(-1)*pt.exp((-1)*(rho/ap)**2)/u

        return hem*mag

    with modelE:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagE())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceE = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})

    return traceE


def MODELF():
    with pm.Model() as modelF:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl)
      #  a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
        da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
    #    xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
    #    yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl,shape=nMagnetograms2)
      #  R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
      #  f=pm.Uniform('f',-1,1)
       # ar=pm.ConstantData('ar',value=0)    
        ar=pm.Uniform('ar',0,2)    
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)


    def modelmagF(a=a,R=R,B0=B0,da=da,N0=N0,xc=xc,yc=yc,alpha=alpha,ar=ar):

        f=0
        g=0

        a0=a
        R0=R[sz[0]]
        B00=B0
        da0=da[sz[0]]

        ff=(1-da0)*(R0+a0)/R0
        ff=pt.clip(ff,0,1)
       # ff[ff>1]=1


        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N0*(1+f*(rho/a0)**2)

        ss=(x- 2*Nt*(1-da0)*(R0+a0)*y/u)
        ap=a0-(ss/np.abs(ss))*ar*da0  #np.arcsin(ff)
        mag = (x- 2*Nt*(1-da0)*(R0+a0)*y/u)*((B00*a0**2)/ap**2)*(-1)*pt.exp((-1)*(rho/ap)**2)/u

        return hem*mag

    with modelF:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagF())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceF = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})

    return traceF



def MODELG():
    with pm.Model() as modelG:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl,shape=nMagnetograms2)
      #  a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
        da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
    #    xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
    #    yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl,shape=nMagnetograms2)
      #  R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
      #  f=pm.Uniform('f',-1,1)
       # ar=pm.ConstantData('ar',value=0)    
        ar=pm.Uniform('ar',0,2)    
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)


    def modelmagG(a=a,R=R,B0=B0,da=da,N0=N0,xc=xc,yc=yc,alpha=alpha,ar=ar):

        f=0
        g=0

        a0=a[sz[0]]
        R0=R[sz[0]]
        B00=B0[sz[0]]
        da0=da[sz[0]]

        ff=(1-da0)*(R0+a0)/R0
        ff=pt.clip(ff,0,1)
       # ff[ff>1]=1


        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N0*(1+f*(rho/a0)**2)

        ss=(x- 2*Nt*(1-da0)*(R0+a0)*y/u)
        ap=a0-(ss/np.abs(ss))*ar*da0  #np.arcsin(ff)
        mag = (x- 2*Nt*(1-da0)*(R0+a0)*y/u)*((B00*a0**2)/ap**2)*(-1)*pt.exp((-1)*(rho/ap)**2)/u

        return hem*mag

    with modelG:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagG())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceG = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})

    return traceG


def MODELH():
    with pm.Model() as modelH:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl,shape=nMagnetograms2)
      #  a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
        da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
    #    xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
    #    yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl,shape=nMagnetograms2)
      #  R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
        f=pm.Uniform('f',-1,1)
       # ar=pm.ConstantData('ar',value=0)    
        ar=pm.Uniform('ar',0,2)    
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)


    def modelmagH(a=a,R=R,B0=B0,da=da,N0=N0,xc=xc,yc=yc,alpha=alpha,ar=ar,f=f):

        #f=0
        g=0

        a0=a[sz[0]]
        R0=R[sz[0]]
        B00=B0[sz[0]]
        da0=da[sz[0]]

        ff=(1-da0)*(R0+a0)/R0
        ff=pt.clip(ff,0,1)
       # ff[ff>1]=1


        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N0*(1+f*(rho/a0)**2)

        ss=(x- 2*Nt*(1-da0)*(R0+a0)*y/u)
        ap=a0-(ss/np.abs(ss))*ar*da0  #np.arcsin(ff)
        mag = (x- 2*Nt*(1-da0)*(R0+a0)*y/u)*((B00*a0**2)/ap**2)*(-1)*pt.exp((-1)*(rho/ap)**2)/u

        return hem*mag

    with modelH:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagH())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceH = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})

    return traceH


def MODELJ():
    
    ff=(np.sum(np.abs(datad)*(datad < 0),axis=(0,1))/fmax)[::3]
    with pm.Model() as modelJ:
        a = pm.Uniform('a',ranges['a'][0]/scl,ranges['a'][1]/scl,shape=nMagnetograms2)
      #  a = pm.Normal('a',mu=aM,sigma=0.5/scl,shape=nMagnetograms2)        
        axf= pm.Uniform('axf',fmax/scl**2,1.5*fmax/scl**2)
    #       axf= pm.Normal('axf',mu=fmax/scl**2,sigma=0.01*fmax/scl**2)        
      #  B0 = pm.Uniform('B0',ranges['B0'][0],ranges['B0'][1])
        B0 = pm.Deterministic('B0',axf/(np.pi*a**2))
        N01 = pm.Gamma('N01',alpha=2.5,beta=10)
        ts=pm.Categorical('ts',p=[0.5,0.5])
    #    ts = pm.ConstantData('ts',value=-1)
        N0 = pm.Deterministic('N0',(2*ts-1)*N01)
       # da=pm.Uniform('da',0,1,shape=nMagnetograms2)
        da=pm.TruncatedNormal('da',mu=ff,sigma=0.05,lower=0,upper=1,shape=nMagnetograms2)                        
        alpha = pm.Uniform('alpha',ranges['alpha'][0],ranges['alpha'][1],shape=nMagnetograms2)
        xc = pm.Uniform('xc',ranges['xc'][0]/scl,ranges['xc'][1]/scl,shape=nMagnetograms2)
        yc = pm.Uniform('yc',ranges['yc'][0]/scl,ranges['yc'][1]/scl,shape=nMagnetograms2)
    #    xc = pm.ConstantData('xc',value=np.array(xc1[::cad])/scl)
    #    yc = pm.ConstantData('yc',value=np.array(yc1[::cad])/scl)                
        R = pm.Uniform('R',ranges['R'][0]/scl,ranges['R'][1]/scl)
      #  R = pm.Normal('R',mu=RM,sigma=0.5/scl,shape=nMagnetograms2)        
      #  d=pm.Deterministic('d',(1-da)*(R+a))
        f=pm.Uniform('f',-1,1)
       # ar=pm.ConstantData('ar',value=0)    
        ar=pm.Uniform('ar',0,2)    
        sg=pm.Uniform('sg',10,300,shape=nMagnetograms2)
    #      sg=pm.Data('sg',sig1)


    def modelmagJ(a=a,R=R,B0=B0,da=da,N0=N0,xc=xc,yc=yc,alpha=alpha,ar=ar,f=f):

        #f=0
        g=0

        a0=a[sz[0]]
        R0=R
        B00=B0[sz[0]]
        da0=da[sz[0]]

        ff=(1-da0)*(R0+a0)/R0
        ff=pt.clip(ff,0,1)
       # ff[ff>1]=1


        x=(sx-xc[sz[0]]+0.5)*(pt.cos(alpha[sz[0]])) + (sy-yc[sz[0]]+0.5)*(pt.sin(alpha[sz[0]]))
        y=-(sx-xc[sz[0]]+0.5)*pt.sin(alpha[sz[0]]) + (sy-yc[sz[0]]+0.5)*pt.cos(alpha[sz[0]])
        xr=pt.sqrt(x**2+((1-da0)*(R0+a0))**2) - R0
        rho=pt.sqrt(xr**2 + y**2)
        u=pt.sqrt(x**2 + ((1-da0)*(R0+a0))**2)
        costh=xr/rho
        Nt=N0*(1+f*(rho/a0)**2)

        ss=(x- 2*Nt*(1-da0)*(R0+a0)*y/u)
        ap=a0-(ss/np.abs(ss))*ar*da0  #np.arcsin(ff)
        mag = (x- 2*Nt*(1-da0)*(R0+a0)*y/u)*((B00*a0**2)/ap**2)*(-1)*pt.exp((-1)*(rho/ap)**2)/u

        return hem*mag

    with modelJ:
        sharedData = pytensor.shared(np.array(apa[ind[0]]))
        model_mag_pm = pm.Deterministic('model_mag_pm', modelmagJ())
        observations = pm.Normal( "obs",  mu=model_mag_pm[0,:], sigma=sg[sz[0]], observed=sharedData)
        traceJ = pm.sample(idata_kwargs={"log_likelihood": True})
    #    traceB = pm.sampling.jax.sample_numpyro_nuts(idata_kwargs={"log_likelihood": True})

    return traceJ
