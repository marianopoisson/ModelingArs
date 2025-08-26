import numpy as np
from scipy.io import readsav
from funciones.func4 import tilt,elong,PILOOP,MFLUX




def opencube(name=None):
    
    
    thr=0
    
    s=readsav('./movies/mov-AR'+name+'.sav')
    v=readsav('./movies/verts-'+name+'.sav')


    nMagnetograms,sz2,sz1, = np.shape(s.movie[np.where(s.lon >= -30)[0][0]
                                              .astype(int):
                                              np.max(np.where(s.lon <= 30))
                                              .astype(int),:,:])


    print(sz2,sz1,nMagnetograms)

    l1=np.where(s.lon >= -30)[0][0].astype(int)
    l2=np.where(s.lon <= 30)[0][-1].astype(int)

    print(s.time[l1],s.time[l2])

    tiempos1=s.time[l1:l2+1]
    for i in range(np.shape(tiempos1)[0]):
        tiempos1[i] = str(tiempos1[i][0:20],"utf-8").replace('Nov','11').replace('May','5').replace('Sep','9').replace('Oct','10').replace('Jan','1').replace('Dec','12').replace('Apr','4').replace('Aug','8').replace('Jul','7')
        
    # levanta los magnetrogramas y los guarda ordenados en data
    nMagnetograms=l2+1-l1
    #data = np.zeros_like(s.movie[l1:l2+1,:,:])  
    data = np.zeros((sz2, sz1, nMagnetograms))

  #  print(np.shape(data))

  #  for i in range(l1,l2+1,1):
  #      data[v.yl[i]:v.yu[i],v.xl[i]:v.xu[i],i-l1]=s.movie[i,v.yl[i]:v.yu[i],v.xl[i]:v.xu[i]]

    #data[abs(data) < 100 ] = 0

    data2 = np.zeros((np.max(v.yu)-np.min(v.yl),np.max(v.xu)-np.min(v.xl), nMagnetograms))
    mask = np.zeros_like(data)

    for i in range(l1,l2+1,1):
        data2[:,:,i-l1] = s.movie[i,np.min(v.yl):np.max(v.yu),np.min(v.xl):np.max(v.xu)]
        mask[v.yl[i]:v.yu[i],v.xl[i]:v.xu[i],i-l1]=1
    
    mask=mask[np.min(v.yl):np.max(v.yu),np.min(v.xl):np.max(v.xu),:]
  #  data2[:,:,:] = data[np.min(v.yl):np.max(v.yu),np.min(v.xl):np.max(v.xu), :]
    sz2,sz1,nMagnetograms = (np.shape(data2))
    print(sz2,sz1)
    
  #  ff=[MFLUX(data2[:,:,j],10)[2] for j in range(nMagnetograms)]
  #  index_max = max(range(len(ff)), key=ff.__getitem__)
  #  datad=data2[:,:,0:index_max]
    datad=data2
#    datad[abs(datad)<thr]=0.0
    sz2,sz1,nMagnetograms = (np.shape(datad))
    
    
    return datad,tiempos1