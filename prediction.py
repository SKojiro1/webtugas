import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
# Enable plots inside the Jupyter NotebookLet the
#%matplotlib inline

from scipy.stats import norm, kurtosis, skew, variation
from sklearn.preprocessing import StandardScaler
import os

def prediction(data):
    df_=[] 
    df = pd.DataFrame(df_, columns=['mean' ,'std' ,'max' ,'min' ,'kurtosis' ,'skew' ,'psd_mean' ,'psd_std' ,'psd_kurtosis' ,'psd_skew' ,'psd_max' ,'psd_min'])
    data_norm = pd.DataFrame(data, columns=['sensor']) 
    dfeat_=[] 
    dfeat = pd.DataFrame(dfeat_, columns=['mean'])  

    n=np.size(data_norm)
    step=25
    step_d=500
    count=0
    for i in range(0,n,step):
        if i+step_d<n:
            label=0
            count=count+1

    npsize=count
    mean=np.zeros(npsize)
    std=np.zeros(npsize)
    dkurtosis=np.zeros(npsize)
    dskew=np.zeros(npsize)
    max=np.zeros(npsize)
    min=np.zeros(npsize)
    freqs=np.zeros(npsize)
    psd=np.zeros(npsize)
    PSD_mean=np.zeros(npsize)
    PSD_std=np.zeros(npsize)
    PSD_kurtosis=np.zeros(npsize)
    PSD_skew=np.zeros(npsize)
    PSD_max=np.zeros(npsize)
    PSD_min=np.zeros(npsize)

    qq=data_norm['sensor']
    step=25
    step_d=500
    count=0
    for i in range(0,n,step):
        if i+step_d<n:
            mean[count]=qq[i:i+step_d].mean()
            std[count]=qq[i:i+step_d].std()
            dkurtosis[count]=qq[i:i+step_d].kurtosis()
            dskew[count]=qq[i:i+step_d].skew()
            max[count]=np.sqrt(qq[i:i+step_d].max()**2)
            min[count]=np.sqrt(qq[i:i+step_d].min()**2)

            freqs, psd = signal.welch(qq[i:i+step_d],nperseg=100, average='median')
            PSD_mean[count]=psd.mean()
            PSD_std[count]=psd.std()
            PSD_kurtosis[count]=kurtosis(psd)
            PSD_skew[count]=skew(psd)
            PSD_max[count]=np.sqrt(psd.max())
            PSD_min[count]=np.sqrt(psd.min())
            count=count+1

    #save to dataframe
    dfeat['mean']=mean
    dfeat['std']=std
    dfeat['max']=max
    dfeat['min']=min
    dfeat['kurtosis']=dkurtosis
    dfeat['skew']=dskew

    dfeat['psd_mean']=PSD_mean
    dfeat['psd_std']=PSD_std
    dfeat['psd_kurtosis']=PSD_kurtosis
    dfeat['psd_skew']=PSD_skew
    dfeat['psd_max']=PSD_max
    dfeat['psd_min']=PSD_max

    #load model
    import joblib
    filename = os.path.join('model', 'bearing_rfclass_170921.sav')  
    #filename = '/model/bearing_rfclass_170921.sav'
    # load the model from disk
    BRCM = joblib.load(filename)

    data2pre=dfeat.to_numpy()
    predict = BRCM.predict(data2pre)
    dpre = pd.DataFrame(predict, columns=['predict'])
    modus=dpre.mode()
    percen=dpre.value_counts(normalize=True) * 100
    return np.array([predict, percen])