import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
#from robustbase import Qn
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
import pickle
import pandas as pd
import glob
import os


def autocorr_decay(dk,A,tau,B):
    return A*(np.exp(-(dk/tau))+B)


def main(root_pth,max_lag):
    models_list = [[1,0,1]]
    task_list = ['Rest']
    nlags=10
    xdata=np.arange(nlags)
    subj_list = []
    fd_list = []
    for task in task_list:
        print(task)
        for f in glob.glob(os.path.join(root_pth,task)+'/*.txt'):
            subj = f.split('/')[-1].split('_')[0]
            subj_list.append(subj)
            fd = np.loadtxt('/home/annatruzzi/MRIautocorr_ARMA/Results/camcanFD/camcan_{}_{}_fd.txt'.format(task.lower(),subj),delimiter='\t',skiprows=1)
            fd_list.append(np.mean(fd,axis=0))
        print(len(subj_list))
        print(len(fd_list))
        df_dict = {'subj': subj_list,
                    'fd':fd_list}
        df = pd.DataFrame(df_dict)
        subj_lowmovement = np.array(df.loc[df['fd']<=0.25]['subj'])
        fd_lowmovement = np.array(df.loc[df['fd']<=0.25]['fd'])
        print(subj_lowmovement)
        print(len(subj_lowmovement))
        np.savetxt('/home/annatruzzi/MRIautocorr_ARMA/Results/camcan/camcan_lowmovement_sublist.csv',subj_lowmovement, fmt='%s')
        
        for model in models_list:
            print(model)
            alltau = np.zeros((len(subj_lowmovement),400))
            allrho1 = np.zeros((len(subj_lowmovement),400))
            allrho2 = np.zeros((len(subj_lowmovement),400))
            modeloutput = {}
            for i,subj in enumerate(subj_lowmovement):
                print(i, subj)
                modeloutput[str(subj)] = {}
                filename = os.path.join(root_pth,task)+'/'+subj+'_schaefer_400ROI_'+task.lower()+'.txt'
                ts_df = np.loadtxt(filename)
                timescale = np.zeros((ts_df.shape[1], nlags))
                ar_list = []
                ma_list = []                
                for ROI in range(0,ts_df.shape[1]):
                    xc=ts_df[:,ROI]-np.mean(ts_df[:,ROI])
                    fullcorr=np.correlate(xc, xc, mode='full')
                    fullcorr=fullcorr / np.max(fullcorr)
                    start=len(fullcorr) // 2
                    stop=start+max_lag
                    timescale[ROI,:]=fullcorr[start:stop]

                for ROI in range(0,ts_df.shape[1]):
                    mod = ARIMA(endog=ts_df[9:,ROI], order=(model[0],model[1],model[2]), enforce_stationarity=False,enforce_invertibility=False) #skipping initial TRs to allow for the field to stabilize
                    res=mod.fit()
                    ar=res.arparams
                    rho0 = 1
                    ar_list.append(res.arparams)
                    if model[2]>0:
                        ma_list.append(res.maparams)                        
                    if model[0] == 1:
                        try:
                            rho1 = ar[0]
                            rho2 = ar[0] * rho1
                            allrho1[i,ROI] = rho1
                            allrho2[i,ROI] = rho2
                        except:
                            allrho1[i,ROI] = np.nan
                            allrho2[i,ROI] = np.nan
                        alltau[i,ROI]=-1/np.log(rho2 / rho1)

                modeloutput[str(subj)]['ar'] = ar_list
                if model[2]>0:
                    modeloutput[str(subj)]['ma'] = ma_list
            tau_df=pd.DataFrame(data=alltau[0:,0:],
                    index=[i for i in range(alltau.shape[0])],
                    columns=[i for i in range(alltau.shape[1])])
            tau_df.insert(loc=0, column='subj', value=subj_lowmovement)
            tau_df.insert(loc=1, column='fd', value=fd_lowmovement)

            allrho1_df=pd.DataFrame(data=allrho1[0:,0:],
                    index=[i for i in range(allrho1.shape[0])],
                    columns=[i for i in range(allrho1.shape[1])])
            allrho1_df.insert(loc=0, column='subj', value=subj_lowmovement)
            allrho1_df.insert(loc=1, column='fd', value=fd_lowmovement)

            allrho2_df=pd.DataFrame(data=allrho2[0:,0:],
                    index=[i for i in range(allrho2.shape[0])],
                    columns=[i for i in range(allrho2.shape[1])])
            allrho2_df.insert(loc=0, column='subj', value=subj_lowmovement)
            allrho2_df.insert(loc=1, column='fd', value=fd_lowmovement)

            tau_df.to_csv('/home/annatruzzi/MRIautocorr_ARMA/Results/camcan/estimatedtau_camcan_{}_{}_lowmovement.csv'.format(task,model))
            np.savetxt('/home/annatruzzi/MRIautocorr_ARMA/Results/camcan/estimatedtau_camcan_{}_{}_lowmovement.txt'.format(task,model),alltau)

            allrho1_df.to_csv('/home/annatruzzi/MRIautocorr_ARMA/Results/camcan/estimatedrho1_camcan_{}_{}_lowmovement.csv'.format(task,model))
            np.savetxt('/home/annatruzzi/MRIautocorr_ARMA/Results/camcan/estimatedrho1_camcan_{}_{}_lowmovement.txt'.format(task,model),allrho1)

            allrho2_df.to_csv('/home/annatruzzi/MRIautocorr_ARMA/Results/camcan/estimatedrho2_camcan_{}_{}_lowmovement.csv'.format(task,model))
            np.savetxt('/home/annatruzzi/MRIautocorr_ARMA/Results/camcan/estimatedrho2_camcan_{}_{}_lowmovement.txt'.format(task,model),alltau)

            with open('/home/annatruzzi/MRIautocorr_ARMA/Results/camcan/params_camcan_{}_{}_lowmovement.pickle'.format(task,model),'wb') as handle:
                    pickle.dump(modeloutput, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    root_pth = '/camcan/schaefer_parc/'
    max_lag = 10
    main(root_pth,max_lag)
