import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
#from robustbase import Qn
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import pickle
import pandas as pd
import glob
import os
import random

def autocorr_decay(dk,A,tau,B):
    return A*(np.exp(-(dk/tau))+B)


def main(root_pth,max_lag):
    random.seed(42)
    models_list = [[1,1,2],[2,0,0],[2,0,1],[2,1,1],[1,1,0],[1,0,1],[1,1,1]]
    nlags=10
    xdata=np.arange(nlags)
    subj_list = []
    onlysubj_list = []
    sess_list = []
    fd = pd.read_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/FDall_dhcp.csv')
    for f1 in glob.glob(root_pth +'/*'):
        subj = f1.split('/')[-1]
        for f2 in glob.glob(os.path.join(root_pth,subj) +'/*'):
            sess = f2.split('/')[-1]
            if np.array(fd.loc[(fd['subj']==subj) & (fd['sess']==sess)]['fd']) < 0.25:
                subj_list.append([subj,sess])
                onlysubj_list.append(subj)
                sess_list.append(sess)
    half_sample = random.sample(subj_list, len(subj_list)//2)
    replication_sample = [item for item in subj_list if item not in half_sample]
    subj_dict = {'half':half_sample,
                'rep':replication_sample}
    subj_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in subj_dict.items()]))
    subj_df.to_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/half_and_replication_subj.csv')
    onlysubj_half = [a[0] for a in half_sample]
    onlysess_half = [a[1] for a in half_sample]
    for model in models_list:
        print(model)
        alltau = np.zeros((len(half_sample),400))
        allrho1 = np.zeros((len(half_sample),400))
        allrho2 = np.zeros((len(half_sample),400))
        modeloutput = {}
        for i,subj in enumerate(half_sample):
            modeloutput[str(subj)] = {}
            filename = root_pth+'/{}/{}/{}_{}_schaefer_timecourse_intermediate.txt'.format(subj[0],subj[1],subj[0],subj[1])
            ts_df = np.loadtxt(filename)
            ar_list = []
            ma_list = []
            timescale = np.zeros((ts_df.shape[1], nlags))
            for ROI in range(0,ts_df.shape[1]):
                    xc=ts_df[9:,ROI]-np.mean(ts_df[9:,ROI])
                    fullcorr=np.correlate(xc, xc, mode='full')
                    fullcorr=fullcorr / np.max(fullcorr)
                    start=len(fullcorr) // 2
                    stop=start+max_lag
                    timescale[ROI,:]=fullcorr[start:stop]
            
            for ROI in range(0,ts_df.shape[1]):
                print(model)   
                mod = ARIMA(endog=ts_df[9:,ROI], order=(model[0],model[1],model[2]), enforce_stationarity=False) #skipping initial TRs to allow for the field to stabilize
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
                        alltau[i,ROI]=-1/np.log(rho2 / rho1)
                    except:
                        allrho1[i,ROI] = np.nan
                        allrho2[i,ROI] = np.nan
                        alltau[i,ROI]= np.nan
                if model[0] == 2:
                    try:
                        rho1 = ar[0]/(1-ar[1])
                        rho2 = ar[0] * rho1 + ar[1] * rho0
                        allrho1[i,ROI] = rho1
                        allrho2[i,ROI] = rho2
                        alltau[i,ROI]=-1/np.log(rho2 / rho1)
                    except:
                        allrho1[i,ROI] = np.nan
                        allrho2[i,ROI] = np.nan
                        alltau[i,ROI]= np.nan
                if model[0] == 3:
                    rho1 = (ar[0]+(ar[2]*ar[1]))/(1-ar[1]-(ar[2]*ar[0])-ar[2]**2)
                    rho2 = ar[0] * rho1 + ar[1] + ar[2] * rho1
                    alltau[i,ROI]=-1/np.log(rho2 / rho1)
            print(alltau.shape)
            modeloutput[str(subj)]['ar'] = ar_list
            if model[2]>0:
                modeloutput[str(subj)]['ma'] = ma_list

        tau_df=pd.DataFrame(data=alltau[0:,0:],
                index=[i for i in range(alltau.shape[0])],
                columns=[i for i in range(alltau.shape[1])])
        tau_df.insert(loc=0, column='subj', value=onlysubj_half)
        tau_df.insert(loc=1, column='sess', value=onlysess_half)

        allrho1_df=pd.DataFrame(data=allrho1[0:,0:],
                index=[i for i in range(allrho1.shape[0])],
                columns=[i for i in range(allrho1.shape[1])])
        allrho1_df.insert(loc=0, column='subj', value=onlysubj_half)
        allrho1_df.insert(loc=1, column='sess', value=onlysess_half)

        allrho2_df=pd.DataFrame(data=allrho2[0:,0:],
                index=[i for i in range(allrho2.shape[0])],
                columns=[i for i in range(allrho2.shape[1])])
        allrho2_df.insert(loc=0, column='subj', value=onlysubj_half)
        allrho2_df.insert(loc=1, column='sess', value=onlysess_half)
        
        tau_df.to_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/estimatedtau_dhcp_{}_lowmovement_halfsample.csv'.format(model))
        np.savetxt('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/estimatedtau_dhcp_{}_lowmovement_halfsample.txt'.format(model),alltau)

        allrho1_df.to_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/estimatedrho1_dhcp_{}_lowmovement_halfsample.csv'.format(model))
        np.savetxt('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/estimatedrho1_dhcp_{}_lowmovement_halfsample.txt'.format(model),allrho1)

        allrho2_df.to_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/estimatedrho2_dhcp_{}_lowmovement_halfsample.csv'.format(model))
        np.savetxt('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/estimatedrho2_dhcp_{}_lowmovement_halfsample.txt'.format(model),allrho2)

        with open('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/params_dhcp_{}_lowmovement_halfsample.pickle'.format(model),'wb') as handle:
                pickle.dump(modeloutput, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    root_pth = '/dhcp/fmri_anna_graham/timecourses_intermediate'
    max_lag = 10
    main(root_pth,max_lag)
