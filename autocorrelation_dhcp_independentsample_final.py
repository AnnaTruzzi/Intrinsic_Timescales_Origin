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
    models_list = [[1,0,1]]
    nlags=10
    xdata=np.arange(nlags)
    ind_sample = []
    subj_set = pd.read_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample/half_and_replication_subj.csv')
    for item in subj_set['rep']:
        ind_sample.append([item.split("'")[1],item.split("'")[3]])
    onlysubj_ind = [a[0] for a in ind_sample]
    onlysess_ind = [a[1] for a in ind_sample]
    for model in models_list:
        print(model)
        alltau = np.zeros((len(ind_sample),400))
        allrho1 = np.zeros((len(ind_sample),400))
        allrho2 = np.zeros((len(ind_sample),400))
        modeloutput = {}
        for i,subj in enumerate(ind_sample):
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

            print(alltau.shape)
            modeloutput[str(subj)]['ar'] = ar_list
            if model[2]>0:
                modeloutput[str(subj)]['ma'] = ma_list

        tau_df=pd.DataFrame(data=alltau[0:,0:],
                index=[i for i in range(alltau.shape[0])],
                columns=[i for i in range(alltau.shape[1])])
        tau_df.insert(loc=0, column='subj', value=onlysubj_ind)
        tau_df.insert(loc=1, column='sess', value=onlysess_ind)

        allrho1_df=pd.DataFrame(data=allrho1[0:,0:],
                index=[i for i in range(allrho1.shape[0])],
                columns=[i for i in range(allrho1.shape[1])])
        allrho1_df.insert(loc=0, column='subj', value=onlysubj_ind)
        allrho1_df.insert(loc=1, column='sess', value=onlysess_ind)

        allrho2_df=pd.DataFrame(data=allrho2[0:,0:],
                index=[i for i in range(allrho2.shape[0])],
                columns=[i for i in range(allrho2.shape[1])])
        allrho2_df.insert(loc=0, column='subj', value=onlysubj_ind)
        allrho2_df.insert(loc=1, column='sess', value=onlysess_ind)
        
        tau_df.to_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample/estimatedtau_dhcp_{}_lowmovement_independentsample.csv'.format(model))
        np.savetxt('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample/estimatedtau_dhcp_{}_lowmovement_independentsample.txt'.format(model),alltau)

        allrho1_df.to_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample/estimatedrho1_dhcp_{}_lowmovement_independentsample.csv'.format(model))
        np.savetxt('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample/estimatedrho1_dhcp_{}_lowmovement_independentsample.txt'.format(model),allrho1)

        allrho2_df.to_csv('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample/estimatedrho2_dhcp_{}_lowmovement_independentsample.csv'.format(model))
        np.savetxt('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample/estimatedrho2_dhcp_{}_lowmovement_independentsample.txt'.format(model),allrho2)

        with open('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample/params_dhcp_{}_lowmovement_independentsample.pickle'.format(model),'wb') as handle:
                pickle.dump(modeloutput, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    root_pth = '/dhcp/fmri_anna_graham/timecourses_intermediate'
    max_lag = 10
    main(root_pth,max_lag)