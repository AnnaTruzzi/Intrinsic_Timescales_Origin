import nibabel as nib
import numpy as np 
import os
from scipy import optimize
import pandas as pd
import nilearn.signal as nil
from scipy import signal

def get_SNR(timecourse):
    #detrend = nil.clean(timecourse,standardize=False) + np.mean(timecourse, axis=0)
    mean = np.mean(timecourse,axis=0)
    std = np.std(timecourse,axis=0)
    snr = mean/std
    return snr,mean,std


def autocorr_decay(dk,A,tau,B):
    return A*(np.exp(-(dk/tau))+B)


def run_tau_estimation(group,subj_list,flag,**kwargs):
    roi=400
    alltau = np.zeros((len(subj_list),roi))
    allSNR = np.zeros((len(subj_list),roi))
    allMEAN = np.zeros((len(subj_list),roi))
    allSTD = np.zeros((len(subj_list),roi))
    nlags = 100
    removelag0 = 1
    if 'dhcp' in group:
        root_pth = '/dhcp/fmri_anna_graham/timecourses'
    else:
        root_pth = '/dhcp/fmri_anna_graham/hcp_timecourses/Schaefer/'

    for i,sub in enumerate(subj_list):
        print(sub)
        if 'dhcp' in group:
            sub_id = sub.split('\'')[1]
            sess_id = sub.split('\'')[3]
            filename = os.path.join(root_pth,f'{sub_id}/{sess_id}/{sub_id}_{sess_id}_task-rest_bold_7net.txt')
        else:
            filename = os.path.join(root_pth,f'{sub}_timeseries_volumetric_perROI_7net.txt')
        ts_df = np.loadtxt(filename)

        if 'dhcp' in group and flag=='drop_scan_dhcp':
            ts_df=signal.decimate(ts_df,2,axis=0)

        if 'dhcp' in group and flag=='low_pass_filter':
            ts_df = signal.decimate(ts_df,2,axis=0)

        if 'dhcp' in group and flag=='term_only':
            ts_df=signal.decimate(ts_df,2,axis=0)

        if flag=='global_signal':
            if 'dhcp' in group:
                ts_df=signal.decimate(ts_df,2,axis=0)
            brain_average=np.nanmean(ts_df,axis=1)
            brain_average=np.reshape(brain_average,(brain_average.shape[0],-1))
            ts_df= ts_df-brain_average

        allSNR[i,:],allMEAN[i:],allSTD[i,:]=get_SNR(ts_df)

        xdata=np.arange(nlags)

        autocorr_values = np.zeros((ts_df.shape[1], nlags))
        for ROI in range(0,ts_df.shape[1]):
            xc=ts_df[:,ROI]-np.mean(ts_df[:,ROI])
            fullcorr=np.correlate(xc, xc, mode='full')
            fullcorr=fullcorr / np.max(fullcorr)
            start=len(fullcorr) // 2
            stop=start+nlags
            autocorr_values[ROI,:]=fullcorr[start:stop]        

        for ROI in range(0,ts_df.shape[1]):
            try:
                A, tau, B = optimize.curve_fit(autocorr_decay,xdata[removelag0:],autocorr_values[ROI,removelag0:],p0=[0,np.random.rand(1)[0]+0.01,0],bounds=(([0,0,-np.inf],[np.inf,np.inf,np.inf])),method='trf',maxfev=1000)[0]
                alltau[i,ROI] = tau
            except:
                alltau[i,ROI] = np.nan

    print(alltau.shape)
    print(allSNR.shape)
    np.savetxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_{group}_7net_{flag}.txt',alltau)
    np.savetxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/SNR_estimation_{group}_7net_{flag}.txt',allSNR)
    np.savetxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/MEAN_estimation_{group}_7net_{flag}.txt',allMEAN)
    np.savetxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/STD_estimation_{group}_7net_{flag}.txt',allSTD)

    tau_df=pd.DataFrame(data=alltau[0:,0:],
            index=[i for i in range(alltau.shape[0])],
            columns=[i for i in range(alltau.shape[1])])
    if 'dhcp' in group:
        tau_df.insert(loc=0, column='subj', value=[item.split('\'')[1] for item in subj_list])
        tau_df.insert(loc=1, column='sess', value=[item.split('\'')[3] for item in subj_list])
    else:
        tau_df.insert(loc=0, column='subj', value=subj_list)
    tau_df.to_csv(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_{group}_7net_{flag}.csv')
