import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from scipy.stats import spearmanr

def get_corr(snr,tau1,tau2,title_flag,outname):
    r1,p1 = spearmanr(snr, tau1)
    r2,p2 = spearmanr(snr, tau2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,5))
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    ax[0].scatter(snr,tau1,alpha=0.5,s=10)
    ax[0].set_ylim((10,35))
    ax[0].set_xlabel('SNR per ROI',fontsize = 12)
    ax[0].set_ylabel('Median Tau per ROI (seconds)',fontsize = 12)
    ax[0].set_title(f'SNR vs Tau dHCP-Group 1 - {title_flag} \n r = {round(r1,4)}, p = {round(p1,4)}',fontsize = 12)
    ax[1].scatter(snr,tau2,alpha=0.5,s=10)
    ax[1].set_ylim((10,35))
    ax[1].set_xlabel('SNR per ROI',fontsize = 12)
    ax[1].set_ylabel('Median Tau per ROI (seconds)', fontsize = 12)
    ax[1].set_title(f'SNR vs Tau dHCP-Group 2 - {title_flag} \n r = {round(r2,4)}, p = {round(p2,4)}',fontsize = 12)
    plt.savefig(os.path.join(rootpth,f'Docs\\figures\\{outname}.pdf'), dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    rootpth = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\'
    snr = np.loadtxt(os.path.join(rootpth,'Data\\sub-CC00058XX09_ses-11300_preproc_bold-snr-mean_individualspace.txt'))
    
    tau1 = np.loadtxt(os.path.join(rootpth,'Results\\dhcp_intermediate\\dhcp_model_selection_halfsample\\estimatedtau_dhcp_[1, 0, 1]_lowmovement_halfsample.txt'))
    p95_tau1 = np.nanpercentile(tau1,95)
    tau1[np.where(tau1>p95_tau1)] = np.nan
    tau1[np.where(tau1<0)] = np.nan
    tau1_mean = np.nanmean(tau1,axis=0) * 0.392        
    
    tau2 = np.loadtxt(os.path.join(rootpth,'Results\\dhcp_intermediate\\dhcp_independent_sample\\estimatedtau_dhcp_[1, 0, 1]_lowmovement_independentsample.txt'))
    p95_tau2 = np.nanpercentile(tau2,95)
    tau2[np.where(tau2>p95_tau2)] = np.nan
    tau2[np.where(tau2<0)] = np.nan
    tau2_mean = np.nanmean(tau2,axis=0) * 0.392         

    get_corr(snr,tau1_mean,tau2_mean,'','Tau_SNR_correlation')
    highsnr_file = np.loadtxt('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\highSNR_ROIs_idx.txt')
    highsnr_idx = [int(i) for i in highsnr_file]
    snr_high = snr[highsnr_idx]
    tau1_high = tau1_mean[highsnr_idx]
    tau2_high = tau2_mean[highsnr_idx]
    get_corr(snr_high,tau1_high,tau2_high,'high SNR','Tau_SNR_correlation_highSNRonly')

    