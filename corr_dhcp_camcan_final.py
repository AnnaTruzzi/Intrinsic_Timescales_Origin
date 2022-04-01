from functools import partial
import pandas as pd
import numpy as np
import glob
import os
from scipy.optimize import curve_fit
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.regression import linear_model
from pingouin import partial_corr


def corr_with_camcan(root_camcan,root_dhcp,sample):
    dhcp_file = os.path.join(root_dhcp,"estimatedtau_dhcp_[1, 0, 1]_lowmovement_{}.txt".format(sample))
    dhcp = np.loadtxt(dhcp_file) * 0.392
    p95_dhcp = np.nanpercentile(dhcp,95)
    dhcp[np.where(dhcp>p95_dhcp)] = np.nan
    dhcp[np.where(dhcp<0)] = np.nan
    dhcp_mean = np.nanmean(dhcp,axis=0)  
    dhcp_nolimbic = dhcp_mean[excludelimbic_idx]
    camcan_file = os.path.join(root_camcan,"estimatedtau_camcan_Rest_[1, 0, 1]_lowmovement.txt")
    camcan = np.loadtxt(camcan_file) * 1.97
    p95_camcan = np.nanpercentile(camcan,95)
    camcan[np.where(camcan>p95_camcan)] = np.nan
    camcan[np.where(camcan<0)] = np.nan
    camcan_mean = np.nanmean(camcan,axis=0)  
    camcan_nolimbic = camcan_mean[excludelimbic_idx]
    r,p = spearmanr(dhcp_nolimbic, camcan_nolimbic)
    plt.scatter(dhcp_nolimbi, camcan_nolimbic,alpha=0.5,s=10)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.ylabel('CamCAN - Tau (seconds)', fontsize = 12)
    plt.xlim((5,35))
    if 'half' in sample:
        plt.xlabel('dHCP Group 1 - Tau (seconds)', fontsize = 12)
        plt.suptitle('CamCAN lowmovement and dHCP Group 1 - No Limbic\n r = {}, p = {}'.format(round(r,2),round(p,5)))
    else:
        plt.xlabel('dHCP Group 2 - Tau (seconds)', fontsize = 12)
        plt.suptitle('CamCAN lowmovement and dHCP Group 2 - No Limbic\n r = {}, p = {}'.format(round(r,4),round(p,5)))
    if os.getlogin() == 'Anna':
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\corr_camcanlowmovement_dhcp_{}_nolimbic.pdf'.format(sample))
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\corr_camcanlowmovement_dhcp_{}_nolimbic.png'.format(sample))
    else:
        plt.savefig('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Docs/figures/corr_camcanlowmovement_dhcp_{}_nolimbic.pdf'.format(sample))
    #plt.show()
    plt.close()


def corr_dhcp(dhcp_group1, dhcp_group2,flag):
    r,p = spearmanr(dhcp_group1, dhcp_group2)
    plt.scatter(dhcp_group1, dhcp_group2,alpha=0.5,s=10)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.xlabel('dHCP Group1 - Tau (seconds)', fontsize = 12)
    plt.ylabel('dHCP Group2 - Tau (seconds)', fontsize = 12)
    plt.ylim((10,35))    
    plt.xlim((10,35))    
    plt.suptitle('dHCP Group1 and dHCP Group 2 - {} \n r = {}, p = {}'.format(flag,round(r,2),round(p,5)))
    if os.getlogin() == 'Anna':
        plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\corr__dhcp_group1AND2_{flag}.pdf')
        plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\corr__dhcp_group1AND2_{flag}.png')
    else:
        plt.savefig(f'/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Docs/figures/corr__dhcp_group1AND2_{flag}.pdf')
    #plt.show()
    plt.close()


def partialcorr_dhcp(dhcp_group1, dhcp_group2,snr):
    cor_dict = {'Group1': dhcp_group1,
                'Group2': dhcp_group2,
                'snr':snr}
    cor_df = pd.DataFrame(cor_dict)
    res = partial_corr(data=cor_df, x='Group1', y='Group2', covar='snr', method='spearman')
    if os.getlogin() == 'Anna':
        res.to_csv(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\corr_dhcp_group1AND2_partialcorr.txt',sep='\t')
    else:
        res.to_csv(f'/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Docs/corr_dhcp_group1AND2_partialcorr.txt',sep='\t')



if __name__ == '__main__':
    sample_list = ['independentsample','halfsample']
    rootpth = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\'
    
    ### Files and indexes for analysis with only high snr ROIs.
    highsnr_file = np.loadtxt('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\highSNR_ROIs_idx.txt')
    highsnr_idx = [int(i) for i in highsnr_file]
    snr = np.loadtxt(os.path.join(rootpth,'Data\\sub-CC00058XX09_ses-11300_preproc_bold-snr-mean_individualspace.txt'))
    
    ### Files and indexes for analysis without limbic system
    network_file = pd.read_csv('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Data\\Schaefer2018_400Parcels_7Networks_order.txt',sep = '\t', header = None)
    limbic_idx = np.array([i for i,roi in enumerate(network_file[1]) if 'Limbic' in roi])
    excludelimbic_idx = [i for i,roi in enumerate(network_file[1]) if i not in limbic_idx]

    if os.getlogin() == 'Anna':
        root_camcan = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\camcan'
    else:
        root_camcan = '/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/camcan/'
    for sample in sample_list:
        if 'half' in sample:
            if os.getlogin() == 'Anna':
                root_dhcp = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_model_selection_halfsample'
            else:
                root_dhcp = '/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample'
            dhcp_group1_file = os.path.join(root_dhcp,"estimatedtau_dhcp_[1, 0, 1]_lowmovement_{}.txt".format(sample))
            dhcp_group1_all = np.loadtxt(dhcp_group1_file) * 0.392
            p95_dhcp1 = np.nanpercentile(dhcp_group1_all,95)
            dhcp_group1_all[np.where(dhcp_group1_all>p95_dhcp1)] = np.nan
            dhcp_group1_all[np.where(dhcp_group1_all<0)] = np.nan
            dhcp_group1_all_mean = np.nanmean(dhcp_group1_all,axis=0)          
            dhcp_group1_nolimbic = dhcp_group1_all_mean[excludelimbic_idx]
        else:
            if os.getlogin() == 'Anna':
                root_dhcp = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_independent_sample'
            else:
                root_dhcp = '/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample'
            dhcp_group2_file = os.path.join(root_dhcp,"estimatedtau_dhcp_[1, 0, 1]_lowmovement_{}.txt".format(sample))
            dhcp_group2_all = np.loadtxt(dhcp_group2_file) * 0.392
            p95_dhcp2 = np.nanpercentile(dhcp_group2_all,95)
            dhcp_group2_all[np.where(dhcp_group2_all>p95_dhcp2)] = np.nan
            dhcp_group2_all[np.where(dhcp_group2_all<0)] = np.nan
            dhcp_group2_all_mean = np.nanmean(dhcp_group2_all,axis=0)  
            dhcp_group2_nolimbic = dhcp_group2_all_mean[excludelimbic_idx]
        corr_with_camcan(root_camcan,root_dhcp,sample)
    corr_dhcp(dhcp_group1_nolimbic, dhcp_group2_nolimbic, flag = 'nolimbic')
    partialcorr_dhcp(dhcp_group1_all_mean, dhcp_group2_all_mean,snr)