import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import pickle
import pandas as pd
import glob
import os
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import statsmodels.api as sm
from statsmodels.formula.api import glm
import nibabel as nib
from statsmodels.regression import linear_model
from scipy.stats import shapiro
from scipy.stats import wilcoxon
from scipy.stats import sem

# function to calculate Cohen's d for independent samples
# from: https://machinelearningmastery.com/effect-size-measures-in-python/
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s



def roi_render(index_list,atlas,outvolume_size,outname):
    rois_data = atlas.get_fdata()
    outvolume = outvolume_size
    outvolume[:,:,:] = 0
    for roi in index_list:
        roi_index = np.where(rois_data==roi+1)
        outvolume[roi_index] = 1
    outimage = nib.Nifti1Image(outvolume, affine=atlas.affine)
    nib.save(outimage,outname)
    outimage.uncache()


def anova_and_plot(root_dhcp, root_camcan,sample,label,unimodal_index,transmodal_index):
    print(sample)
    dhcp_file = os.path.join(root_dhcp,"estimatedtau_dhcp_[1, 0, 1]_lowmovement_{}.txt".format(sample))
    dhcp_all = np.loadtxt(dhcp_file) * 0.392
    p95_dhcp = np.nanpercentile(dhcp_all,95)
    dhcp_all[np.where(dhcp_all>p95_dhcp)] = np.nan
    dhcp_all[np.where(dhcp_all<0)] = np.nan
    dhcp_unimodal = np.nanmean(dhcp_all[:,np.array(unimodal_index)],axis=1)
    dhcp_transmodal = np.nanmean(dhcp_all[:,np.array(transmodal_index)],axis=1)
    camcan_file = os.path.join(root_camcan,"estimatedtau_camcan_Rest_[1, 0, 1]_lowmovement.txt")
    camcan_all = np.loadtxt(camcan_file) * 1.97
    p95_camcan = np.nanpercentile(camcan_all,95)
    camcan_all[np.where(camcan_all>p95_camcan)] = np.nan
    camcan_all[np.where(camcan_all<0)] = np.nan
    camcan_unimodal = np.nanmean(camcan_all[:,np.array(unimodal_index)],axis=1)
    camcan_transmodal = np.nanmean(camcan_all[:,np.array(transmodal_index)],axis=1)
    network_labels = np.concatenate((np.repeat('Unimodal',dhcp_unimodal.shape[0]),np.repeat('Transmodal',dhcp_transmodal.shape[0]),np.repeat('Unimodal',camcan_unimodal.shape[0]),np.repeat('Transmodal',camcan_transmodal.shape[0])),axis=None)
    group_labels = np.concatenate((np.repeat(f'dHCP {label}',dhcp_unimodal.shape[0]+dhcp_transmodal.shape[0]),np.repeat('CamCAN',camcan_unimodal.shape[0]+camcan_transmodal.shape[0])),axis=None)
    net_dict = {'Network_type':network_labels,
                'Group_type': group_labels,
                'Tau': np.concatenate((dhcp_unimodal,dhcp_transmodal,camcan_unimodal,camcan_transmodal),axis=None)}
    net_db = pd.DataFrame(net_dict)


    sns.histplot(dhcp_unimodal)
    s,p= shapiro(dhcp_unimodal)
    plt.suptitle(f'dhcp unimodal - {label} \n Shapiro: s = {s}, p {p}')
    #plt.show()
    plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\distribution_dhcp_unimodal_{label}_highSNRonly.png')

    sns.histplot(dhcp_transmodal)
    s,p= shapiro(dhcp_transmodal)
    plt.suptitle(f'dhcp transmodal - {label} \n Shapiro: s = {s}, p {p}')
    #plt.show()
    plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\distribution_dhcp_transmodal_{label}_highSNR_only.png')

    sns.histplot(camcan_transmodal)
    s,p= shapiro(camcan_transmodal)
    plt.suptitle(f'camcan transmodal \n Shapiro: s = {s}, p {p}')
    #plt.show()
    plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\distribution_camcan_transmodal_highSNRonly.png')

    sns.histplot(camcan_unimodal)
    s,p= shapiro(camcan_unimodal)
    plt.suptitle(f'camcan unimodal \n Shapiro: s = {s}, p {p}')
    #plt.show()
    plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\distribution_camcan_unimodal_highSNRonly.png')

    with open(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\unimodalVStransmodal_analysisResults_{label}.txt', 'w') as f:
        statistic,p = kruskal(dhcp_unimodal,dhcp_transmodal,camcan_unimodal,camcan_transmodal)
        print(statistic,p)
        f.write(f'Results for CamCAN vs {label} \n')
        f.write(f'\n Kruskal test: chi = {statistic}, p = {round(p,4)} \n')

        ###### ACROSS GROUPS - WITHIN NETWORKS
        t_unimodal,p_unimodal = mannwhitneyu(dhcp_unimodal,camcan_unimodal)
        d_unimodal = cohend(dhcp_unimodal,camcan_unimodal)
        print('UNIMODAL - t = {}, p = {}, d = {}'.format(t_unimodal,p_unimodal,d_unimodal))
        f.write('###### \n')
        f.write(f'ACROSS GROUPS - WITHIN NETWORKS \n')
        f.write(f'UNIMODAL - t = {t_unimodal}, p = {p_unimodal}, d = {d_unimodal} \n')

        t_transmodal,p_transmodal = mannwhitneyu(dhcp_transmodal,camcan_transmodal)
        d_transmodal = cohend(dhcp_transmodal,camcan_transmodal)
        print('TRANSMODAL - t = {}, p = {}, d = {}'.format(t_transmodal,p_transmodal,d_transmodal))
        f.write(f'TRANSMODAL - t = {t_transmodal}, p = {p_transmodal}, d = {d_transmodal} \n')

        ###### WITHIN GROUPS - ACROSS NETWORKS
        t_dhcp,p_dhcp = wilcoxon(dhcp_unimodal,dhcp_transmodal)
        d_dhcp = cohend(dhcp_unimodal,dhcp_transmodal)
        print('dHCP - t = {}, p = {}, d = {}'.format(t_dhcp,p_dhcp,d_dhcp))
        f.write('###### \n')
        f.write(f'WITHIN GROUPS - dHCP \n')
        f.write(f'dHCP - t = {t_dhcp}, p = {p_dhcp}, d = {d_dhcp} \n')

        t_camcan,p_camcan = wilcoxon(camcan_unimodal,camcan_transmodal)
        d_camcan = cohend(camcan_unimodal,camcan_transmodal)
        print('CamCAN - t = {}, p = {}, d = {}'.format(t_camcan,p_camcan,d_camcan))
        f.write(f'CamCAN  - t = {t_camcan}, p = {p_camcan}, d = {d_camcan} \n')

    sns.distplot(np.concatenate((dhcp_unimodal,dhcp_transmodal,camcan_unimodal,camcan_transmodal),axis=None))
    plt.suptitle('All Tau distribution - high SNR')
    if os.getlogin() == 'Anna':
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\unimodal_transmodal_taudistribution_bothsamples_lowmovement_highSNRonly.pdf')
    else:
        plt.savefig('/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Docs/figures/unimodal_transmodal_taudistribution_bothsamples_lowmovement_highSNRonly.pdf')
    plt.close()

    costum_palette = [sns.xkcd_rgb['lightblue'],sns.xkcd_rgb["burnt orange"]]
    sns.set_palette(costum_palette)
    g1 = sns.barplot(x='Group_type', y='Tau', hue='Network_type', data = net_dict, ci=None)
    x_coords = [p.get_x() + 0.5*p.get_width() for p in g1.patches]
    y_coords = [p.get_height() for p in g1.patches]
    plt.errorbar(x_coords, y_coords, fmt='none', yerr=[sem((dhcp_unimodal-dhcp_transmodal)/2),sem((dhcp_unimodal-dhcp_transmodal)/2),sem((camcan_unimodal-camcan_transmodal)/2),sem((camcan_unimodal-camcan_transmodal)/2)], c="black", elinewidth=2)
    plt.ylim((0,35))
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 15)
    plt.ylabel('Tau (in seconds)', fontsize = 15)
    ax = g1.axes
    # unimodal dhcp vs camcan
    ax.plot([-0.2,-0.2, 0.8,0.8], [29.9,30,30,29.9], lw=1.5, color = 'black')
    ax.text((-0.2+0.8)*.5, 30, "***", ha='center', va='bottom',fontsize = 20)    
    ax.plot([0.2,0.2, 1.2,1.2], [25.9,26,26,25.9], lw=1.5, color = 'black')
    ax.text((0.2+1.2)*.5, 26, "***", ha='center', va='bottom',fontsize = 20)    
    # unimodal vs transmodal camcan
    ax.plot([0.8,0.8, 1.2,1.2], [19.9,20,20,19.9], lw=1.5, color = 'black')
    ax.text((0.8+1.2)*.5, 20, "**", ha='center', va='bottom',fontsize = 20)    

    plt.suptitle(f'dHCP {label} vs CamCAN - no limbic')
    if os.getlogin() == 'Anna':
        plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\transmodalVSunimodal_finalised_barplot_{label}_nolimbic.pdf')
        plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\transmodalVSunimodal_finalised_barplot_{label}_nolimbic.png')
    else:
        plt.savefig(f'/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Docs/figures/transmodalVSunimodal_finalised_barplot_{label}_nolimbic.pdf')
        plt.savefig(f'/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Docs/figures/transmodalVSunimodal_finalised_barplot_{label}_nolimbic.png')
    plt.show()
    plt.close()
    a=1



if __name__ == '__main__':
    label_list = ['Group1','Group2']
    rootpth = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\'
    highsnr_file = np.loadtxt('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\highSNR_ROIs_idx.txt')
    highsnr_idx = [int(i) for i in highsnr_file]
    lowsnr_file = np.loadtxt('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\lowSNR_ROIs_idx.txt')
    lowsnr_idx = [int(i)+1 for i in lowsnr_file]

    ### Files and indexes for analysis without limbic system
    network_file = pd.read_csv('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Data\\Schaefer2018_400Parcels_17Networks_order.txt',sep = '\t', header = None)
    limbic_idx = np.array([i for i,roi in enumerate(network_file[1]) if 'Limbic' in roi])
    with open('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Data\\roi_by_netclass.pickle','rb') as f:
        network_file_Ito = pickle.load(f)    
    excludelimbic_idx = [roi-1 for i,roi in enumerate(network_file_Ito['transmodal']+network_file_Ito['unimodal']) if roi-1 not in limbic_idx]

    roi_names_all = np.array(network_file[1])
    lowsnr_names = roi_names_all[lowsnr_idx]

    unimodal_index = [i-1 for i in network_file_Ito['unimodal']]
    transmodal_index = [i-1 for i in network_file_Ito['transmodal']]

    unimodal_index = [i for i in unimodal_index if i in excludelimbic_idx]
    transmodal_index = [i for i in transmodal_index if i in excludelimbic_idx]

    atlas_dhcp = nib.load('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Data\\schaefer_40weeks.nii.gz')
    outvolume_size_dhcp = np.zeros((202, 274, 217))
    atlas_camcan  = nib.load('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Data\\Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm_resliced.nii.gz')
    outvolume_size_camcan = np.zeros((61, 73, 61))
    roi_render(unimodal_index,atlas_dhcp,outvolume_size_dhcp,"C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_unimodal_render_nolimbic.nii.gz")
    roi_render(unimodal_index,atlas_camcan, outvolume_size_camcan,"C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\camcan\\camcan_unimodal_render_nolimbic.nii.gz")
    roi_render(transmodal_index,atlas_dhcp,outvolume_size_dhcp,"C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_transmodal_render_nolimbic.nii.gz")
    roi_render(transmodal_index,atlas_camcan, outvolume_size_camcan,"C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\camcan\\camcan_transmodal_render_nolimbic.nii.gz")
    roi_render(lowsnr_idx,atlas_camcan, outvolume_size_camcan,"C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\camcan\\camcan_lowSNR_render.nii.gz")
    roi_render(lowsnr_idx,atlas_dhcp, outvolume_size_dhcp,"C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_lowSNR_render.nii.gz")


    for label in label_list:
        if 'Group1' in label:
            sample = 'halfsample'
            if os.getlogin() == 'Anna':
                root_dhcp = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_model_selection_halfsample'
            else:
                root_dhcp = '/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_model_selection_halfsample'
        else:
            sample = 'independentsample'
            if os.getlogin() == 'Anna':
                root_dhcp = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_independent_sample'
            else:
                root_dhcp = '/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/dhcp_intermediate/dhcp_independent_sample'
        
        if os.getlogin() == 'Anna':
            root_camcan = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\camcan\\'
        else:
            root_camcan = '/dhcp/fmri_anna_graham/MRIautocorr_ARMA/Results/camcan/'
        anova_and_plot(root_dhcp, root_camcan,sample,label,unimodal_index,transmodal_index)
