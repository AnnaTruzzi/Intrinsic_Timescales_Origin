from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats.stats import spearmanr
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import pickle
import pandas as pd
import glob
import os
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import zscore
from statsmodels.regression import linear_model
import matplotlib.patches as mpatches

def main_dhcp(root_pth,sample,network_names):
    dhcp_file = os.path.join(root_pth,"estimatedtau_dhcp_[1, 0, 1]_lowmovement_{}.txt".format(sample))
    dhcp = np.loadtxt(dhcp_file) * 0.392
    p95_dhcp = np.nanpercentile(dhcp,95)
    dhcp[np.where(dhcp>p95_dhcp)] = np.nan
    dhcp[np.where(dhcp<0)] = np.nan
    dhcp_mean = np.nanmean(dhcp,axis=0)  
    dhcp_mean_zscore = zscore(dhcp_mean)
    dhcp_network = []
    dhcp_network_zscore = []
    network_4_dict = []
    for network in network_names:
        network_index = []
        for i,roi in enumerate(roi_net_list):
                if network in roi:
                       network_index.append(i)
        dhcp_network.extend(dhcp_mean[np.array(network_index)])
        dhcp_network_zscore.extend(dhcp_mean_zscore[np.array(network_index)])
        network_4_dict.extend(np.repeat(network,len(network_index)))
    plot_dict = {'Network': network_4_dict,
                'Tau': dhcp_network,
                'Tau_zscore':dhcp_network_zscore}
    plot_db = pd.DataFrame(plot_dict)
    colors=['#F1F0F0','#4D3896','#EF5E9F','#68246A','#EB933B','#BAC733','#55A045','#2D6A58']
    labels=['Visual','Somatomotor','Temporoparietal','Ventral attention','Dorsal attention','Default','Control','Limbic']
    custompalette = sns.set_palette(sns.color_palette(colors))
    sns.set_context(rc = {'patch.linewidth': 0.5})
    g1=sns.barplot(x = 'Network', y = 'Tau', data = plot_db, capsize=.2,
                order = ['Vis','Som','Tem','Sal','Dor','Def','Con','Lim'],
                palette=custompalette, linestyle = "-", linewidth = 1, edgecolor = "black")
    plt.ylim((0,50))
    g1.set(xlabel=None)
    g1.set(xticklabels=[])
    g1.tick_params(bottom=False)
    patches = []
    for i,label in enumerate(labels):
        patches.append(mpatches.Patch(color=colors[i], label=label))
    plt.legend(handles=patches,fontsize = 12)
    plt.tight_layout()
    if 'half' in sample:
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_dhcp_Group1.pdf',dpi=300)
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_dhcp_Group1.png',dpi=300)
    else:
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_dhcp_Group2.pdf',dpi=300)
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_dhcp_Group2.png',dpi=300)
    #plt.show()
    plt.close()
    colors=['#F1F0F0','#4D3896','#EF5E9F','#68246A','#EB933B','#BAC733','#55A045','#2D6A58']
    labels=['Visual','Somatomotor','Temporoparietal','Ventral attention','Dorsal attention','Default','Control','Limbic']
    custompalette = sns.set_palette(sns.color_palette(colors))
    sns.barplot(x = 'Network', y = 'Tau_zscore', data = plot_db, capsize=.2,
                order = ['Vis','Som','Tem','Sal','Dor','Def','Con','Lim'],
                palette=custompalette, linestyle = "-", linewidth = 1, edgecolor = "black")
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    if 'half' in sample:
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_zscore_dhcp_Group1.png',dpi=300)
    else:
        plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_zscore_dhcp_Group2.png',dpi=300)
    #plt.show()
    plt.close()

    return plot_db

def main_camcan(root_pth,network_names):
    camcan_file = os.path.join(root_pth,"estimatedtau_camcan_Rest_[1, 0, 1]_lowmovement.txt")
    camcan = np.loadtxt(camcan_file) * 1.97
    p95_camcan = np.nanpercentile(camcan,95)
    camcan[np.where(camcan>p95_camcan)] = np.nan
    camcan[np.where(camcan<0)] = np.nan
    camcan_mean = np.nanmean(camcan,axis=0)  
    camcan_mean_zscore = zscore(camcan_mean)
    camcan_network = []
    camcan_network_zscore = []
    network_4_dict = []
    for network in network_names:
        network_index = []
        for i,roi in enumerate(roi_net_list):
            if network in roi:
                network_index.append(i)
        camcan_network.extend(camcan_mean[np.array(network_index)])
        camcan_network_zscore.extend(camcan_mean_zscore[np.array(network_index)])
        network_4_dict.extend(np.repeat(network,len(network_index)))
    plot_dict = {'Network': network_4_dict,
                'Tau': camcan_network,
                'Tau_zscore':camcan_network_zscore}
    plot_db = pd.DataFrame(plot_dict)
    colors=['#F1F0F0','#4D3896','#EF5E9F','#68246A','#EB933B','#BAC733','#55A045','#2D6A58']
    labels=['Visual','Somatomotor','Temporoparietal','Ventral attention','Dorsal attention','Default','Control','Limbic']
    custompalette = sns.set_palette(sns.color_palette(colors))
    g1 = sns.barplot(x = 'Network', y = 'Tau', data = plot_db, capsize=.2,
                order = ['Vis','Som','Tem','Sal','Dor','Def','Con','Lim'],
                palette=custompalette, linestyle = "-", linewidth = 1, edgecolor = "black")
    plt.ylim((0,50))
    g1.set(xlabel=None)
    g1.set(xticklabels=[])
    g1.tick_params(bottom=False)
    plt.setp(g1.patches, linewidth=1)
    plt.tight_layout()
    plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_camcan_lowmovement.pdf',dpi=300)
    plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_camcan_lowmovement.png',dpi=300)
    #plt.show()
    plt.close()
    colors=['#F1F0F0','#4D3896','#EF5E9F','#68246A','#EB933B','#BAC733','#55A045','#2D6A58']
    labels=['Visual','Somatomotor','Temporoparietal','Ventral attention','Dorsal attention','Default','Control','Limbic']
    custompalette = sns.set_palette(sns.color_palette(colors))
    sns.barplot(x = 'Network', y = 'Tau_zscore', data = plot_db, capsize=.2,
                order = ['Vis','Som','Tem','Sal','Dor','Def','Con','Lim'],
                palette=custompalette,linestyle = "-", linewidth = 1, edgecolor = "black")
    plt.xticks(rotation=90,fontsize=10)
    plt.tight_layout()
    plt.savefig('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_schaefer_networks_zscore_camcan_lowmovement.png',dpi=300)
    #plt.show()
    plt.close()

    return plot_db


def plot_correlations(db,networks_name,sample):
    for name in network_names:
        if 'half' in sample:
            group = 'Group1'
        elif 'independent' in sample:
            group = 'Group2'
        r,p = spearmanr(db.loc[(db['Group']=='dhcp') & (db['Network']==name)]['Tau'],db.loc[(db['Group']=='camcan') & (db['Network']==name)]['Tau'])
        N = len(db.loc[(db['Group']=='dhcp') & (db['Network']==name)]['Tau'])
        plt.scatter(db.loc[(db['Group']=='dhcp') & (db['Network']==name)]['Tau'],db.loc[(db['Group']=='camcan') & (db['Network']==name)]['Tau'],color='#007FFF')
        plt.xlabel(f'dhcp {name}',fontsize=12)
        plt.ylabel(f'camcan {name}',fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlim((10,32))
        plt.suptitle(f'{name} - camcan vs dhcp {group} - 8 Net \n r = {round(r,3)}, p = {round(p,8)}, N = {N}')
        plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_Net_dhcp{group}_camcan_{name}.pdf',dpi=300)
        plt.savefig(f'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Docs\\figures\\bynet\\8_Net_dhcp{group}_camcan_{name}.png',dpi=300)
        #plt.show()
        plt.close()


if __name__ == '__main__':
    rootpth = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\'
    snr = np.loadtxt(os.path.join(rootpth,'Data\\sub-CC00058XX09_ses-11300_preproc_bold-snr-mean_individualspace.txt'))
    snr = snr-np.mean(snr)
    sample_list = ['independentsample','halfsample']
    network_file = pd.read_csv('C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Data\\Schaefer2018_400Parcels_17Networks_order.txt',sep = '\t', header = None)
    roi_net_list = [i.split("_")[2][0:3] for i in network_file[1]]
    network_names = list(set(roi_net_list))
    
    root_camcan = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\camcan\\'
    camcan_db = main_camcan(root_camcan,network_names)
    camcan_db['Group'] = np.repeat('camcan',len(camcan_db['Tau']))
    for sample in sample_list:
        if 'half' in sample:
            root_pth = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_model_selection_halfsample'
        else:
            root_pth = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project dHCP_Autocorrelation\\MRIautocorr_ARMA\\Results\\dhcp_intermediate\\dhcp_independent_sample'
        dhcp_db = main_dhcp(root_pth,sample,network_names)
        dhcp_db['Group'] = np.repeat('dhcp',len(dhcp_db['Tau']))
        alldb = pd.concat([dhcp_db,camcan_db])
        plot_correlations(alldb,network_names,sample)

