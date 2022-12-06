import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import optimize
from unimodal_vs_transmodal import unimodal_vs_transmodal
import tau_estimation
from tau_estimation import run_tau_estimation
import brain_renders
from scipy.stats import spearmanr
import subprocess
#import rpy2.robjects as ro
import correlations
import pickle  
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import sem

def plot_distribution(data,outname):
    sns.distplot(data)
    plt.savefig(outname)
    plt.close()

def corr_with_snr(x,y):
    r,p = spearmanr(x, y)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (12,5))
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    ax.scatter(x,y,alpha=0.5, color='#014182')
    #ax.set_ylim((10,35))
    ax.set_xlabel('Mean SNR per ROI',fontsize = 12)
    ax.set_ylabel('Mean Tau per ROI (seconds)',fontsize = 12)
    ax.set_title(f'SNR vs Tau {group} \n r = {round(r,4)}, p = {round(p,4)}',fontsize = 12)
    plt.savefig(os.path.join(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/tau_corr_snr_{group}.png'), dpi=300)
    plt.savefig(os.path.join(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/tau_corr_snr_{group}.pdf'), dpi=300)
    plt.show()
    plt.close()


def get_net_dict():
    network_file7 = pd.read_csv('/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/Schaefer2018_400Parcels_7Networks_order.txt',sep = '\t', header = None)
    roi_names_all = np.array(network_file7[1])
    net_list = np.unique(np.array([i.split('_')[2][0:3] for i in roi_names_all]))
    myorder = [6,5,3,0,1,2,4]
    net_list = [net_list[i] for i in myorder]
    print(net_list)
    net_dict = {}
    for netnum,net in enumerate(net_list):
        net_dict[net] = []
        for i,roi in enumerate(roi_names_all):
            if net in roi:
                net_dict[net].append(i)    
    return net_dict

#def friedman_test(data):
#    r=ro.r
#    r.source('friedman_test.r')
#    p=r.rtest(data)
#    return p


groups_list = ['dhcp_group1','dhcp_group2','hcp']

net_dict = get_net_dict()


run_within_analysis = False
run_tau_estimation_analysis = False
run_brainrenders = False
run_between_analysis = True



###########################
### Within group analysis #
###########################
if run_within_analysis:

    for group in groups_list:
        if 'dhcp' in group:
            TR = 0.392
        else:
            TR = 0.72

        subj_file = pd.read_csv(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/{group}_subj_list.csv')
        subj_list = list(subj_file.iloc[:,0])
        dual_sess_subj = list(pd.read_csv(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/48preterm.csv').iloc[:,0])
        dual_sess_subj = ['sub-'+subj for subj in dual_sess_subj]

        if 'dhcp' in group:
            #onlysubj = [subj.split('\'')[1] for subj in subj_list]
            #dual_sess_subj = set([x for x in onlysubj if onlysubj.count(x) > 1])
            subj_list = [subj for subj in subj_list if subj.split('\'')[1] not in dual_sess_subj]
        
        if run_tau_estimation_analysis:
            run_tau_estimation(group, subj_list)

        # load tau file
        tau = np.loadtxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_{group}_7net.txt')
        
        # remove outliers and multiply by TR
        p95_tau = np.nanpercentile(tau,95)
        tau[np.where(tau>p95_tau)] = np.nan
        tau = tau * TR

        highnan_idx = []
        for i,row in enumerate(range(0,tau.shape[0])):
            tau_row = tau[row]
            nan_percentage = (np.count_nonzero(np.isnan(tau_row))/tau_row.shape[0])*100
            if nan_percentage>50:
                highnan_idx.append(i)
        tau = np.delete(tau,highnan_idx,axis=0)
        np.savetxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_in_seconds_{group}_7net.txt',tau)

        # plot distribution
        plot_distribution(tau,f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/tau_distribution_{group}_7net.png')
        plt.style.use('classic')
        im = plt.imshow(tau)
        plt.colorbar(im)
        plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/tau_2Ddist_{group}_7net.png')
        plt.close()

        tau_mean = np.nanmean(tau,axis=0)
        np.savetxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_ROImean_{group}_7net.txt',tau_mean)
        # check correlation to snr and plot


        ## SNR analysis
        snr = np.loadtxt('/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/SNR_estimation_dhcp_group1_7net.txt')
        mean = np.loadtxt('/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/MEAN_estimation_dhcp_group1_7net.txt')
        std = np.loadtxt('/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/STD_estimation_dhcp_group1_7net.txt')
        snr_mean=np.mean(snr,axis=0)
        net_dict = get_net_dict()        
        if 'group1' in group:
            corr_with_snr(snr_mean,tau_mean)
        
        # get data for brain rendering
        if run_brainrenders:
            brain_renders.brainrenders(group,tau_mean,net_dict)


        ### barplots for single networks
        net_name_list_plot=[]
        tau_bynet_plot = []
        for net in net_dict.keys():
            net_name_list_plot.extend(np.repeat(net,len(net_dict[net])))
            tau_bynet_plot.extend(tau_mean[net_dict[net]])
        plot_bynet_dict = {'net_name':net_name_list_plot,'tau':tau_bynet_plot}
        plot_bynet_df = pd.DataFrame(plot_bynet_dict)

        colors=['#9e43a2','#6241c7','#019529','#9cef43','#ffa62b','#ff6cb5','#ffffff']
        labels=['Visual','Somatomotor','Limbic','Fronto-Parietal','Default','Dorsal attention','Ventral attention']
        custompalette = sns.set_palette(sns.color_palette(colors))
        sns.set_context(rc = {'patch.linewidth': 0.5})
        g1 = sns.barplot(data=plot_bynet_df, x='net_name', y='tau', capsize=.2,
                palette=custompalette, linestyle = "-", edgecolor = "black",ci=None,dodge=False)
        plt.ylim((0,8))
        g1.set(xlabel=None)
        g1.set(xticklabels=[])
        g1.tick_params(bottom=False)
        patches = []
        for i,label in enumerate(labels):
            patches.append(mpatches.Patch(color=colors[i], label=label))

        x_coords = [p.get_x() + 0.5*p.get_width() for p in g1.patches]
        y_coords = [p.get_height() for p in g1.patches]
        
        error_list = []
        for net in plot_bynet_df['net_name'].unique():
            error = sem(np.array(plot_bynet_df[plot_bynet_df['net_name']==net]['tau']))
            error_list.append(error)            

        plt.errorbar(x_coords, y_coords, fmt='none', yerr=error_list, c="black", elinewidth=3)
        #plt.legend(handles=patches,fontsize = 12)
        plt.tight_layout()
        plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/tau_bynet_{group}_7net.png')
        plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/tau_bynet_{group}_7net.pdf')
        plt.close()


        ##TODO: can we implement Friedman test here in python?


############################
### Between group analysis #
############################
if run_between_analysis:
    dhcp_group1 = np.loadtxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_in_seconds_dhcp_group1_7net.txt')
    dhcp_group2 = np.loadtxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_in_seconds_dhcp_group2_7net.txt')
    hcp = np.loadtxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_in_seconds_hcp_7net.txt')
    snr = np.loadtxt('/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/SNR_estimation_dhcp_group1.txt')

    dhcp_group1_mean = np.loadtxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_ROImean_dhcp_group1_7net.txt')
    dhcp_group2_mean = np.loadtxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_ROImean_dhcp_group2_7net.txt')
    hcp_mean = np.loadtxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_estimation_ROImean_hcp_7net.txt')
    snr_mean=np.mean(snr,axis=0)

    correlations.run_and_plot_corr(dhcp_group1_mean,dhcp_group2_mean,'dhcp_group1','dhcp_group2',f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/corr_dhcp1_dhcp2_7net.png',xlim=(0,8),ylim=(0,8))
    correlations.run_and_plot_partial_corr(dhcp_group1_mean,dhcp_group2_mean,snr_mean,f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/partial_corr_dhcp1_dhcp2_snr_7net.csv')

    correlations.run_and_plot_corr(hcp_mean,dhcp_group1_mean,'hcp','dhcp_group1',f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/corr_dhcp1_hcp_7net.pdf',xlim=(0,6),ylim=(0,8))
    correlations.run_and_plot_corr(hcp_mean,dhcp_group2_mean,'hcp','dhcp_group2',f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/corr_dhcp2_hcp_7net.pdf',xlim=(0,6),ylim=(0,8))


    # Corr by net
    correlations.run_and_plot_corr_bynet(hcp_mean,dhcp_group1_mean,'hcp','dhcp_group1',net_dict,xlim=(0,6),ylim=(0,8))
    correlations.run_and_plot_corr_bynet(hcp_mean,dhcp_group2_mean,'hcp','dhcp_group2',net_dict,xlim=(0,6),ylim=(0,8))


    unimodal_index=[]
    transmodal_index=[]
    for key in net_dict.keys():
        if 'Vis' in key or 'Som' in key:
            unimodal_index.extend(net_dict[key])
        else:
            transmodal_index.extend(net_dict[key])

    unimodal_vs_transmodal(dhcp_group1,hcp,unimodal_index,transmodal_index,'dhcp_group1','hcp',flag='')
    unimodal_vs_transmodal(dhcp_group2,hcp,unimodal_index,transmodal_index,'dhcp_group2','hcp',flag='')

