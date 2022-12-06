import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.regression import linear_model
from scipy.stats import shapiro
from scipy.stats import wilcoxon
from scipy.stats import sem
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
import seaborn as sns

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


def unimodal_vs_transmodal(group1, group2, unimodal_idx, transmodal_idx, label1, label2,flag):
    group1_unimodal = np.nanmean(group1[:,np.array(unimodal_idx)],axis=1)
    group1_transmodal = np.nanmean(group1[:,np.array(transmodal_idx)],axis=1)
    group2_unimodal = np.nanmean(group2[:,np.array(unimodal_idx)],axis=1)
    group2_transmodal = np.nanmean(group2[:,np.array(transmodal_idx)],axis=1)
    network_labels = np.concatenate((np.repeat('Unimodal',group1_unimodal.shape[0]),np.repeat('Transmodal',group1_transmodal.shape[0]),np.repeat('Unimodal',group2_unimodal.shape[0]),np.repeat('Transmodal',group2_transmodal.shape[0])),axis=None)
    group_labels = np.concatenate((np.repeat(label1,group1_unimodal.shape[0]+group1_transmodal.shape[0]),np.repeat(label2,group2_unimodal.shape[0]+group2_transmodal.shape[0])),axis=None)
    net_dict = {'Network_type':network_labels,
                'Group_type': group_labels,
                'Tau': np.concatenate((group1_unimodal,group1_transmodal,group2_unimodal,group2_transmodal),axis=None)}
    net_db = pd.DataFrame(net_dict)


    sns.distplot(group1_unimodal)
    s,p= shapiro(group1_unimodal)
    plt.suptitle(f'{label1} unimodal \n Shapiro: s = {s}, p {p}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/unimodal_dist_{label1}{flag}_7net.png')
    plt.close()

    sns.distplot(group1_transmodal)
    s,p= shapiro(group1_transmodal)
    plt.suptitle(f'{label1} transmodal \n Shapiro: s = {s}, p {p}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/transmodal_dist_{label1}{flag}_7net.png')
    plt.close()

    sns.distplot(group2_transmodal)
    s,p= shapiro(group2_transmodal)
    plt.suptitle(f'{label2} transmodal \n Shapiro: s = {s}, p {p}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/transmodal_dist_{label2}{flag}_7net.png')
    plt.close()

    sns.distplot(group2_unimodal)
    s,p= shapiro(group2_unimodal)
    plt.suptitle(f'{label2} unimodal \n Shapiro: s = {s}, p {p}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/unimodal_dist_{label2}{flag}_7net.png')
    plt.close()


    with open(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/unimodal_vs_transmodal_stats_{label1}_{label2}{flag}_7net.txt', 'w') as f:
        statistic,p = kruskal(group1_unimodal,group1_transmodal,group2_unimodal,group2_transmodal)
        print(statistic,p)
        f.write(f'Results for {label1} vs {label2} \n')
        f.write(f'\n Kruskal test: chi = {statistic}, p = {round(p,4)} \n')

        ###### ACROSS GROUPS - WITHIN NETWORKS
        t_unimodal,p_unimodal = mannwhitneyu(group1_unimodal,group2_unimodal)
        d_unimodal = cohend(group1_unimodal,group2_unimodal)
        print('UNIMODAL - t = {}, p = {}, d = {}'.format(t_unimodal,p_unimodal,d_unimodal))
        f.write('###### \n')
        f.write(f'ACROSS GROUPS - WITHIN NETWORKS \n')
        f.write(f'UNIMODAL - t = {t_unimodal}, p = {p_unimodal}, d = {d_unimodal} \n')

        t_transmodal,p_transmodal = mannwhitneyu(group1_transmodal,group2_transmodal)
        d_transmodal = cohend(group1_transmodal,group2_transmodal)
        print('TRANSMODAL - t = {}, p = {}, d = {}'.format(t_transmodal,p_transmodal,d_transmodal))
        f.write(f'TRANSMODAL - t = {t_transmodal}, p = {p_transmodal}, d = {d_transmodal} \n')

        ###### WITHIN GROUPS - ACROSS NETWORKS
        t_dhcp,p_dhcp = wilcoxon(group1_unimodal,group1_transmodal)
        d_dhcp = cohend(group1_unimodal,group1_transmodal)
        print('dHCP - t = {}, p = {}, d = {}'.format(t_dhcp,p_dhcp,d_dhcp))
        f.write('###### \n')
        f.write(f'WITHIN GROUPS - dHCP \n')
        f.write(f'dHCP - t = {t_dhcp}, p = {p_dhcp}, d = {d_dhcp} \n')

        t_camcan,p_camcan = wilcoxon(group2_unimodal,group2_transmodal)
        d_camcan = cohend(group2_unimodal,group2_transmodal)
        print('hcp - t = {}, p = {}, d = {}'.format(t_camcan,p_camcan,d_camcan))
        f.write(f'hcp  - t = {t_camcan}, p = {p_camcan}, d = {d_camcan} \n')

    costum_palette = [sns.xkcd_rgb['lightblue'],sns.xkcd_rgb["burnt orange"]]
    sns.set_palette(costum_palette)
    g1 = sns.boxplot(x='Group_type', y='Tau', hue='Network_type', data = net_dict)
    #g1 = sns.swarmplot(x='Group_type', y='Tau', hue='Network_type', data = net_dict)
    #g1 = sns.stripplot(x='Group_type', y='Tau', hue='Network_type',data= net_dict,color='black')
    #x_coords = [p.get_x() + 0.5*p.get_width() for p in g1.patches]
    #y_coords = [p.get_height() for p in g1.patches]
    #plt.errorbar(x_coords, y_coords, fmt='none', yerr=[sem((group1_unimodal-group1_transmodal)/2),sem((group1_unimodal-group1_transmodal)/2),sem((group2_unimodal-group2_transmodal)/2),sem((group2_unimodal-group2_transmodal)/2)], c="black", elinewidth=2)
    plt.ylim((-1,20))
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 15)
    plt.ylabel('Tau (in seconds)', fontsize = 15)
    ax = g1.axes
    # unimodal dhcp vs hcp
    ax.plot([-0.2,-0.2, 0.8,0.8], [15.9,15.95,15.95,15.9], lw=1.5, color = 'black')
    ax.text((-0.2+0.8)*.5, 16, "***", ha='center', va='bottom',fontsize = 20)
    # transmodal dhcp vs hcp    
    ax.plot([0.2,0.2, 1.2,1.2], [17.9,17.95,17.95,17.9], lw=1.5, color = 'black')
    ax.text((0.2+1.2)*.5, 18, "***", ha='center', va='bottom',fontsize = 20)    
    # unimodal vs transmodal hcp
    ax.plot([0.8,0.8, 1.2,1.2], [14.9,14.95,14.95,14.9], lw=1.5, color = 'black')
    ax.text((0.8+1.2)*.5, 15, "***", ha='center', va='bottom',fontsize = 20)    
    # unimodal vs transmodal dhcp
    ax.plot([-0.2,-0.2,0.2,0.2], [12.9,12.95,12.95,12.9], lw=1.5, color = 'black')
    ax.text((-0.2+0.2)*.5, 13, "***", ha='center', va='bottom',fontsize = 20)    

    plt.suptitle(f'{label1} vs {label2}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/unimodal_transmodal_{label1}_{label2}{flag}_7net.png')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/unimodal_transmodal_{label1}_{label2}{flag}_7net.pdf')
    plt.close()
