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
import statsmodels.api as sm
from pingouin import partial_corr

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


def age_model(tau,scan_age,experience_age,network_type,label,flag,snr_flag):  
    data_dict = {'scan_age':np.array(scan_age),
                    'experience_age':np.array(experience_age),
                    'tau':tau}
    data_df = pd.DataFrame(data_dict)
    X = data_df[['scan_age','experience_age']]
    X = sm.add_constant(X) 
    model=sm.OLS(data_df['tau'],X)
    results=model.fit()
    print(results.summary()) 
    r_collinearity,p_collinearity = spearmanr(np.array(scan_age),np.array(experience_age))
    
    plt.scatter(data_df['scan_age'],data_df['tau'])
    plt.suptitle('Relation between scan_age and tau')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/age_model_scan_age_relation_{label}_{network_type}_{flag}{snr_flag}.png')
    plt.close()

    plt.scatter(data_df['experience_age'],data_df['tau'])
    plt.suptitle('Relation between experience_age and tau')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/age_model_experience_age_relation_{label}_{network_type}_{flag}{snr_flag}.png')
    plt.close()

    par_corr_scan_age = partial_corr(data=data_df, x='scan_age', y='tau', covar='experience_age', method='spearman')
    par_corr_exp_age = partial_corr(data=data_df, x='experience_age', y='tau', covar='scan_age', method='spearman')
    with open(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/age_model_{label}_{network_type}_{flag}{snr_flag}.txt', 'w') as f:
        print(results.summary(), file=f)
        f.write(f'\n Collinearity check: r={r_collinearity}, p={p_collinearity} \n')
        f.write('####  Partial corr - scan_age & tau \n')
        print(par_corr_scan_age,file=f)
        f.write('####  Partial corr - experience_age & tau \n')
        print(par_corr_exp_age,file=f)


def age_corr(tau,experience_age,network_type,label,flag,snr_flag):
    outliers=['','_nooutliers']
    for outlier_flag in outliers:
        if outlier_flag=='_nooutliers':
            nooutlier_idx = np.where(experience_age<(np.mean(experience_age+(2*np.std(experience_age)))))
            experience_age = experience_age[nooutlier_idx]
            tau = tau[nooutlier_idx]
            xlim = (-0.05,7)
        else:
            xlim = (-0.05,18)
        r,p = spearmanr(np.array(experience_age),np.array(tau))
        plt.scatter(np.array(experience_age),np.array(tau),alpha=0.5,s=15,color='#014182')
        plt.xlim(xlim)
        plt.ylim((-0.05,14))
        plt.suptitle(f'Experience_age and Tau - {label} \n r={r}, p={p}')
        plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/experience_age_corr_with_tau_{label}_{network_type}_{flag}{snr_flag}{outlier_flag}.png')
        plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/experience_age_corr_with_tau_{label}_{network_type}_{flag}{snr_flag}{outlier_flag}.pdf')
        plt.close()


def unimodal_vs_transmodal(group1, group2, unimodal_idx, transmodal_idx, group2_weights, label1, label2,flag,snr_flag):
    if snr_flag=='':
        group1_unimodal = np.nanmean(group1[:,np.array(unimodal_idx)],axis=1)
        group1_transmodal = np.nanmean(group1[:,np.array(transmodal_idx)],axis=1)
        group2_unimodal = np.nanmean(group2[:,np.array(unimodal_idx)],axis=1)
        group2_transmodal = np.nanmean(group2[:,np.array(transmodal_idx)],axis=1)
    else:
        group1_unimodal = np.nanmean(group1[:,np.array(unimodal_idx)],axis=1)
        group1_transmodal = np.nanmean(group1[:,np.array(transmodal_idx)],axis=1)
        group2_unimodal = np.nansum(group2[:,np.array(unimodal_idx)]*group2_weights[:,np.array(unimodal_idx)],axis=1) / np.sum(group2_weights[:,np.array(unimodal_idx)],axis=1)
        group2_transmodal = np.nansum(group2[:,np.array(transmodal_idx)]*group2_weights[:,np.array(transmodal_idx)],axis=1) / np.sum(group2_weights[:,np.array(transmodal_idx)],axis=1)

    network_labels = np.concatenate((np.repeat('Unimodal',group1_unimodal.shape[0]),np.repeat('Transmodal',group1_transmodal.shape[0]),np.repeat('Unimodal',group2_unimodal.shape[0]),np.repeat('Transmodal',group2_transmodal.shape[0])),axis=None)
    group_labels = np.concatenate((np.repeat(label1,group1_unimodal.shape[0]+group1_transmodal.shape[0]),np.repeat(label2,group2_unimodal.shape[0]+group2_transmodal.shape[0])),axis=None)
    net_dict = {'Network_type':network_labels,
                'Group_type': group_labels,
                'Tau': np.concatenate((group1_unimodal,group1_transmodal,group2_unimodal,group2_transmodal),axis=None)}
    net_db = pd.DataFrame(net_dict)

    subj_list_dhcp = list(pd.read_csv(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/tau_subj_list_{label1}_7net_{flag}.csv')['subj'])
    dhcp_demographic_info=pd.read_csv(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/participants_dhcp.txt',sep='\t')
    dhcp_scan_info=pd.read_csv(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/dhcp_scan_info.csv')

    scan_age = []
    birth_age = []
    for subj in subj_list_dhcp:
        scan_age.extend(dhcp_scan_info[dhcp_scan_info['subj']==subj]['scan_age'].values)
        print(subj)
        print(dhcp_scan_info[dhcp_scan_info['subj']==subj]['scan_age'].values)
        birth_age.extend(dhcp_demographic_info[dhcp_demographic_info['participant_id']==subj.split('-')[-1]]['birth_age'].values)

    experience_age = np.array(scan_age)-np.array(birth_age)

    if 'drop_scan' in flag:
        age_corr(group1_unimodal,experience_age,'unimodal',label1,flag,snr_flag)
        age_corr(group1_transmodal,experience_age,'transodal',label1,flag,snr_flag)
    
    sns.distplot(group1_unimodal)
    s,p= shapiro(group1_unimodal)
    plt.suptitle(f'{label1} unimodal \n Shapiro: s = {s}, p {p}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/unimodal_dist_{label1}_7net_{flag}{snr_flag}.png')
    plt.close()

    sns.distplot(group1_transmodal)
    s,p= shapiro(group1_transmodal)
    plt.suptitle(f'{label1} transmodal \n Shapiro: s = {s}, p {p}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/transmodal_dist_{label1}_7net_{flag}{snr_flag}.png')
    plt.close()

    sns.distplot(group2_transmodal)
    s,p= shapiro(group2_transmodal)
    plt.suptitle(f'{label2} transmodal \n Shapiro: s = {s}, p {p}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/transmodal_dist_{label2}_7net_{flag}{snr_flag}.png')
    plt.close()

    sns.distplot(group2_unimodal)
    s,p= shapiro(group2_unimodal)
    plt.suptitle(f'{label2} unimodal \n Shapiro: s = {s}, p {p}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/unimodal_dist_{label2}_7net_{flag}{snr_flag}.png')
    plt.close()


    with open(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/unimodal_vs_transmodal_stats_{label1}_{label2}_7net_{flag}{snr_flag}.txt', 'w') as f:
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
    if flag=='drop_scan_dhcp':
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

    plt.suptitle(f'{label1} vs {label2}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/unimodal_transmodal_{label1}_{label2}_7net_{flag}{snr_flag}.png')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/unimodal_transmodal_{label1}_{label2}_7net_{flag}{snr_flag}.pdf')
    plt.close()
