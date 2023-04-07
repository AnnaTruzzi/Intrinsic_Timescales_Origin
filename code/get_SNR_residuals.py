import numpy as np 
import os
from scipy import optimize
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pingouin import partial_corr
from statsmodels.regression import linear_model
import matplotlib.pyplot as plt

def get_SNR_res(tau_array, snr_array, group, flag):
    plt.scatter(snr_array,tau_array)
    plt.suptitle(f'SNR vs Tau - {group}')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/snr_relation_with_tau_{group}_7net_{flag}.png')
    plt.savefig(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/figures/tau_bynet_{group}_7net_{flag}.pdf')
    plt.close()

    snr_array = sm.add_constant(snr_array)
    model=sm.OLS(tau_array,snr_array)
    results=model.fit()
    print(results.summary())

    residuals = results.res
    np.savetxt(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/residuals_SNR_{group}_7net_{flag}.txt',residuals)

    
