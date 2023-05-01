import numpy as np 
import os
from scipy import optimize
import pandas as pd
from hep_ml import reweight
from sklearn.model_selection import train_test_split


def re_weight_dist(hcp_snr,dhcp_snr):
    bins_reweighter = reweight.BinsReweighter(n_bins=100, n_neighs=1)
    bins_reweighter.fit(hcp_snr.flatten(), dhcp_snr.flatten())
    bins_weights = bins_reweighter.predict_weights(hcp_snr.flatten())

    bins_weights = bins_weights.reshape((hcp_snr.shape[0],hcp_snr.shape[1]))

    return bins_weights


