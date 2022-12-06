import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    infopth = '/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data' 
    hcp_info = pd.read_csv(os.path.join(infopth,'hcp_participants_info.csv'))
    subj_list = pd.read_csv(os.path.join(infopth,'hcp_subj_list.csv'))['subj'].tolist()

    hcp_info2 = hcp_info[hcp_info['Subject'].isin(subj_list)]

    nfemales = hcp_info2['Gender'].value_counts()[0]
    nmales = hcp_info2['Gender'].value_counts()[1]

    min_age = np.min(np.array(hcp_info2['Age']))
    max_age = np.max(np.array(hcp_info2['Age']))
    mean_age = np.mean(np.array([22,36]))

    with open(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/demographics_hcp.txt','w') as f:
        f.write(f'tot sample:{hcp_info2.shape[0]} \n')
        f.write(f'tot females:{nfemales} \n')
        f.write(f'tot males:{nmales} \n')
        f.write(f'min age:{min_age} \n')
        f.write(f'max age:{max_age} \n')
        f.write(f'mean age:{mean_age} \n')



