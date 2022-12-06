import numpy as np 
import pandas as pd 
import os


if __name__ == '__main__':
    infopth = '/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data' 
    dhcp_info = pd.read_csv(os.path.join(infopth,'participants_dhcp.txt'), sep='\t')
    dhcp_scan_info = pd.read_csv(os.path.join(infopth,'dhcp_scan_info.csv'))
    dual_sess_subj = list(pd.read_csv(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/48preterm.csv').iloc[:,0])
    dual_sess_subj = ['sub-'+subj for subj in dual_sess_subj]

    group_list = ['dhcp_group1','dhcp_group2']
    for group in group_list:
        dhcp_subj = pd.read_csv(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/{group}_subj_list.csv',header=None)
        subj_list = list(dhcp_subj.iloc[:,0])
        subj_list = [subj for subj in subj_list if subj.split('\'')[1] not in dual_sess_subj]
        
        dhcp_subj_tot = [code.split("'")[1].split('-')[-1] for code in subj_list]
        dhcp_sess_tot = [code.split("'")[3].split('-')[-1] for code in subj_list]
        #for code in dhcp_subj:
                #dhcp_subj_tot.append(code.split("'")[1].split('-')[-1])
                #dhcp_sess_tot.append(code.split("'")[3].split('-')[-1])
        dhcp_scan_age = []
        dhcp_gender = []
        dhcp_birthday = []
        dhcp_weight = []
        dhcp_scan_gender = []
        import collections
        print([item for item, count in collections.Counter(dhcp_subj_tot).items() if count > 2])
        for i,subj in enumerate(dhcp_subj_tot):
            #print(subj)
            curr_sess = dhcp_sess_tot[i].split('-')[-1]
            dhcp_gender.append(dhcp_info.loc[dhcp_info['participant_id']==subj]['gender'].values[0])
            dhcp_birthday.append(dhcp_info.loc[dhcp_info['participant_id']==subj]['birth_age'].values[0])
            dhcp_weight.append(dhcp_info.loc[dhcp_info['participant_id']==subj]['birth_weight'].values[0])
            dhcp_scan_age.append(dhcp_scan_info.loc[(dhcp_scan_info['subj']==f'sub-{subj}') & (dhcp_scan_info['sess']==int(curr_sess))]['scan_age'].values[0])
        
        single_subj = list(set(dhcp_subj_tot))
        for i,subj in enumerate(single_subj):
            dhcp_scan_gender.append(dhcp_scan_info.loc[(dhcp_scan_info['subj']==f'sub-{subj}')]['scan_sex'].values[0])
        
        with open(f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/demographics_{group}.txt','w') as f:
            dhcp_birthday_mean = np.nanmean(dhcp_birthday)
            dhcp_birthday_sd = np.nanstd(dhcp_birthday)
            print(f'dhcp {group} birthday max: {np.nanmax(dhcp_birthday)}')
            print(f'dhcp {group} birthday min: {np.nanmin(dhcp_birthday)}')
            f.write(f'dhcp {group} birthday max: {np.nanmax(dhcp_birthday)} \n')
            f.write(f'dhcp {group} birthday min: {np.nanmin(dhcp_birthday)} \n')
            f.write(f'dhcp {group} birthday mean: {np.nanmean(dhcp_birthday)} \n')
            f.write(f'dhcp {group} birthday SD: {np.nanstd(dhcp_birthday)} \n')

            if 0 in dhcp_weight:
                dhcp_weight = np.array(dhcp_weight)
                dhcp_weight[np.where(dhcp_weight==0)[0]]=np.nan
                print(f'there are this many zero weight:{np.where(dhcp_weight==0)}')
            dhcp_weight_mean = np.nanmean(dhcp_weight)    
            dhcp_weight_sd = np.nanstd(dhcp_weight)
            print(f'dhcp {group} weight max: {np.nanmax(dhcp_weight)} \n')
            print(f'dhcp {group} weight min: {np.nanmin(dhcp_weight)} \n')
            f.write(f'dhcp {group} weight max: {np.nanmax(dhcp_weight)} \n')
            f.write(f'dhcp {group} weight mean: {np.nanmean(dhcp_weight)} \n')
            f.write(f'dhcp {group} weight SD: {np.nanstd(dhcp_weight)} \n')

            dhcp_scan_age_mean = np.nanmean(dhcp_scan_age)
            dhcp_scan_age_sd = np.nanstd(dhcp_scan_age)
            print(f'dhcp {group} age max: {np.nanmax(dhcp_scan_age)} \n')
            print(f'dhcp {group} age min: {np.nanmin(dhcp_scan_age)} \n')
            f.write(f'dhcp {group} age max: {np.nanmax(dhcp_scan_age)} \n')
            f.write(f'dhcp {group} age min: {np.nanmin(dhcp_scan_age)} \n')
            f.write(f'dhcp {group} age mean: {np.nanmean(dhcp_scan_age)} \n')
            f.write(f'dhcp {group} age SD: {np.nanstd(dhcp_scan_age)} \n')


            dhcp_males = len([x for x in dhcp_scan_gender if x=='Male'])
            dhcp_females = len([x for x in dhcp_scan_gender if x=='Female'])
            print(f'dhcp {group} males n: {dhcp_males}')
            print(f'dhcp {group} females n: {dhcp_females}')
            f.write(f'dhcp {group} males n: {dhcp_males} \n')
            f.write(f'dhcp {group} females n: {dhcp_females}')




