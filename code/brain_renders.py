import numpy as np
import pickle
import pandas as pd
import glob
import os
import nibabel as nib

def render(index_list,out_values,atlas,outvolume_size,outname):
    atlas_data = atlas.get_fdata()
    outvolume = np.zeros((outvolume_size.shape[0],outvolume_size.shape[1],outvolume_size.shape[2]))
    for i,roi in enumerate(index_list):
        roi_index = np.where(atlas_data==roi+1)
        outvolume[roi_index] = out_values[i]
    outimage = nib.Nifti1Image(outvolume, affine=atlas.affine)
    nib.save(outimage,outname)
    outimage.uncache()
    outimage.uncache()


def brainrenders(group,tau_mean,net_dict):
    if 'dhcp' in group:
        ## Load templates and set outvolume size
        atlas = nib.load('/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/schaefer_40weeks_7net.nii.gz')
        outvolume_size = np.zeros((202, 274, 217))
    else:
        atlas  = nib.load('/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
        outvolume_size = np.zeros((91, 109, 91))

    atlas_data = atlas.get_fdata()

    ##### TAU distribution visualisation    
    print(f'Working on tau render for {group}....')
    render(range(0,tau_mean.shape[0]),tau_mean,atlas,outvolume_size,f"/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/{group}_tauvalues_render_7net.nii.gz")

    ##### ROI visualisation and grouping

    ## Load ROI label file
    print(f'Working on roi renders....')
    network_file7 = pd.read_csv('/dhcp/fmri_anna_graham/dhcp_hcp_timescales/data/Schaefer2018_400Parcels_7Networks_order.txt',sep = '\t', header = None)
    roi_names_all = np.array(network_file7[1])

    ## Get indexes and names of unimodal vs transmodal rois + make brain render file.
    unimodal_index=[]
    transmodal_index=[]
    for key in net_dict.keys():
        if 'Vis' in key or 'Som' in key:
            unimodal_index.extend(net_dict[key])
        else:
            transmodal_index.extend(net_dict[key])
    roi_value = np.concatenate((np.repeat(1,len(unimodal_index)),np.repeat(2,len(transmodal_index))))
    uni_vs_trans_index = unimodal_index + transmodal_index
    render(uni_vs_trans_index,roi_value,atlas, outvolume_size,f"/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/{group}_uni_vs_transmodal_render_all_7net.nii.gz")

    ## Get indexes and names of 7 networks + make brain render file unique for all nets.
    net_idx=[]
    netnum_list = []
    for netnum,net in enumerate(net_dict.keys()):
        net_idx.extend(net_dict[net])
        netnum_list.extend(np.repeat(netnum+1,len(net_dict[net])))
    render(net_idx,netnum_list,atlas,outvolume_size,f"/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/{group}_7networks_render.nii.gz")


    for net in net_dict.keys():
        idx=net_dict[net]
        render(idx, np.repeat(1,len(idx)),atlas,outvolume_size,f'/dhcp/fmri_anna_graham/dhcp_hcp_timescales/results/{group}_{net}_render_7net.nii.gz')
