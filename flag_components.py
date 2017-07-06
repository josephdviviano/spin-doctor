#!/usr/bin/env python

import numpy as np
import nibabel as nib
import matplotlib as ml
import matplotlib.pyplot as plt
from scipy.fftpack import fft2
from skimage.draw import circle
from copy import copy


def zscore(X):
    """Z-score data"""
    return((X - X.mean()) / X.std())

def zscore_brain_slice(X):
    """
    Z-score a brain slice excluding the outside of the brain.
    """
    X[X == 0] = np.nan
    X = (X - np.nanmean(X)) / np.nanstd(X)
    X[np.isnan(X)] = 0

    return(X)

def main(input_comps, output_csv, plot=False):

    freq_thresh = 1.8
    std_thresh = 1.5

    # load NIFTI ICA components from MELODIC
    data = nib.load(input_comps).get_data()
    x, y, z, comps =  data.shape
    #pad = 2 # ignore extreme slices
    pxx_his, pxx_mid, pxx_los, stds, ratios, totals, comp_n = [], [], [], [], [], [], []

    for comp in range(comps):

        # extract the 3d volume for this component's loadings
        d = copy(data[:,:,:,comp])
        d_std = copy(d)
        d_std[d_std == 0] = np.nan

        # march through the slices
        for zslice in range(z):

            # calculate deviation of slice from vol
            idx = np.setdiff1d(np.arange(z), zslice)
            vol_std = np.nanstd(d_std[:, :, idx])
            vol_mean = np.nanmean(d_std[:, :, idx])

            zslice_mean = np.nanmean(d_std[:, :, zslice])
            zslice_z = (zslice_mean - vol_mean) / vol_std #

            # calculate normalized k-space of slice
            pxx_data = zscore_brain_slice(copy(d[:,:,zslice]))
            pxx = abs(fft2(pxx_data))**2

            # high mask is the outer corners of k-space
            hi_mask = np.ones((x,y))
            rr, cc = circle(x/2, y/2, x/2)
            hi_mask[rr, cc] = 0
            hi_mask = hi_mask.astype(np.bool)

            # mid mask is from diamater of image (excluding corners) to 1/8th radius
            mid_mask = np.zeros((x,y))
            mid_mask[rr,cc] = 1
            rr, cc = circle(x/2, y/2, x/8)
            mid_mask[rr, cc] = 0
            mid_mask = mid_mask.astype(np.bool)

            # low mask is very centre of image (1/8th radius)
            lo_mask = np.zeros((x,y))
            rr, cc = circle(x/2, y/2, x/8)
            lo_mask[rr, cc] = 1
            lo_mask = lo_mask.astype(np.bool)

            # get mean power from each frequency bin
            hi_pxx = np.mean(pxx[hi_mask])
            mid_pxx = np.mean(pxx[mid_mask])
            lo_pxx = np.mean(pxx[lo_mask])

            pxx_total = np.sum(pxx)
            pxx_ratio = hi_pxx / lo_pxx

            pxx_his.append(hi_pxx)
            pxx_mid.append(mid_pxx)
            pxx_los.append(lo_pxx)
            ratios.append(pxx_ratio)
            totals.append(pxx_total)
            comp_n.append(comp)
            stds.append(zslice_z)

    # normalize all measures across all slices across all components
    comp_n = np.hstack(comp_n)
    pxx_his = zscore(np.hstack(pxx_his))
    pxx_los = zscore(np.hstack(pxx_los))
    pxx_mid = zscore(np.hstack(pxx_mid))
    ratios = zscore(np.hstack(ratios))
    totals = zscore(np.hstack(totals))
    stds = zscore(np.hstack(stds))

    # we want components that have both an unusual signal amplitude in one slice
    # relative to other slices in the same component, which also has a lot of
    # mid-frequency energy

    remove_list = []
    flag = False
    for comp in range(comps):
        idx = comp_n == comp
        for zslice in range(z):
            if stds[idx][zslice] >= std_thresh and pxx_mid[idx][zslice] >= freq_thresh:
                print('comp={}, z={}, std={}, mid freq pxx={}'.format(
                    comp+1, zslice+1, stds[idx][zslice], pxx_mid[idx][zslice]))
                flag = True

        if flag:
            remove_list.append(str(comp+1)) # add comp to be removed

            if plot:
                plt.plot(stds[idx], label='slice standard deviation')
                plt.plot(pxx_mid[idx], label='mid freq power')
                plt.legend()
                plt.title('comp={}'.format(comp+1))
                plt.ylim(-5, 5)
                plt.show()
            flag = False

    f = open(output, 'wb')
    f.write(','.join(remove_list) + '\n')


if __name__ == '__main__':

    input_comps = '/scratch/jviviano/20160623_Ex05620_STOP2MR_STAF002_SpiralSeparated/sprlOUT/Prestats.feat/filtered_func_data.ica/melodic_IC.nii.gz'
    output = 'test.csv'
    main(input_comps, output, plot=False)


