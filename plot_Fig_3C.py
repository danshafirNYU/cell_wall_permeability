#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 12:12:57 2025

This function plots Fig3C in menuscript https://doi.org/10.1101/2025.09.12.675941 

@author: danshafir
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import re
import random

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


global_font_size = 14

# Set global font size
plt.rcParams.update({
    "font.size": global_font_size,         # axis labels, tick labels
    "axes.labelsize": global_font_size,    # x and y axis labels
    "xtick.labelsize": global_font_size,   # x tick labels
    "ytick.labelsize": global_font_size,   # y tick labels
    "legend.fontsize": global_font_size,   # legend
})


def compute_survival_probability(first_passage_times, N, dN=1):
    """
    Compute survival probability up to step N, sampled every dN steps.

    Parameters:
    - first_passage_times: 1D np.ndarray of first passage times (or list)
    - N: total number of steps to compute survival up to
    - dN: step resolution for computing survival probabilities

    Returns:
    - survival_probs: np.ndarray of survival probabilities at times [dN, 2*dN, ..., N]
    """
    first_passage_times = np.array(first_passage_times)
    first_passage_times = np.where(first_passage_times == -1, np.inf, first_passage_times)
    num_particles = len(first_passage_times)
    sorted_times = np.sort(first_passage_times)

    times = np.arange(dN, N + 1, dN)
    survival_probs = np.empty(len(times))

    for i, t in enumerate(times):
        survivors = np.searchsorted(sorted_times, t, side='right')
        survival_probs[i] = (num_particles - survivors) / num_particles

    return survival_probs

def load_all_batches_resampled_with_replacement(directory='.', prefix='data_batch_', extension='.bin', dtype=np.double):
    file_pattern = os.path.join(directory, prefix + '*' + extension)
    files = glob.glob(file_pattern)
    resampled_files = random.choices(files, k=len(files))
    if not files:
        raise FileNotFoundError(f"No files matching pattern {file_pattern}")
    data_list = []
    for file in resampled_files:
        data = np.fromfile(file, dtype=dtype)
        data_list.append(data)
    return np.concatenate(data_list)

def load_all_batches(directory='.', prefix='data_batch_', extension='.bin', dtype=np.double):
    file_pattern = os.path.join(directory, prefix + '*' + extension)
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching pattern {file_pattern}")

    data_list = []
    for file in files:
        data = np.fromfile(file, dtype=dtype)
        
        print(np.where(data<1)[0].size)
        
        data_list.append(data)

    return np.concatenate(data_list)

W = 8  # This is the cell wall width in units of (mesh_scale * step_size)
cell_width_list = np.array([800])
folder_str='Fig3C_data/'

p_occupation_list = np.array([0.67])
particle_size_list = np.array([4])

mesh_scale = 4  # defines the characteristic wall unit size which can be occupied or not 

# Plot
linetype = np.array(['-', '--', '-.'])
colors = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
    'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy', 'teal'
]

plt.figure(figsize=(6, 5))

for idx, particle_size in enumerate(particle_size_list):
    
    N_max = int(10**9.38)
    dN = int(1e6)
    
    step_size = 0.9 #in nano-meters
    dl = 9e-10 # nano meter
    D = 0.81*(1e-6 )**2 # micro meter ^ 2 / sec
    delta_t = dl**2 / D / 2
    print(delta_t)
    
    delta_t = delta_t / 60 # in minutes
    color_id=0
    
    for cell_width in cell_width_list:
        for p in p_occupation_list:
            
            
            full_path = folder_str + f"p_{p:.3f}_W_{W}_periodic_y_particle_size_{particle_size}_mesh_{mesh_scale}"
            
            try:
                samples = load_all_batches(directory=full_path)
            except FileNotFoundError:
                print(f"Warning: Folder '{full_path}' not found or empty. Skipping.")
                continue
            
            time_vec = np.arange(dN, N_max + 1, dN );
            time_vec = time_vec * delta_t            
            survival_probs = compute_survival_probability(samples , N_max, dN)
            
            
            
            # Calculation of the standard error using bootstrap method
            ########################################################################
            std_vec = np.zeros(time_vec.size)
            N_resamps = 200
            survival_probs_resamp = np.zeros([N_resamps, time_vec.size])
            
            for resamp_id in range(N_resamps):
                samples = load_all_batches_resampled_with_replacement(directory=full_path)
                survival_probs_resamp[resamp_id] = compute_survival_probability(samples , N_max, dN)
                
            std_vec = np.std(survival_probs_resamp, axis=0, ddof=0)
            
            #########################################################################
            # end of calcualtion of the standard error
            
            
            # Plotting section
            #############################################################
            step = 100  # show error bars every 100 points
            plt.plot(
                time_vec, survival_probs,
                label=f'$p={p}, \; size={particle_size*step_size} nm$',
                linestyle=linetype[idx],
                linewidth=2,
                color=colors[color_id]
            )
            plt.errorbar(
                time_vec[::step],             # subsample x
                survival_probs[::step],       # subsample y
                yerr=std_vec[::step],         # subsample errors
                fmt='none',                   # no extra markers
                ecolor=colors[color_id],      # error bar color
                capsize=3
            )
            color_id = color_id + 1
            

particle_size = 5
###################3
# New part
full_path = folder_str + f"p_{p:.3f}_W_{W}_periodic_y_particle_size_{particle_size}_mesh_{mesh_scale}"
try:
    samples = load_all_batches(directory=full_path)
except FileNotFoundError:
    print(f"Warning: Folder '{full_path}' not found or empty. Skipping.")
survival_probs = compute_survival_probability(samples , N_max, dN)
####################
plt.plot(time_vec, survival_probs,
         label=f'$p={p}, \; size={particle_size*step_size} nm$', linestyle = linetype[idx], linewidth=2, color = colors[color_id])

plt.xlabel('time (minutes)')
plt.ylabel('Survival Probability')
plt.xlim([-0.1, 20])
plt.ylim([-0.05, 1.05])

plt.legend()
plt.tight_layout()
plt.show()