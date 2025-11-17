#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 12:12:57 2025

This function plots Fig3B in menuscript https://doi.org/10.1101/2025.09.12.675941 


@author: danshafir
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import random

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
        data_list.append(data)

    return np.concatenate(data_list)


t_min = 1.0 # the value of time we want to check for. How many are still inside at t_min, in units of minutes
W = 8 # This is the cell wall width in units of (mesh_scale * step_size)
mesh_scale = 4 # defines the characteristic wall unit size which can be occupied or not 

relax_percent = 10
cell_width_list = np.array([800])
folder_str = f'Fig3B_data/{relax_percent}_percent/'
particle_size_list = np.array([4, 5])

# Plot
linetype = np.array(['-*', '-*', '-'])
colors = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
    'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy', 'teal'
]


for idx, particle_size in enumerate(particle_size_list):
    
    color_id = 0  # the id of the color, starting over for each particle size
    N_max = int(10**8.7) # total steps in the simulation
    dN = int(1e4) # resolution of the plot in time steps
    delta_t = 0.5e-6 / 60 # This depends on step size and Diffusion coefficient
    data_folders = glob.glob(os.path.join(folder_str,f"*_size_{particle_size}_*"))
    p_occupation_list = sorted([float(os.path.basename(x).split("_")[1]) for x in data_folders])
    
    print(p_occupation_list)
    

    for cell_width in cell_width_list:
        
        fraction_inside = np.zeros(np.size(p_occupation_list))
        std_vec = np.zeros(np.size(p_occupation_list))
        
        for p_ind, p in enumerate(p_occupation_list):

            full_path = folder_str + f"p_{p:.3f}_W_{W}_relax_{relax_percent}_periodic_y_particle_size_{particle_size}_mesh_{mesh_scale}"
            try:
                samples = load_all_batches(directory=full_path)
            except FileNotFoundError:
                print(f"Warning: Folder '{full_path}' not found or empty. Skipping.")
                continue
            
            time_vec = np.arange(dN, N_max + 1, dN );
            time_vec = time_vec * delta_t
            time_ind = np.where(time_vec > t_min)[0][0]
            
            survival_probs = compute_survival_probability(samples , N_max, dN)
            fraction_inside[p_ind] = survival_probs[time_ind]
            
            # Calculation of the standard error
            ######################################################3
            N_resamps = 200
            fraction_inside_resamp = np.zeros([N_resamps])
            
            for resamp_id in range(N_resamps):
                samples = load_all_batches_resampled_with_replacement(directory=full_path)
                survival_probs_temp = compute_survival_probability(samples , N_max, dN)
                fraction_inside_resamp[resamp_id] = survival_probs_temp[time_ind]
                
            std_vec[p_ind] = np.std(fraction_inside_resamp, ddof=0)
            ##########################################################
            # end of calcualtion of the standard error
            
            
        ###############################################
        # plot part
        
        plt.errorbar(
            p_occupation_list,           # x values
            fraction_inside,             # y values (means)
            yerr=std_vec,                # error bar heights (std devs)
            fmt='-',                     # line style ('-' for line, 'o-' for line+markers)
            label=f'$size={particle_size} nm$',
            linewidth=3,
            color=colors[idx],
            capsize=7                    # small horizontal cap on error bars
        )
        
        color_id = color_id + 1

plt.xlabel('p, density of the wall')
plt.ylabel(f'Fraction inside at $t={t_min}$ minutes')

# plt.axvline(x=0.485, color='magenta', linestyle='--', linewidth=3)
# plt.axvline(x=0.75, color='magenta', linestyle='--', linewidth=3)

# 5 percent
# plt.axvline(x=0.51, color='magenta', linestyle='--', linewidth=3)
# plt.axvline(x=0.685, color='magenta', linestyle='--', linewidth=3)

# 10 percent
plt.axvline(x=0.505, color='magenta', linestyle='--', linewidth=3)
plt.axvline(x=0.64, color='magenta', linestyle='--', linewidth=3)

plt.xlim([0.0,1.02])
plt.legend()
plt.tight_layout()
plt.show()
