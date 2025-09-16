#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 20:45:58 2025

This function plots various trajectory examples inside the cell.
This is Figure S6 in SI in menuscript https://doi.org/10.1101/2025.09.12.675941

@author: danshafir
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

global_font_size = 18

# Set global font size
plt.rcParams.update({
    "font.size": global_font_size,         # axis labels, tick labels
    "axes.labelsize": global_font_size,    # x and y axis labels
    "xtick.labelsize": global_font_size,   # x tick labels
    "ytick.labelsize": global_font_size,   # y tick labels
    "legend.fontsize": global_font_size,   # legend
})

def plot_time_colored_trajectory(pos, t, ax, cmap="viridis", lw=2):
    """
    Plot a trajectory with color corresponding to absolute simulation time.
    
    Parameters
    ----------
    pos : np.ndarray
        Array of shape (N, 2) containing x,y positions.
    t : np.ndarray
        Array of shape (N,) containing times corresponding to positions.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    cmap : str, optional
        Colormap name (default 'viridis').
    lw : float, optional
        Line width (default 2).
    """
    if pos.shape[0] != len(t):
        raise ValueError("pos and t must have the same length")
    
    # Prepare line segments
    points = pos.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Normalize by absolute times
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t)
    lc.set_linewidth(lw)
    
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", "box")
    
    # Return the LineCollection so user can attach colorbar
    return lc


def break_y_periodic_jumps(x, y, h):
    """
    Insert NaNs in the trajectory to break the line when y jumps across periodic boundary.
    
    Parameters:
        x (array-like): x positions of the trajectory
        y (array-like): y positions of the trajectory
        h (int or float): height of the box (y in [0, h))

    Returns:
        tuple: (x_plot, y_plot) with NaNs inserted to break lines
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    x_plot = []
    y_plot = []
    
    for i in range(len(x) - 1):
        x_plot.append(x[i])
        y_plot.append(y[i])
        
        # Check for a wrap in y
        if abs(y[i+1] - y[i]) > h / 2:
            x_plot.append(np.nan)
            y_plot.append(np.nan)

    # Add final point
    x_plot.append(x[-1])
    y_plot.append(y[-1])
    
    return np.array(x_plot), np.array(y_plot)

def set_x_y_labels(ax):
    ax.set_xlim(-L - W - 1, 2 + W)
    ax.set_ylim(-2, h + 1)
    ax.set_aspect('equal')

    # Define your new tick positions (numeric positions)
    new_ticks_x = [-L - W, -L, -L/2, 0, W]
    # Define the labels you want to show
    new_labels_x = [
    r"$-(R\!+\!\delta)$", # tighter
    r"$-R$", 
    r"$0$", 
    r"$R$", 
    r"$R+\delta$"
    ]
    # Set them
    ax.set_xticks(new_ticks_x)
    ax.set_xticklabels(new_labels_x)

    # Define your new tick positions (numeric positions)
    new_ticks_y = [0 , h/4 , h/2, 3*h/4, h]
    # Define the labels you want to show
    new_labels_y = ["0", "1/4 h", "1/2 h", "3/4 h", "h"]
    # Set them
    ax.set_yticks(new_ticks_y)
    ax.set_yticklabels(new_labels_y)
    ax.yaxis.set_ticks([])  # removes both ticks and labels

def plot_wall(obs_or_not, W, h, size, ax, color, x_start = 0):
    # The wall section
    # size - size of the lattice unit

    # plot the obstacles
    for i in range(0, W):
        for j in range(0, h):
            if obs_or_not[i,j] > 0:
                ax.add_patch(patches.Rectangle(((x_start + i)* size + 0.5, j * size - 0.5), size, size, color=color))

rng = np.random.default_rng()
# p_occupation = 0.4
W = 20
h = 200
L = 200
N_steps = int(1e5)
multi_res = 2

fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of subplots
axs = axs.flatten()  # make it a 1D array for easy indexing

subplot_titles = ["Particle escaped through a sparse wall (p=0.3)", "Particle is still trapped inside (p=0.5)", "Particle escaped (p=0.5)", "Particle can never exit with a highly dense wall (p=0.8)"]

for file_id in range(4):
    ax = axs[file_id]  # pick the correct subplot
    
    data = np.load(f"FigS6_data/trajectory_example_plot_data_id_{file_id}.npz")
    x_break = data["x_break"]
    y_break = data["y_break"]
    obs_or_not_x2 = data["obs_or_not_x2"]
    obs_or_not_left_x2 = data["obs_or_not_left_x2"]

    size = 1  # for plotting purposes of the cell wall unit size on screen
    
    plot_wall(obs_or_not_x2, multi_res * W, multi_res * h, size / multi_res, ax, 'black')
    
    multi_res = 2  # obs_or_not_left_x2 = wall_higher_resolution(obs_or_not_left, W, h, multi_res)
    plot_wall(obs_or_not_left_x2, multi_res * W, multi_res * h, size / multi_res, ax, 'black', (-h-W)*multi_res)
    
    ax.plot([W + size / 2, W + size / 2], [0, h], linestyle='--', color='blue', linewidth=2)
    ax.plot([-L-W + size / 2, -L-W + size / 2], [0, h], linestyle='--', color='blue', linewidth=2)

    t = np.arange(x_break.size)
    pos = np.vstack([x_break, y_break]).T
    lc = plot_time_colored_trajectory(pos, t, ax, cmap="tab10")
    
    # x and y axis tick labels
    set_x_y_labels(ax)
    ax.tick_params(axis='x', labelrotation=90)

    ax.set_title(subplot_titles[file_id])
    
    
# plt.subplots_adjust(wspace=0.0)  # smaller = less space between columns
fig.subplots_adjust(hspace=0.2)  # adjust the number until it looks right

plt.tight_layout(rect=[0.1, 0, 1, 1])  # leave space on left for colorbar

# Add colorbar linked to the entire figure
cbar = fig.colorbar(lc, ax=axs, location='left', fraction=0.05, pad=0.05)
cbar.set_label("Relative time", fontsize=global_font_size, labelpad=10)
# Get min/max of time vector
t_min, t_max = t.min(), t.max()
# Example: 11 ticks (0%, 10%, ..., 100%)
num_ticks = 11
real_time = np.linspace(t_min, t_max, num_ticks)
cbar.set_ticks(real_time)
# Normalized labels 0–1
ticks = np.linspace(0, 1, num_ticks)
cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
# Change font size of tick labels
for tick in cbar.ax.get_yticklabels():
    tick.set_fontsize(global_font_size)   # desired font size

############################


# plt.imshow(data, rasterized=True)
plt.show()