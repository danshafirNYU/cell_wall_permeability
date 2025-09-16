#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 20:45:58 2025

This function simulates the random walk of particles of varying sizes 
in an environment with obstacles, representing a "cell wall." 
The obstacle field `obs_or_not` is initially generated at a base 
resolution of 1×1 and reused for all particle sizes.

- For a 1×1 particle, each step corresponds to one unit cell in the obstacle grid.
- For larger particles (e.g., 2×2, 3×3), the particle still moves by one unit per step,
  but the wall is upscaled in resolution accordingly (e.g., doubled for a 2×2 particle).
  This ensures that the particle moves through a finer version of the same environment.
- The simulation tracks the bottom-right corner of the particle to determine its position.
- Since the space is refined by a factor equal to the particle size, the number of steps
  is proportionally increased (e.g., 2× more steps for a 2×2 particle).
- For plotting, the trajectory is downscaled by 1 / particle_size to restore the original
  physical scale:
    - Each step appears smaller (e.g., ½ per move for a 2×2 particle).
    - Each obstacle block is visually resized (e.g., ½ × ½ instead of 1 × 1).
- The trajectory is plotted using the center of the particle for better visual representation.

@author: danshafir
"""

import os
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


def plot_time_colored_trajectory(pos, t, ax, h, cmap="viridis", lw=2):
    """
    Plot a trajectory with color corresponding to absolute simulation time,
    handling periodic jumps in y by breaking the line.

    Parameters
    ----------
    pos : np.ndarray
        Array of shape (N, 2) containing x,y positions.
    t : np.ndarray
        Array of shape (N,) containing times corresponding to positions.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    h : float
        Periodic box height (y in [0, h)).
    cmap : str, optional
        Colormap name (default 'viridis').
    lw : float, optional
        Line width (default 2).
    """
    if pos.shape[0] != len(t):
        raise ValueError("pos and t must have the same length")

    # Break jumps in periodic y
    x, y = pos[:,0], pos[:,1]
    x_plot, y_plot = break_y_periodic_jumps(x, y, h)
    
    # Times need to be aligned with NaNs inserted
    t_plot = []
    j = 0
    for i in range(len(x_plot)):
        if np.isnan(x_plot[i]):
            t_plot.append(np.nan)
        else:
            t_plot.append(t[j])
            j += 1
    t_plot = np.array(t_plot)

    # Build segments between valid points (ignoring NaNs)
    mask = ~(np.isnan(x_plot) | np.isnan(y_plot))
    points = np.array([x_plot[mask], y_plot[mask]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Filter t values to match segment count
    t_valid = t_plot[mask]
    t_mid = (t_valid[:-1] + t_valid[1:]) / 2  # color by midpoint time

    # Normalize color scale
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t_mid)
    lc.set_linewidth(lw)

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal", "box")

    return lc

def plot_time_colored_trajectory_old(pos, t, ax, cmap="viridis", lw=2):
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

def sim(W, L, h, F, particle_size, mesh_size, obs_or_not, obs_or_not_left, steps_increments_generation, N_max):
    """
    Simulate a 2D biased / or non-biased random walk until x >= W or until N_max steps are taken.

    Parameters:
    - W: int, target x-coordinate to reach (or thickness of the wall).
    - F: float, bias in x-direction.
    - steps_increments_generation: int, number of steps generated each check to see if we crossed W.
    - N_max: int, maximum number of steps before giving up.
    - p_occupation: float, site occupation probability (currently unused).

    Returns:
    - int: step index at which x >= W was first reached, or -1 if not reached within N_max steps.
    """
    # set up the enviornment
    # h = 1001 - for particle size = 1
    # Left to (and including) x = 0 we have an empty lattice. right to x = 0 we have the wall which is obstructed.
    # Periodic boundary conditions at y = 0 and y = h, and reflective at x = - L
    # The thickness of the wall is w
    
    #  everything scales by particle size
    N_max = mesh_size * N_max
    steps_increments_generation = steps_increments_generation * mesh_size
    W = W * mesh_size
    h = h * mesh_size
    L = L * mesh_size # cell_half_width
    
    
    # Transition probabilities
    B = 1 / (np.exp(F / 2) + np.exp(-F / 2) + 2)
    p_left = B * np.exp(-F / 2)
    p_right = B * np.exp(F / 2)
    p_up = B
    p_down = B

    rng = np.random.default_rng()
    
    # ------------------------------------------------------------
    
    exit_while = False
    # We start somewhere in the left box of size L * h 
    x0 = -rng.integers(0, L)
    y0 = rng.integers(0, L)
    position = np.array([x0, y0])
    total_steps = 0
    
    x_vec = np.zeros(N_max + 1)
    y_vec = np.zeros(N_max + 1)
    x_vec[0] = x0
    y_vec[0] = y0


    while total_steps < N_max:
        batch_size = min(steps_increments_generation, N_max - total_steps)
        steps = rng.choice([[1, 0], [-1, 0], [0, 1], [0, -1]],
                           size=batch_size,
                           p=[p_right, p_left, p_up, p_down])
        
        for step in steps:
            x, y = position + step
            
            # check for periodic boundary conditions in y direction
            if y > h - 1:
                y = y - h
            if y < 0:
                y = y + h
            
            x_left = x - particle_size + 1 # left edge of the particle
            # Left Wall interactions
            if - L - W < x_left < - L + 1:
                x0 = x - 1 + W + L  # right x
                y0 = y      # bottom y
                x1 = x0 - particle_size + 1  # left x
                y1 = y + particle_size - 1   # top y
                
                blocked = False
                
                if step[0] > 0:  # moving right
                    if x0 < W:
                        for j in range(particle_size):
                            if obs_or_not_left[x0, (y0 + j) % h] > 0:
                                blocked = True
            
                elif step[0] < 0:  # moving left
                        for j in range(particle_size):
                            if obs_or_not_left[x1, (y0 + j) % h] > 0:
                                blocked = True
                
                elif step[1] > 0:  # moving up
                    for i in range(particle_size):
                        if x1 + i < W:
                            if obs_or_not_left[x1 + i, y1 % h] > 0:
                                blocked = True
            
                elif step[1] < 0:  # moving down
                    for i in range(particle_size):
                        if x1 + i < W:
                            if obs_or_not_left[x1 + i, y0] > 0:
                                blocked = True
            
                if blocked:
                    total_steps += 1
                    x_vec[total_steps], y_vec[total_steps] = position
                    continue

            # Right wall interactions
            # ------------------------------------------
            if 0 < x < W + 1:
                x0 = x - 1  # bottom-right x
                y0 = y      # bottom-right y
                x1 = x0 - particle_size + 1  # bottom-left x
                y1 = y + particle_size - 1   # top-right y
            
                blocked = False
            
                if step[0] > 0:  # moving right
                    for j in range(particle_size):
                        if obs_or_not[x0, (y0 + j) % h] > 0:
                            blocked = True
            
                elif step[0] < 0:  # moving left
                    if x1 > -1:
                        for j in range(particle_size):
                            if obs_or_not[x1, (y0 + j) % h] > 0:
                                blocked = True
                
                elif step[1] > 0:  # moving up
                    for i in range(particle_size):
                        if x1 + i > -1:
                            if obs_or_not[x1 + i, y1 % h] > 0:
                                blocked = True
            
                elif step[1] < 0:  # moving down
                    for i in range(particle_size):
                        if x1 + i > -1:
                            if obs_or_not[x1 + i, y0] > 0:
                                blocked = True
            
                if blocked:
                    total_steps += 1
                    x_vec[total_steps], y_vec[total_steps] = position
                    continue

            # This is the case for particle_size = 1
            # if 0 < x < W + 1: # are we inside the wall
            #     #  is the target we jump to an obstacle site
            #     if obs_or_not[x - 1, y] > 0:
            #         total_steps += 1
            #         x_vec[total_steps], y_vec[total_steps] = position
            #         continue
            
            # ------------------------------------------
                
            total_steps += 1
            position[0] = x
            position[1] = y
            x_vec[total_steps], y_vec[total_steps] = position
            
            
            
            if position[0] > W or position[0] - particle_size + 1 < - L - W + 1:
                exit_while = True
                break
            
        if exit_while:
            break
            
    return x_vec[0:total_steps+1], y_vec[0:total_steps+1]


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

def wall_higher_resolution(obs_or_not, W, h, res_multi):
    # Takes a wall and enhances the resolution but keeps the same geometry.
    # W, h - dimensions of the original wall (obs_or_not should be W x h)
    # res_multi - integer scaling factor
    new_obs_or_not = np.zeros([res_multi * W, res_multi * h])

    for i in range(W):
        for j in range(h):
            for dx in range(res_multi):
                for dy in range(res_multi):
                    new_obs_or_not[res_multi * i + dx, res_multi * j + dy] = obs_or_not[i, j]
    
    return new_obs_or_not
    
def plot_wall(obs_or_not, W, h, size, ax, color, x_start = 0):
    # The wall section
    # size - size of the lattice unit

    # plot the obstacles
    for i in range(0, W):
        for j in range(0, h):
            if obs_or_not[i,j] > 0:
                ax.add_patch(patches.Rectangle(((x_start + i)* size + 0.5, j * size - 0.5), size, size, color=color))

rng = np.random.default_rng()
p_occupation = 0.4
W = 20
h = 200;
L = 200;
N_steps = int(1e5)
obs_or_not = rng.choice([1,0], size=[W, h], p = [p_occupation, 1 - p_occupation])
particle_size = 2
multi_res = 2
obs_or_not_x2 = wall_higher_resolution(obs_or_not, W, h, multi_res)


fig, ax = plt.subplots()

size = 1
# plot_wall(obs_or_not, W, h, size, ax, 'red')
plot_wall(obs_or_not_x2, multi_res * W, multi_res * h, size / multi_res, ax, 'black')

obs_or_not_left = rng.choice([1,0], size=[W, h], p = [p_occupation, 1 - p_occupation])
multi_res = 2
obs_or_not_left_x2 = wall_higher_resolution(obs_or_not_left, W, h, multi_res)
plot_wall(obs_or_not_left_x2, multi_res * W, multi_res * h, size / multi_res, ax, 'black', (-L-W)*multi_res)

ax.plot([W + size / 2, W + size / 2], [0, h], linestyle='--', color='blue', linewidth=2)
ax.plot([-L-W + size / 2, -L-W + size / 2], [0, h], linestyle='--', color='blue', linewidth=2)

########################
# Optional: Example trajectory
x = np.zeros([N_steps]);
while x.size > N_steps - 1:
    x, y = sim(W, L, h, 0.0, particle_size, multi_res, obs_or_not_x2, obs_or_not_left_x2, 100, N_steps)
x = x / multi_res
y = y / multi_res

x_break, y_break = break_y_periodic_jumps(x, y, h)
# ax.plot(x_break, y_break, color='blue', label='trajectory')
# ax.plot(x[0], y[0], marker='o', color='black', markersize=10)  # marker 'o' = circle
# ax.plot(x[-1], y[-1], marker='o', color='green', markersize=10)  # marker 'o' = circle
#########################

t = np.arange(x_break.size)
pos = np.vstack([x_break, y_break]).T

lc = plot_time_colored_trajectory_old(pos, t, ax, cmap="tab10")

# Add colorbar linked to time
cbar = plt.colorbar(lc, ax=ax, location='left')
# cbar.set_label("Relative time")
cbar.set_label("Relative time", fontsize=global_font_size, labelpad=10)


# Get the min and max of the data (time vector)
t_min, t_max = t.min(), t.max()

# Decide the tick positions you want
# Example: 11 ticks corresponding to 0%, 10%, ..., 100% of the time
num_ticks = 11
real_time = np.linspace(t_min, t_max, num_ticks)

# Set the ticks and labels on the colorbar
cbar.set_ticks(real_time)

# Define ticks: 0, 0.1, 0.2, ..., 1.0
ticks = np.linspace(0, 1, 11)  # 11 values from 0 to 1
cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])

# Change font size of tick labels
for tick in cbar.ax.get_yticklabels():
    tick.set_fontsize(global_font_size)   # desired font size

############################



ax.set_xlim(-L - W - 1, 2 + W)
ax.set_ylim(-2, h + 1)
ax.set_aspect('equal')
# ax.set_title(f'Particle trajectory in an obstructered space, particle of size {multi_res} x {multi_res}. Trajectory tracks center of the paricle.')
# ax.legend()


# Define your new tick positions (numeric positions)
new_ticks_x = [-L - W, -L, -L/2, 0, W]
# Define the labels you want to show
new_labels_x = ["$-R-\\delta$", "$-R$", "$0$", "$R$", "$R+\\delta$"]

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

plt.show()


# Base filename without the ID
base_filename = "trajectory_example_plot_data_id_"

# Find the next available ID
i = 0
while os.path.exists(f"{base_filename}{i}.npz"):
    i += 1


# Save with the available ID
filename = f"{base_filename}{i}.npz"
np.savez(filename,
         x_break=x_break,
         y_break=y_break,
         obs_or_not_x2=obs_or_not_x2,
         obs_or_not_left_x2=obs_or_not_left_x2)

# data = np.load("trajectory_example_plot_data.npz")
# x_break = data["x_break"]
# y_break = data["y_break"]
# obs_or_not_x2 = data["obs_or_not_x2"]
# obs_or_not_left_x2 = data["obs_or_not_left_x2"]