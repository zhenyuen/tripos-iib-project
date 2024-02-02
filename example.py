import numpy as np
import matplotlib.pyplot as plt

def visualise_langevin(times, truth_x, truth_y):
    pos_x = [s[0, :] for s in truth_x]
    vel_x = [s[1, :] for s in truth_x]
    pos_y = [s[0, :] for s in truth_y]
    vel_y = [s[1, :] for s in truth_y]
    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(5, 5))
    ax1.plot(times, pos_x, label='$x$')
    ax1.plot(times, pos_y, label='$y$')
    ax1.set_xlabel("Time, $t$")
    ax1.set_ylabel("Position")
    ax1.legend()
    ax2.plot(times, vel_x, label='$\dot{x}$')
    ax2.plot(times, vel_y, label='$\dot{y}$')
    ax2.set_xlabel("Time, $t$")
    ax2.set_ylabel("Velocity")
    ax2.legend()
    fig.tight_layout()
    return fig

def visualise_process(times, truth_x, truth_y):
    pos_x = [s[0, :] for s in truth_x]
    pos_y = [s[0, :] for s in truth_y]
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax1.plot(times, pos_x, label='$x$')
    ax1.plot(times, pos_y, label='$y$')
    ax1.set_xlabel("Time, $t$")
    ax1.set_ylabel("Position")
    ax1.legend()
    fig.tight_layout()
    return fig

def visualise_singer(times, truth_x, truth_y):
    pos_x = [s[0, :] for s in truth_x]
    vel_x = [s[1, :] for s in truth_x]
    acc_x = [s[2, :] for s in truth_x]
    pos_y = [s[0, :] for s in truth_y]
    vel_y = [s[1, :] for s in truth_y]
    acc_y = [s[2, :] for s in truth_y]
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, figsize=(5, 7.5))
    ax1.plot(times, pos_x, label='$x$')
    ax1.plot(times, pos_y, label='$y$')
    ax1.set_xlabel("Time, $t$")
    ax1.set_ylabel("Position")
    ax1.legend()
    ax2.plot(times, vel_x, label='$\dot{x}$')
    ax2.plot(times, vel_y, label='$\dot{y}$')
    ax2.set_xlabel("Time, $t$")
    ax2.set_ylabel("Velocity")
    ax2.legend()
    ax3.plot(times, acc_x, label='$\ddot{x}$')
    ax3.plot(times, acc_y, label='$\ddot{y}$')
    ax3.set_xlabel("Time, $t$")
    ax3.set_ylabel("Acceleration")
    ax3.legend()
    fig.tight_layout()
    return fig

visualise_erv = visualise_langevin