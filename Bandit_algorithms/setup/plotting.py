from matplotlib import pyplot as plt
from collections import namedtuple
import numpy as np

def plt_initial_distibution(y,hide=False):
    fig=plt.figure(figsize=(5,3))
    x=range(len(y))
    plt.bar(x,y,1/1.6)
    plt.xlabel("Arm")
    plt.ylabel("probality distribution")
    if(hide==False):
        plt.show(fig)
    else :
        plt.close(fig)
    return fig

def plt_estimate(stats,hide=False):
    fig1=plt.figure(figsize=(6,4))
    plt.plot(stats.cumulative_reward)
    plt.xlabel("Timestep")
    plt.ylabel("cumulative_reward")

    if(hide==False):
        plt.show(fig1)
    else :
        plt.close(fig1)
    fig2=plt.figure(figsize=(6,4))
    plt.plot(stats.regret)
    plt.xlabel("Timestep")
    plt.ylabel("regret")

    if(hide==False):
        plt.show(fig2)
    else :
        plt.close(fig2)
    return fig1,fig2
