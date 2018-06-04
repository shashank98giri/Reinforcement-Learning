import numpy as np
import sys
import setup.plotting as plotting
from matplotlib import pyplot as plt
from matplotlib import pylab
import matplotlib.gridspec as gridspec
from collections import namedtuple
Stats=namedtuple("Stats",["cumulative_reward","regret"])

class Test(object):
    def __init__(self,env,agent):
        self.env=env
        self.agent=agent


    def run_bandit(self,trials=1000,display_frequency=1):

        print("Distribution:",self.env.dst,flush=True)
        print("Rewards {}\n most optimal reward {}".format(self.env.reward,self.env.most_optimal))
        plotting.plt_initial_distibution(self.env.reward)
        stats=Stats(cumulative_reward=np.zeros(trials,dtype=np.longdouble),
        regret=np.zeros(trials,dtype=np.longdouble))

        cumulative_reward=0.0
        regret=0.0

        for i in range(trials):
            action=self.agent.act()

            reward=self.env.step(action)
            self.agent.feedback(action,reward)
            cumulative_reward+=reward
            regret+=self.env.calcute_gap(action)

            stats.cumulative_reward[i]=cumulative_reward
            stats.regret[i]=regret
        print("---------------------------------------------------------")
        print("Policy {}\n Arms pull {}\n max arm pull:{}".format(self.agent.name,self.agent.total_counts,
                np.argmax(self.agent.total_counts)),flush=True)
        plotting.plt_estimate(stats)
