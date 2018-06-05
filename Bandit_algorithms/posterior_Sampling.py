import numpy as np
import sys
from setup.bandits import Environment
from setup.testing import Test
import math
from Bandit_algorithms.greedy import Policy

class Posterior(Policy):
    def __init__(self,num_actions):
        super().__init__(num_actions)
        self.name="Posterior Sampling"
        self.num_actions=num_actions
        self.total_counts=np.zeros(num_actions,dtype=np.int)
        self.total_success=np.ones(self.num_actions,dtype=np.float32)
        self.total_failures=np.ones(self.num_actions,dtype=np.float32)

    def act(self):
        beta=np.add(self.total_failures,self.total_success)
        mean=np.divide(self.total_success,beta)
        arm=np.argmax(mean)
        return arm

    def feedback(self,action,reward):
        self.total_failures[action]+=1-reward
        self.total_success[action]+=reward
        self.total_counts[action]+=1

num_actions=10
if __name__=="main":
    env=Environment(num_actions)
    agent=Posterior(num_actions)
    test=Test(env,agent)
    test.run_bandit()
