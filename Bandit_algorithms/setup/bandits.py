import numpy as np
import sys

class Environment(object):
    def __init__(self,num_actions,distribution="Bernoulli",evaluation_speed=548):
        self.dst=distribution
        np.random.seed(evaluation_speed)
        self.num_actions=num_actions
        if(distribution=="Bernoulli"):
            self.reward=np.random.rand(num_actions)
            self.most_optimal=np.argmax(self.reward)
        else:
            sys.exit(0)

    def step(self,action):
        reward=0.0
        if(self.dst=="Bernoulli"):
            reward=np.random.binomial(1,self.reward[action])
        else:
            sys.exit(0)
        return reward

    def calcute_gap(self,action):
        return np.absolute(self.reward[self.most_optimal]-self.reward[action])
