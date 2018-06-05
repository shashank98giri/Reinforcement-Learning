import numpy as np
import sys
from setup.bandits import Environment
from setup.testing import Test
import math

class Policy(object):
    def __init__(self,actions):
        self.num_actions=actions
    def act(self):
        pass
    def feedback(self,action,reward):
        pass

class Greedy(Policy):
    def __init__(self,num_actions,**kwargs):
        super().__init__(num_actions)
        self.total_counts=np.zeros(self.num_actions)
        self.name="Greedy"
        self.total_reward=np.zeros(self.num_actions)
    def act(self):
        current_track=np.divide(self.total_reward,self.total_counts,where=self.total_counts>0)
        current_track[self.total_counts<=0]=0
        taken_action=np.argmax(current_track)
        return taken_action
    def feedback(self,action,reward):
        self.total_reward[action]+=reward
        self.total_counts[action]+=1

class EpsilonGreedy(Greedy):
    def __init__(self,num_actions,epsilon=0.5,**kwargs):
        super().__init__(num_actions)
        self.name="EpsilonGreedy"
        self.epsilon=epsilon
        if(epsilon > 1 or epsilon<0):
            print("Epsilon should be between 0 and 1",flush=True)
            sys.exit(0)
    def act(self):
        choice=np.random.binomial(1,self.epsilon)
        if(choice==1):
            return np.random.choice(self.num_actions)
        else:
            return super().act()


num_actions=10

if __name__=="main":
    env=Environment(num_actions)
    agent=Greedy(num_actions)
    test=Test(env,agent)
    test.run_bandit()
