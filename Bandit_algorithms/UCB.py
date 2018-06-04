import numpy as np
import sys
from setup.bandits import Environment
from setup.testing import Test
import math
from Bandit_algorithms.greedy import Policy,Greedy

class UCB(Greedy):
    def __init__(self, num_actions):
        super().__init__( num_actions)
        self.name = "UCB"
        self.round = 0

    def act(self):
        current_action = None
        self.round += 1
        if self.round <= self.num_actions:
            current_action = self.round-1;
        else:
            """At round t, play the arms with maximum average and exploration bonus"""
            expected_arm=np.divide(self.total_rewards,self.total_counts)
            expected_arm=np.add(expected_arm,
                np.divide(np.ones_like(self.total_counts)*math.sqrt(2*math.log(self.round)),self.total_counts))
            current_action = np.argmax(expected_arm)
        return current_action

num_actions=10

if __name__=="main":
    env=Environment(num_actions)
    agent=UCB(num_actions)
    test=Test(env,agent)
    test.run_bandit()
