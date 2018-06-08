import numpy as np
from env.cliff_walking import CliffWalking
import itertools
from collections import defaultdict,namedtuple
from matplotlib import pyplot as plt
import pandas as pd
Stats=namedtuple("Stats",["episode_length","episode_rewards"])

plt.style.use('ggplot')
class Sarsa(object):
    def __init__(self,num_episodes,num_actions,epsilon,gamma):
        #setting class variables
        self.q=defaultdict(lambda:np.random.rand(num_actions))

        self.epsilon=epsilon
        self.gamma=gamma
        self.num_actions=num_actions
        self.num_episodes=num_episodes
        self.env=CliffWalking(self.num_actions)

    def policy(self,state):
        A=np.ones(self.num_actions)*self.epsilon/self.num_actions
        A[np.argmax(self.q[state])]+=1-self.epsilon
        return A

    def learner(self,alpha=0.5):
        stats=Stats(episode_length=np.zeros(self.num_episodes),
        episode_rewards=np.zeros(self.num_episodes))

        for episode in range(self.num_episodes):

            state=self.env.reset()
            action_prob=self.policy(state)
            action=np.random.choice(np.arange(self.num_actions),p=action_prob)

            for i in itertools.count():

                next_state,reward,done=self.env.step(action)
                next_action_prob=self.policy(next_state)
                next_action=np.random.choice(np.arange(self.num_actions),p=next_action_prob)

                #TD error
                td_error=reward+self.gamma*self.q[next_state][next_action]-self.q[state][action]

                # TD update
                self.q[state][action]+=td_error*alpha

                stats.episode_length[episode]=i
                stats.episode_rewards[episode]+=reward

                if done:
                    break
                state=next_state
                action=next_action
        return self.q,stats

    def plot(self,s):
        fig=plt.figure(figsize=(12,7))
        plt.plot(s.episode_length)
        plt.xlabel('episode count')
        plt.ylabel(' episode_length')
        plt.title('Episode length')
        plt.show(fig)
        fig=plt.figure(figsize=(10,7))
        rewards=pd.Series(s.episode_rewards).rolling(10,min_periods=10).mean()
        plt.plot(rewards)
        plt.xlabel('episode count')
        plt.ylabel('reward')
        plt.title('Episode reward')
        plt.show(fig)


num_episodes=200
num_actions=4
epsilon=0.1
gamma=1.0
sarsa=Sarsa(num_episodes,num_actions,epsilon,gamma)
Q,stats=sarsa.learner()
sarsa.plot(stats)
