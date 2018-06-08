import numpy as np
import sys
from gym.envs.toy_text import discrete

up=0
down=1
right=2
left=3
class CliffWalking(discrete.DiscreteEnv):
    def lim_coordinates(self,coord):
        coord[0]=min(coord[0],self.shape[0]-1)
        coord[0]=max(coord[0],0)
        coord[1]=min(coord[1],self.shape[1]-1)
        coord[1]=max(coord[1],0)
        return coord

    def calculate_trans_prob(self,current,delta):
        new_position=np.array(current)+np.array(delta)
        new_position=self.lim_coordinates(new_position)
        new_state=np.ravel_multi_index(tuple(new_position),self.shape)
        reward=-100 if self.__cliff[tuple(new_position)] else -1
        isdone=self.__cliff[tuple(new_position)] or tuple(new_position)==(3,11)
        return [(new_state,reward,isdone)]

    def __init__(self,num_action):
        self.shape=(4,12)
        nS=np.prod(self.shape)
        nA=num_action

        self.__cliff=np.zeros(self.shape,dtype=np.bool)
        self.__cliff[3,1:-1]=True

        ## calculating transitions
        P={}

        for s in range(nS):
            P[s]={a:[] for a in range(nA)}
            pos=np.unravel_index(s,self.shape)
            P[s][up]=self.calculate_trans_prob(pos,[-1,0])
            P[s][down]=self.calculate_trans_prob(pos,[1,0])
            P[s][right]=self.calculate_trans_prob(pos,[0,1])
            P[s][left]=self.calculate_trans_prob(pos,[0,-1])

        isd=np.zeros(nS)
        isd[np.ravel_multi_index((3,0),self.shape)]=1.0
        super().__init__(nS,nA,P,isd)

    def reset(self):
        self.s=np.argmax(self.isd)
        return self.s

    def step(self,action):
        self.s = self.P[self.s][action][0][0]
        reward = self.P[self.s][action][0][1]
        done = self.P[self.s][action][0][2]

        #self.s,reward,done=self.P[self.s][action][0]
        return (self.s, reward, done)

    def render(self,mode='human',close=False):
        if close:
            return
        output=StringIO() if mode=='ansi' else sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3,11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")
