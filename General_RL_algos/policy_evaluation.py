import mdp
import numpy as np
import plotting

def policy_eval_two_arrays():
    state_count=mdp.get_state_count()
    gamma=0.9
    theta=0.001 ##minimum value of delta
    delta_values=[]
    V = state_count*[0]
    while(True):
        delta=0
        V_t1=state_count*[0]
        for state in range(state_count):
            for actions in mdp.get_actions(state):
                next_state,reward,prob=mdp.get_state_transition(state,actions)
                V_t1[state]+=prob*(reward+gamma*V[next_state])
            delta=max(delta,abs(V_t1[state]-V[state]))
        delta_values.append(delta)
        V=V_t1
        if(delta<theta):
            break
    plotting.plot_values(delta_values,"step-updating")
    return V
def inline_updates():
    state_count=mdp.get_state_count()
    gamma=0.9
    theta=0.001 ##minimum value of delta
    delta_values=[]
    V = state_count*[0]
    while(True):
        delta=0
        for state in range(state_count):
            v=0;
            for actions in mdp.get_actions(state):
                next_state,reward,prob=mdp.get_state_transition(state,actions)
                v+=prob*(reward+gamma*V[next_state])
            delta=max(delta,abs(V[state]-v))
            V[state]=v
        delta_values.append(delta)

        if(delta<theta):
            break
    plotting.plot_values(delta_values,"Inline")
    return V

def main():
    V=policy_eval_two_arrays()
    a=np.append(V,0)
    a.reshape(4,4)
    inline_updates()

if __name__=="__main__":
    main()
