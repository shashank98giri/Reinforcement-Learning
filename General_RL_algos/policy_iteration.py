import mdp
import numpy as np
import plotting
import  policy_evaluation
def policy_iteration():
    state_count=mdp.get_state_count()
    gamma=0.9
    theta=0.001
    V=state_count*[0]
    pi=state_count*[0]
    for state in range(state_count):
        pi[state]=np.random.choice(mdp.get_actions(state))
    unstable=True

    while(unstable):
        while(True):
            delta=0
            for state in range(state_count):
                next_state,reward,prob=mdp.get_state_transition(state,pi[state])
                v=prob*(reward+gamma*V[next_state])
                delta=max(delta,abs(V[state]-v))
                V[state]=v

            if(delta<theta):
                break
        unstable=False

        for state in range(state_count):
            value_functio=[]
            for actions in mdp.get_actions(state):
                next_state,reward,prob=mdp.get_state_transition(state,actions)
                value_functio.append(prob*(reward+gamma*V[next_state]))
            next_optimal_action=mdp.get_actions(state)[np.argmax(value_functio)]
            if(next_optimal_action!=pi[state]):
                unstable=True
            pi[state]=next_optimal_action


    return V,pi

def main():
    V,pi=policy_iteration()
    print(V)
    print(pi)

if __name__=="__main__":
    main()
