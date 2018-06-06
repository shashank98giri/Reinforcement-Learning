import mdp
import numpy as np
import plotting

def policy_iteration():
    state_count=mdp.get_state_count()
    gamma=0.9
    theta=0.001
    V=state_count*[0]
    pi=state_count*[0]
    cnt=0;
    for state in range(state_count):
        pi[state]=np.random.choice(mdp.get_actions(state))
    unstable=True

    while(unstable):

        while(True):
            cnt+=1
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


    return V,pi,cnt

def value_iteration():
    state_count=mdp.get_state_count()
    gamma=0.9
    theta=0.001
    V=state_count*[0]
    pi=state_count*[0]
    cnt=0
    while(True):
        delta=0
        cnt+=1
        for state in range(state_count):
            v=[]
            for actions in mdp.get_actions(state):
                next_state,reward,prob=mdp.get_state_transition(state,actions)
                v.append(prob*(reward+gamma*V[next_state]))
            most_optimal=np.amax(v,axis=0)
            delta=max(delta,abs(most_optimal-V[state]))
            V[state]=most_optimal
        if delta<theta:
            break
    for state in range(state_count):
        value_functio=[]
        for actions in mdp.get_actions(state):
            next_state,reward,prob=mdp.get_state_transition(state,actions)
            value_functio.append(prob*(reward+gamma*V[next_state]))
        optimal_action=mdp.get_actions(state)[np.argmax(value_functio)]
        pi[state]=optimal_action
    return V,pi,cnt

def main():
    V,pi,cnt=policy_iteration()
    print("Following Policy Iteration Method\nCount:%s\n"%(cnt))
    print(np.append(V,0).reshape(4,4))
    print(np.append(pi,"down").reshape(4,4))
    V,pi,cnt=value_iteration()
    print("\nFollowing Value Iteration\nCount:%s\n"%(cnt))
    print(np.append(V,0).reshape(4,4))
    print(np.append(pi,"down").reshape(4,4))

if __name__=="__main__":
    main()
