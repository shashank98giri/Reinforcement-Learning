import numpy as np

def get_state_count():
    return 15

def get_actions(state):
    actions=["up","down","left","right"]
    return actions

def get_state_transition(state,action):
    next_state=map_value[(state,action)]
    reward=0 if next_state == 0 else -1
    prob=0.25
    return (next_state,reward,prob)

def create_transition_map():
    map_value={}
    map_value[(0,"right")]=0
    map_value[(0,"left")]=0
    map_value[(0,"down")]=0
    map_value[(0,"up")]=0

    map_value[(1,"right")]=2
    map_value[(1,"left")]=0
    map_value[(1,"down")]=5
    map_value[(1,"up")]=1

    map_value[(2,"right")]=3
    map_value[(2,"left")]=1
    map_value[(2,"down")]=6
    map_value[(2,"up")]=2

    map_value[(3,"right")]=3
    map_value[(3,"left")]=2
    map_value[(3,"down")]=7
    map_value[(3,"up")]=3

    map_value[(4,"right")]=5
    map_value[(4,"left")]=4
    map_value[(4,"down")]=8
    map_value[(4,"up")]=0

    map_value[(5,"right")]=6
    map_value[(5,"left")]=3
    map_value[(5,"down")]=9
    map_value[(5,"up")]=1

    map_value[(6,"right")]=7
    map_value[(6,"left")]=5
    map_value[(6,"down")]=10
    map_value[(6,"up")]=2

    map_value[(7,"right")]=7
    map_value[(7,"left")]=6
    map_value[(7,"down")]=11
    map_value[(7,"up")]=3

    map_value[(8,"right")]=9
    map_value[(8,"left")]=8
    map_value[(8,"down")]=12
    map_value[(8,"up")]=4

    map_value[(9,"right")]=10
    map_value[(9,"left")]=8
    map_value[(9,"down")]=13
    map_value[(9,"up")]=5

    map_value[(10,"right")]=11
    map_value[(10,"left")]=9
    map_value[(10,"down")]=14
    map_value[(10,"up")]=6

    map_value[(11,"right")]=12
    map_value[(11,"left")]=10
    map_value[(11,"down")]=0
    map_value[(11,"up")]=7

    map_value[(12,"right")]=13
    map_value[(12,"left")]=12
    map_value[(12,"down")]=12
    map_value[(12,"up")]=8

    map_value[(13,"right")]=14
    map_value[(13,"left")]=12
    map_value[(13,"down")]=13
    map_value[(13,"up")]=9

    map_value[(14,"right")]=0
    map_value[(14,"left")]=13
    map_value[(14,"down")]=14
    map_value[(14,"up")]=10

    return map_value

map_value=create_transition_map()
