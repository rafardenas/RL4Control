#Created by Rafael Cardenas

import gym
import sys
import pickle
import numpy as np



env = gym.make('FrozenLake-v0') #if is_slippery: the environment is stochastic

#Value table

v_table = np.zeros(env.env.nS)
a_table = np.zeros(env.env.nS, dtype=int)

def value_iteration(state, gamma, V_past):
    """
    Compute the optimal state value for each state
    """
    V_past = V_past.flatten()
    action_values = []
    for trans in state:
        v_star = 0
        for action in trans:
            v_star += action[0] * (action[2] + gamma * V_past[action[1]])
        action_values.append(v_star)

    V_star = np.max(action_values)
    A_star = int(np.argmax(action_values))
    return V_star, A_star

def optimal_policy(a_table):
    optimal_actions_table = np.zeros(env.env.nS, dtype=str)
    actions = ["<", "v", ">", "^"]
    for i in range(env.env.nS):
        optimal_actions_table[i] = actions[a_table[i]]
    return optimal_actions_table

def policy_evaluation

    


#learning

episodes = 100
gamma = 0.9
eps = 1e-2
dif = 1000

while dif > eps:


    v_table_last = v_table.copy()
    for j in range(env.env.nS):
        state = env.env.P[j].values()
        V_s, A_s = value_iteration(state, gamma, v_table_last)
        v_table[j] = V_s
        a_table[j] = A_s

    dif = np.sum(np.fabs(v_table - v_table_last))

v_table = v_table.reshape(4,-1)
print(v_table)


a_t = optimal_policy(a_table)
a_t = a_t.reshape(4,-1) 
print(a_t)
env.render()

