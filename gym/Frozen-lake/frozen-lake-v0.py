#Created by Rafael Cardenas
#Inspired by: https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

#nice to have:
    #number of iterations until success
    #effect of gamma over the reward value. (plot)
    #does the speed change when implemented with other data structures
    #number of training episodes vs average reward (plot)
    #do the training and then the polikcy eval;uation at the same time



import gym
import sys
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt 



env = gym.make('FrozenLake-v0') #if is_slippery: the environment is stochastic

#Value and actions tables

v_table = np.zeros(env.env.nS)
v_table2 = np.zeros(env.env.nS)
a_table = np.zeros(env.env.nS, dtype=int)
a_table2 = np.zeros(env.env.nS, dtype=int)

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

def value_iteration2(state, gamma, V_past):
    """
    Compute the optimal state value for each state, but summing the value over all the possible actions.
    """
    V_past = V_past.flatten()
    action_values = []
    for trans in state:
        v_star = 0
        for action in trans:
            v_star += action[0] * (action[2] + gamma * V_past[action[1]]) #action 0 = probability of ending in state s'
        action_values.append(v_star)                                      #action[2] = reward r(s,a,s') remember that it is possible to end up in the same state
                                                                          #action[1] = next state (s')
    V_star = sum(action_values)
    A_star = int(np.argmax(action_values))
    return V_star, A_star


def optimal_policy(a_table):
    """
    Extracts optimal policy in terms of actions defined by the env
    """
    optimal_actions_table = np.zeros(env.env.nS, dtype=str)
    actions = ["<", "v", ">", "^"]
    for i in range(env.env.nS):
        optimal_actions_table[i] = actions[a_table[i]]
    return optimal_actions_table

def run_episode(policy, gamma = 1, render = False):
    """
    render: if true: show the steps the agent took
    Runs an episode of the environment using the optimal policy
    """
    step_idx = 0
    tot_reward = 0
    obs = env.reset()
    while True:
        if render:
            env.render()
        obs, rew, done, _ = env.step(policy[obs])
        tot_reward += (gamma ** step_idx * rew)
        step_idx += 1
        if done:
            break
    return tot_reward

def policy_evaluation(policy, episodes, render = False):
    """
    Run episodes for n times and compute their reward
    Returns: 
    average total reward
    """
    rewards_array = [run_episode(policy, gamma, render) for _ in range(episodes)]

    return np.mean(rewards_array)

#Learning and initial params

episodes = 100
eval_episodes =  1
gamma = 0.9
eps = 1e-3
dif = 1000

start_time = time.clock()

while dif > eps:
    v_table_last = v_table.copy()
    v_table2_last = v_table2.copy()
    for j in range(env.env.nS):
        state = env.env.P[j].values()
        V_s, A_s = value_iteration(state, gamma, v_table_last)
        V_s2, A_s2 = value_iteration2(state, gamma, v_table2_last)
        v_table[j], v_table2[j] = V_s, V_s2
        a_table[j], a_table2[j] = A_s, A_s2
    dif = np.sum(np.fabs(v_table - v_table_last))
print(time.clock() - start_time, "seconds")         #Printing the elapsed time for the convergence of the Value iteration alg

print(v_table.reshape(4,-1))                        #optimal values
print(v_table2.reshape(4,-1))
print(optimal_policy(a_table).reshape(4,-1))        #optimal actions
print(a_table)                                      #optimal actions (by name according to env)
print(policy_evaluation(a_table, eval_episodes, render=True))      #Average return for n episodes


plt.matshow(v_table.reshape(4,-1), cmap='cool')     #Color map for value function
plt.colorbar()
plt.show()
