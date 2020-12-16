#Dynamic Programming Agents

import numpy as np
import gym

class DP_Agent(object):
    
    def __init__(self, env, v_table, gamma, policy):
        """Agent based on DP algorithms
        args:
        env: gym environment
        v_table = value functions per state
        gamma: gamma
        policy: policy, start with random
        """
        self.env = env
        self.v_table = v_table  
        self.gamma = gamma
        self.policy = policy


    def policy_evaluation(self):
        """
        Evaluate the current policy
        """
        eps = 1e-1
        idx = 0
        stable_value = False
        while True:
            idx += 1
            v_table_anterior = self.v_table.copy()
            delta = 0
            backup_v = []
            for state in range(self.env.env.nS):
                v = 0
                for poss_a in self.policy[state]:
                    act = self.env.env.P[state][poss_a]
                    for blow in act:
                        v += blow[0] * (blow[2] + self.gamma * v_table_anterior[blow[1]])
                backup_v.append(self.gamma * v_table_anterior[blow[1]])

                self.v_table[state] = v
            delta = max(delta, sum(np.fabs(self.v_table - v_table_anterior)))
            if np.max(backup_v) > 1e3 : break
            if delta < eps:
                stable_value = True
                break
        return stable_value


    def policy_improvement(self):
        """
        Improves the current policy by choosing the action(s) that maximize the action value for every state
        """
        stable_policy = True
        old_p = self.policy.copy()
        self.policy = {i:[] for i in range(env.observation_space.n)} #start with empty policy
        for i in range(env.env.nS):
            q_sa = [sum([p*(r + self.gamma * self.v_table[ns]) for p, ns, r, _ in self.env.env.P[i][a]]) for a in range(self.env.env.nA)]
            opt_action = np.argmax(q_sa)
            self.policy[i].append(opt_action)

            for j in range(self.env.env.nA):
                if j not in self.policy[i] and q_sa[j] == q_sa[opt_action]: #append one or more actions with the same value
                    self.policy[i].append(j)
                    #check if policy was not the same as previous one
                    if j not in old_p[i]:
                        stable_policy = False
        return stable_policy

    def policy_iteration(self):
        idx = 0
        while True:
            idx += 1
            stable_v = self.policy_evaluation()
            stable_p = self.policy_improvement()
            if stable_v and stable_p:
                print("policy and valye stable")
                print("iteratios to converge: " + str(idx))
                break


    def value_iteration(self):
        """
        Compute the optimal state value for each state
        """
        eps = 1e-1
        max_iter = 10000
        iter = 0
        while True: 
            v_past = self.v_table.copy()

            for state in range(self.env.env.nS):
                v = [sum([p * (r + self.gamma * v_past[ns]) for p, ns, r, _ in self.env.env.P[state][a]]) for a in range(self.env.env.nA)]
                self.v_table[state] = np.max(v)        
                self.policy[state] = np.argmax(v)

            if np.sum(np.fabs(self.v_table - v_past)) < eps:
                break
            iter += 1
            if iter > max_iter:
                break
        self.v_table = self.v_table.reshape(4,-1)
        self.iterations = iter


env = gym.make('FrozenLake-v0')
v_table = np.zeros(env.env.nS, dtype=np.float)
policy = dict.fromkeys(range(env.observation_space.n), [0]) #random policy in dictionary, faster!

policy = {i:[np.random.choice(env.env.nA)] for i in range(env.observation_space.n)} #random policy in dictionary, faster!
gamma = 0.9

iter = 1  #number of episodes
agent = DP_Agent(env, v_table, gamma, policy) #summon agent :)
"""agent.policy_evaluation()
agent.policy_improvement()"""
for i in range(iter):
    agent.policy_iteration()    #policy iteration


print(agent.v_table.reshape(4,-1))
print(agent.policy)
   