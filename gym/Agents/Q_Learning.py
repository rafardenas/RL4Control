#Model Free RL: Q-Learning
#Mountain Car
# actions: 0,1,2 = left, nothing, right
# obs: [position (-1.2 to 0,6), velocity (-0.07 to 0.07)].
# done if: position > 0.5 && episode lenght >200
# reward: 1 if flag reached, -1 if position < 0.5; 0 otherwise


import gym
import numpy as np
from itertools import product #trick




class Q_agent():
    def __init__(self, env, gamma, states, qsa_pairs, policy, eps=1/3): #env?
        self.env = env
        self.eps = eps
        self.gamma = gamma
        self.N = 0 #decay rate for learning rate (may be fancier)
        self.states = states
        self.qsa_pairs = qsa_pairs
        self.policy = policy
        self.total_reward = 0

    def state_discretizer(self, s, v): #going to be inefficient
        for i in self.states:
            if not s >= i[0]:
                s = i[0]
                break
        if v >= 0:
            v = 1
        else:
            v = 0

        return tuple((s, v))

    def learn(self, iterations):
        s = self.env.reset()
        s = self.state_discretizer(s[0], s[1])
        self.N = 0
        #print(s) 
        
        
        for i in range(iterations):
            self.env.render()
            self.N += 1
            alpha = 1 / self.N
            action = self.policy[s]
            #print(action)
            ns, r, done, _ = self.env.env.step(action) #act and observe
            p = np.random.uniform()
            ns = self.state_discretizer(ns[0], ns[1]) #push the obs into an interval between states
            #print(ns) 

            if p > self.eps: #exploit
                self.qsa_pairs[s][action] = (1 - alpha) * self.qsa_pairs[s][action] + alpha * (r + self.gamma * self.qsa_pairs[ns][np.argmax(self.qsa_pairs[s])]) 
                self.policy[s] =  np.argmax(self.qsa_pairs[s]) #replacing action for best current action
            else: #explore
                self.qsa_pairs[s][action] = (1 - alpha) * self.qsa_pairs[s][action] + alpha * (r + self.gamma * self.qsa_pairs[ns][np.random.choice(3)]) #select random action

            self.total_reward += self.gamma ** self.N * r

            if i % 100 == 0:
                print("Iteration number {}, current reward: {}".format(i,self.total_reward))
            s = ns
        print(alpha)
        env.close() 


        def run_episode(self, env, policy):
            """
            Run episode with learned policy
            """
            s = self.env.reset()
            s = self.state_discretizer(s[0], s[1])
            while not done:
                ns, r, done, _ = env.env.step()


env = gym.make("MountainCar-v0")
gamma = 0.9

positions = np.linspace(env.env.min_position, env.env.max_position, 8) #discretise positions
states = list(product(positions, [0,1])) #tuples: (discretised positions, neg or positive vel)
qsa_pairs = {i:[0 for i in range(3)] for i in states}

#another alpha (recall that it has to tend to 0 in the limit):


policy = {i:np.random.choice(3) for i in states} #random initial policy     
MrQ = Q_agent(env, gamma, states, qsa_pairs, policy, eps=1/3)

for i in range(50):
    MrQ.learn(500)


run_episode(env, policy)


print(policy)
print("\n")
print("\n")
print(qsa_pairs)
