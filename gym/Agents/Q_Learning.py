#Model Free RL: Q-Learning
#Mountain Car
# actions: 0,1,2 = left, nothing, right
# obs: [position (-1.2 to 0,6), velocity (-0.07 to 0.07)].
# done if: position > 0.5 && episode lenght >200
# reward: 1 if flag reached, -1 if position < 0.5; 0 otherwise


import gym
import numpy as np
from itertools import product #trick
import matplotlib.pyplot as plt


class Q_agent():
    def __init__(self, env, episodes, gamma, states, qsa_pairs, policy, eps=1/4, render = True): #env?
        self.env = env
        self.eps = eps
        self.episodes = episodes
        self.gamma = gamma
        self.N = 0 #decay rate for learning rate (may be fancier)
        self.states = states
        self.qsa_pairs = qsa_pairs
        self.policy = policy
        self.total_reward = 0
        self.render = render
        self.steps = []
        self.finaleps = 0
        self.episode_num = 0
        self.returns = []

    def state_discretizer(self, s, v): #going to be inefficient
        for i in self.states:
            if not s >= i[0]:
                s = i[0]
                break
        for i in self.states:    
            if not v >= i[1]:
                v = i[1]
                break
        return tuple((s, v))

    def eps_decay(self, xi, ei, episodes):
        G = -np.log(0.1) * xi * episodes
        if ei < G:
            behave = np.exp(-ei / (episodes * xi))
        else:
            behave = 0.1
        return behave


    def act(self, state, eps):
        probs = np.ones(len(self.qsa_pairs[state])) * (eps / len(self.qsa_pairs[state]))
        A = np.argmax(self.qsa_pairs[state])
        probs[A] = 1 - eps + (eps / len(self.qsa_pairs[state]))
        action = np.random.choice(len(self.qsa_pairs[state]), p = probs)
        return action
    

    def learn(self, iterations):
        """The inner loop is the actions we take, in other words, the duration of the episode."""
        for e in range(self.episodes):
            s = self.env.reset()
            s = self.state_discretizer(s[0], s[1])
            self.N = 0 
            G = 0
            self.episode_num += 1
            #print(self.episode_num)

            for i in range(iterations):
                #print(s)
                if self.render:
                    self.env.render()
                self.N += 1
                alpha = 1 / self.N
                eps_decayed = self.eps_decay(self.eps, self.episode_num, iterations)
                #print(self.eps)
                action = self.act(s, eps_decayed)
                self.finaleps = eps_decayed
                policy[s] = action
                ns, r, done, _ = self.env.env.step(action) #act and observe
                
                ns = self.state_discretizer(ns[0], ns[1]) #push the obs into an interval between states
                G = r + self.gamma * G         
                p = np.random.uniform()
                if p > eps_decayed: #exploit
                    self.qsa_pairs[s][action] = (1 - alpha) * self.qsa_pairs[s][action] + alpha * (r + self.gamma * self.qsa_pairs[ns][np.argmax(self.qsa_pairs[s])]) 
                    self.policy[s] =  np.argmax(self.qsa_pairs[s]) #replacing action for best current action
                else: #explore
                    self.qsa_pairs[s][action] = (1 - alpha) * self.qsa_pairs[s][action] + alpha * (r + self.gamma * self.qsa_pairs[ns][np.random.choice(3)]) #select random action
                
                if ns[0] >= 0.56:
                    self.steps.append(self.N)
                    #print("success")
                    self.returns.append(G)
                    break

                self.total_reward += self.gamma ** self.N * r

                #if i % 100 == 0:
                #    print("Iteration number {}, current reward: {}".format(i,self.total_reward))
                s = ns
            
            progress = [0.2, 0.4, 0.6, 0.8, 1]
            self.returns.append(G)
            if e / self.episodes in progress:
                print("Training Progress = {} %".format((e / self.episodes)*100))
            env.close() 
        return G

    def run_episode(self, env, policy):
        """
        Run episode with learned policy
        """
        Gt = 0
        s = self.env.reset()
        s = self.state_discretizer(s[0], s[1])
        done = False

        while not done:
            ns, r, done, _ = env.env.step(policy[s])
            Gt += r + self.gamma * Gt
            s = ns
        return Gt


env = gym.make("MountainCar-v0")
gamma = 0.9
steps = 8

positions = np.round((np.linspace(env.env.min_position, env.env.max_position, steps)), 2) #discretise positions
vel = np.round((np.linspace(-env.env.max_speed, env.env.max_speed, steps)), 2) #discretise vel
# # uncomment for consider only position (cross prod) neg or pos velocity as state
#states = list(product(positions, [0,1])) #tuples: (discretised positions, neg or positive vel) 
states = list(product(positions, vel)) #tuples: (discretised positions in intervals, for vel: same setting as positions)
qsa_pairs = {i:[0 for i in range(3)] for i in states}
training_episodes = 100


policy = {i:np.random.choice(3) for i in states} #random initial policy     
MrQ = Q_agent(env, training_episodes, gamma, states, qsa_pairs, policy, eps=1/6, render=False)
expected_retu = []
Gi = MrQ.learn(300) #training method

##TODO: Validation of the agent

steps = np.array(MrQ.steps)
#print(MrQ.returns)

plt.plot(MrQ.returns)
#plt.show()

#print(steps.mean())
#MrQ.run_episode(env, learned_policy)

#run_episode(env, policy)


#print(policy)
#print("\n")
#print("\n")
#print(qsa_pairs)
#print(steps)
