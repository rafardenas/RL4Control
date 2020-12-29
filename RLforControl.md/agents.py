#RL for control, agents
#27.12.2020

import numpy as np
from collections import defaultdict
from itertools import product


#base class for all the agents
#state-action pairs are keys, values of the dictionay are (of course) the values of the actions (as list)

"""to define:  
self.Q_table --pending to revise--
algorithm
"""

class Agent:
    r"""
    Base class for all the RL agents
    
    Params:
        table(dict): State- or State-Action value function in tabular form
        disc(float): discount factor used for the return
        lr(float): learning rate
        movements(int): number of possible control actions (used for indexing purposes aka steps)
    """

    def __init__(self, movements, num_actions, modulus, lr = 1, disc1 = 0.85):
        self.movements = movements
        self.num_actions = num_actions
        self.modulus = modulus
        self.lr = lr
        self.disc1 = disc1



    def act(self, table, state, eps, s):
        r"""
        Act accordingly to state as argument for policy and using e-greedy strategy for exploration. 
        This strategy is implemented as described in page 101 (S&B).
        """
        p = np.random.uniform()
        #print("from state: {}".format(state))
        #print(list(self.d.items())[0])
        probs = np.ones(self.num_actions) * (eps / int(self.num_actions)) #values are the values of the dictionary
        #print(eps)
        A = np.argmax(table[state])
        #print("no key error")
        #print(A)
        probs[A] = 1 - eps + (eps / int(self.num_actions))
        #print(probs)
        action = np.random.choice(self.num_actions, p = probs)

        return action
    
    def learned(self, table):
        dictionary = table
        return dictionary 



    

class Monte_Carlo(Agent):
    """
    Monte Carlo agent class
    Params:
        state_UB(list): Upper bund of the state values 
        disc2(float): discount factor for terminal state for the incremental implementation
    Attr:
        dcount: dictionary with the number of times a state has been visited
        d: dictionary with state values
    """
    def __init__(self, movements, num_actions, modulus, state_UB, lr = 1, disc1 = 0.85, disc2 = 0.9):
        super().__init__(movements, num_actions, modulus, disc1 = disc1, lr = 1)
        self.state_UB = state_UB
        self.disc2 = disc2
        

        fs = [str(i) for i in modulus]
        decimals = [f[::-1].find('.') for f in fs]

        holder = {i:[] for i in range(len(state_UB))} #one key for every state variable, i.e. variables in the states will be indexed as were inputed
        for i in range(len(state_UB)): #create the unidimensional mesh grid for every state variable
            holder[i] = np.linspace(0, int(self.state_UB[i]), int(self.state_UB[i]/self.modulus[i] + 1), dtype = np.float64)
            holder[i] = np.round(holder[i], decimals[i])
        #print(holder)
        discrete_states = tuple([v for v in holder.values()]) #tuple with the arrays of mesh grids
        p = list(product(*discrete_states, range(1, int(self.movements) + 1))) #list of outer product of all the discrete states from the mesh
        #p = list(product(*discrete_states)) #list of outer product of all the discrete states from the mesh

        self.d = {i:np.random.randn(self.num_actions) for i in p} #initialisind random values for all the states, hope it works ;)
        self.dcount = {i:np.zeros(self.num_actions, dtype=int) for i in p}

    def printd(self, key = None):
        if key != None:
            print(self.d[key])
        
        else:
            for i in range(200):
                print(list(self.d.keys())[i + 15000])

    def Incremental(self, state, action, reward):
        r"""
        On policy Monte Carlo control, incremental implementation
        -------
        Params:
            state(tuple): current state of the system
            action(tuple): action(s) taken
            reward(tuple): reward(s) 
        """
        
        #print(reward)
        #print(state)
        #print(action)
        for i in range(self.movements[0] - 1, -1, -1):
            #print(tuple(state[i])) 
            #print("i: " + str(i))
            idx = action[i]
            
            Gt = 0
            self.dcount[tuple(state[i])][int(idx)] += 1

            #if i > self.movements[0] - 1:
            if i < 1:
                Gt += self.disc2 * reward[-1] 
            else:
                #for j in range(len(reward) - 1, len(reward) - i - 1, -1):
                for j in range(1, i + 2):
                    #print(reward[-j])
                    Gt = reward[-j] + self.disc1 * Gt
               

            #print(Gt)
            #print(idx)
            alpha = self.lr / self.dcount[tuple(state[i])][int(idx)]              # updating learning parameter with number of visits to current state
            V = self.d[tuple(state[i])][int(idx)]
            self.d[tuple(state[i])][int(idx)] = (1-alpha) * V + alpha * Gt
            #print(self.d[tuple(state[i])])
        return

    def First_Visit(self, state, action, reward):
        r"""
        Monte Carlo First Visit with e-greedy policy, implemented as S&B pp. 101
        -------
        Params:
            state(tuple): current state of the system
            action(tuple): action(s) taken
            reward(tuple): reward(s) 
        """

        states_actions = [(tuple(state[i]), int(self.movements - i)) for i in range(len(action))]
      
        for s_a in set(states_actions):
            idx_sa = [i for i,s in enumerate(states_actions) if s_a == s]
            time_to_term = int(self.movements - idx_sa[0])
            gt = sum([(reward[-i] * self.disc1 ** (i-1)) for i in range(1, time_to_term + 1)])
            self.dcount[(s_a[0][0], s_a[0][1], s_a[1])][int(action[idx_sa[0]])] += 1
            ns = self.dcount[(s_a[0][0], s_a[0][1], s_a[1])][int(action[idx_sa[0]])]
            self.d[(s_a[0][0], s_a[0][1], s_a[1])][int(action[idx_sa[0]])] = gt / ns

        return 

            
            
            
