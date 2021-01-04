#RL for control, agents
#27.12.2020

import numpy as np
from collections import defaultdict
from itertools import product

#base class for all the agents
#state-action pairs are keys, values of the dictionay are (of course) the values of the actions (as list)


class Agent:
    r"""
    Agent's Base class
    
    Params:
        movements(array): number of possible control actions in one episode  (used also for indexing purposes aka steps)
        num_actions(array): number of available control actions 
        modulus(array): graularity of the state space i.e. the upper bound of the state space divided by the modulus is the number of states
        lr(float): learning rate
        movements(int): number of possible control actions
        disc(float): discount factor gamma
    """

    def __init__(self, movements, num_actions, modulus, learning_algo, lr = 1, disc1 = 0.85):
        self.movements = movements
        self.num_actions = num_actions
        self.modulus = modulus
        self.learning_algo = learning_algo
        self.lr = lr
        self.disc1 = disc1



    def act(self, table, state, eps, s):
        r"""
        Act accordingly to defined policy, use state as argument for policy and using e-greedy strategy for exploration. 
        This strategy is implemented as described in page 101 (S&B).
        """
        probs = np.ones(self.num_actions) * (eps / int(self.num_actions)) #values are the values of the dictionary
        A = np.argmax(table[state])
        probs[A] = 1 - eps + (eps / int(self.num_actions))
        action = np.random.choice(self.num_actions, p = probs)

        return action
    
    def act_greedy(self, table, state):
        "Act greedy w.r.t. the state-action value, used to run validation experiments"
        action = np.argmax(table[state])
        return action

    def learned(self, table):
        "Saving learned values for validation"
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
    def __init__(self, movements, num_actions, modulus, learning_algo, state_UB, lr = 1, disc1 = 0.85, disc2 = 0.9):
        super().__init__(movements, num_actions, modulus, learning_algo, disc1 = disc1, lr = 1)
        self.state_UB = state_UB
        self.disc2 = disc2
        
        fs = [str(i) for i in modulus]
        decimals = [f[::-1].find('.') for f in fs]

        holder = {i:[] for i in range(len(state_UB))} #one key for every state variable, i.e. variables in the states will be indexed as were inputed
        for i in range(len(state_UB)): #create the unidimensional mesh grid for every state variable
            holder[i] = np.linspace(0, int(self.state_UB[i]), int(self.state_UB[i]/self.modulus[i] + 1), dtype = np.float64)
            holder[i] = np.round(holder[i], decimals[i])
        discrete_states = tuple([v for v in holder.values()]) #tuple with the arrays of mesh grids
        p = list(product(*discrete_states, range(1, int(self.movements) + 1))) #list of outer product of all the discrete states from the mesh
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
        for i in range(self.movements[0] - 1, -1, -1):
            idx = action[i]
            Gt = 0
            self.dcount[tuple(state[i])][int(idx)] += 1
            if i < 1:
                Gt += self.disc2 * reward[-1] 
            else:
                for j in range(1, i + 2):
                    Gt = reward[-j] + self.disc1 * Gt

            alpha = self.lr / self.dcount[tuple(state[i])][int(idx)]              # updating learning parameter with number of visits to current state
            V = self.d[tuple(state[i])][int(idx)]
            self.d[tuple(state[i])][int(idx)] = (1-alpha) * V + alpha * Gt
        return

    def First_Visit(self, state, action, reward):
        r"""
        NOT UPDATED FOR THE REWRITTEN CODE
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


class Q_Learning(Agent):
    """
    Off-policy control, Temporal difference learning i.e. Q-Learning algorithm
    """
    def __init__(self, movements, num_actions, modulus, learning_algo, state_UB, lr = 1, disc1 = 0.85, disc2 = 0.9):
        super().__init__(movements, num_actions, modulus, learning_algo, disc1 = disc1, lr = 1)
        self.state_UB = state_UB
        self.disc2 = disc2

        fs = [str(i) for i in modulus]
        decimals = [f[::-1].find('.') for f in fs]
        holder = {i:[] for i in range(len(state_UB))}                                             #one key for every state variable, i.e. variables in the states will be indexed as were inputed
        for i in range(len(state_UB)):                                                            #create the unidimensional mesh grid for every state variable
            holder[i] = np.linspace(0, int(self.state_UB[i]), int(self.state_UB[i]/self.modulus[i] + 1), dtype = np.float64)
            holder[i] = np.round(holder[i], decimals[i])
        discrete_states = tuple([v for v in holder.values()])                                     #tuple with the arrays of mesh grids
        p = list(product(*discrete_states, range(1, int(self.movements) + 1)))                    #list of outer product of all the discrete states from the mesh
        self.d = {i:np.random.randn(self.num_actions) for i in p}                                 #initialising random values for all the states, hope it works ;)
        self.dcount = {i:np.zeros(self.num_actions, dtype=int) for i in p}

    def Q_max(self, next_state):
        "Select greedily best action given state and return its state-action value"
        best_Q = np.max(self.d[tuple(next_state)])
        return best_Q


        
    def Q_learn(self, state, action, reward):
        r"""
        Off-policy control, Temporal difference learning (TD(0))
        -------
        Params:
            state(tuple): current state of the system
            action(tuple): action(s) taken
            reward(tuple): reward(s) 
        """
        #Loop through the states forward, does it make a difference like in MC?? YES! But why??
        
        for i in range(0, self.movements[0] - 1):
            #Using different gamma in T state
            #For TD target, we don't have to actually TAKE the best action i.e. integrate using the "best" possible action and getting a reward, instead,
            #we take the value as if we had taken the action, and then follow the current policy. Basically off policy learning
            idx = action[i]
            self.dcount[tuple(state[i])][int(idx)] += 1
            next_state =  state[i + 1]
            TD_target = reward[i+1] + self.disc1 * self.Q_max(next_state)
            alpha = self.lr / self.dcount[tuple(state[i])][int(idx)]        
            Q = self.d[tuple(state[i])][int(idx)]
            TD_error = TD_target - Q
            self.d[tuple(state[i])][int(idx)] = Q + alpha * TD_error
        return

    def double_Q(self, state, action, reward):
        r"""
        Double Q learning
        -------
        Params:
            state(tuple): current state of the system
            action(tuple): action(s) taken
            reward(tuple): reward(s) 
        """
        
        print(reward)
        print(state)
        print(action)
        
        for i in range(self.movements[0] - 1, -1, -1):
            #Using different gamma in T state
            #For TD target, we don't have to actually take the best action i.e. integrate using the "best" possible action and getting a reward, we sume that step is done recursively
            if i < self.movements[0] - 1:
                idx = action[i]
                self.dcount[tuple(state[i])][int(idx)] += 1
                next_state =  state[i + 1]
                print("state "  + str(tuple(state[i])))
                print("next state "  + str(next_state))
                print("Action " + str(idx))
                print(reward[-i - 2])
                TD_target = reward[-i - 2] + self.disc1 * self.Q_max(next_state)
                print("TD_target" + str(TD_target))
                alpha = self.lr / self.dcount[tuple(state[i])][int(idx)]              # updating learning parameter with number of visits to current state
        
                Q = self.d[tuple(state[i])][int(idx)]
                print("Q: " + str(Q))
                TD_error = TD_target - Q
                self.d[tuple(state[i])][int(idx)] = Q + alpha * TD_error
                print(Q + alpha * TD_error)
                print(self.d[tuple(state[i])])
                #TD_target = reward[-1] + self.disc2 * self.Q_max(next_state)
        return